from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pyscf import fci
    from pyscf.fci import cistring, direct_spin1, spin_op

    _PYSCF_IMPORT_ERROR = None
except ImportError as exc:
    fci = cistring = direct_spin1 = spin_op = None
    _PYSCF_IMPORT_ERROR = exc

from ...operator.transforms import UnsupportedTransformError
from ...utils.dependencies import require_dependency
from ...utils.solver import coerce_symbolic_operator, normalize_electron_count
from ..base import SolverBackend
from ..capabilities import BackendCapabilities


@dataclass(frozen=True)
class SpinSectorFCIProblem:
    """Exact spin-conserving FCI problem in PySCF's alpha/beta CI basis.

    Design choice:
    for general lattice Hamiltonians, `mbkit` stores already-normal-ordered
    fermionic coefficients. Reconstructing bare PySCF ERI blocks from those
    coefficients is fragile for multi-orbital same-spin interactions. Instead,
    this backend compiles the symbolic operator directly into the chosen
    `(n_up, n_down)` sector and then uses PySCF's CI utilities for density
    matrices and spin observables.
    """

    space: object
    norb: int
    nelec: tuple[int, int]
    constant: complex
    one_body_terms: tuple[tuple[int, int, complex], ...]
    two_body_terms: tuple[tuple[int, int, int, int, complex], ...]
    alpha_strings: tuple[int, ...]
    beta_strings: tuple[int, ...]
    basis_states: tuple[int, ...]
    basis_index: dict[int, int]

    @property
    def num_spin_orbitals(self) -> int:
        return 2 * self.norb

    @property
    def ci_shape(self) -> tuple[int, int]:
        return (len(self.alpha_strings), len(self.beta_strings))


# Backward-compatible aliases for historical imports in tests and examples.
UHFFCIHamiltonian = SpinSectorFCIProblem


def _real_if_close_scalar(value: complex, *, atol: float = 1e-10):
    scalar = complex(value)
    if abs(scalar.imag) <= atol:
        return float(scalar.real)
    return scalar


def _physical_spin_orbital_to_block_mode(space, mode_index: int) -> int:
    mode = space.unpack_mode(mode_index)
    spatial = mode.site * space.num_orbitals_per_site + mode.orbital_index
    if mode.spin == "up":
        return spatial
    if mode.spin == "down":
        return space.num_spatial_orbitals + spatial
    raise UnsupportedTransformError(f"Unknown spin label {mode.spin!r} in PySCF lowering.")


def _count_bits(value: int) -> int:
    return int(value.bit_count())


def _occ_list_to_bits(occ_list) -> int:
    bits = 0
    for orbital in occ_list:
        bits |= 1 << int(orbital)
    return bits


def _apply_annihilation(state: int, mode: int) -> tuple[int, int] | None:
    mask = 1 << int(mode)
    if state & mask == 0:
        return None
    parity = _count_bits(state & (mask - 1))
    sign = -1 if parity % 2 else 1
    return sign, state ^ mask


def _apply_creation(state: int, mode: int) -> tuple[int, int] | None:
    mask = 1 << int(mode)
    if state & mask:
        return None
    parity = _count_bits(state & (mask - 1))
    sign = -1 if parity % 2 else 1
    return sign, state | mask


def _apply_term(state: int, modes: tuple[int, ...], actions: tuple[str, ...]) -> tuple[complex, int] | None:
    sign = 1.0 + 0.0j
    current = int(state)
    for mode, action in zip(reversed(modes), reversed(actions)):
        if action == "destroy":
            step = _apply_annihilation(current, mode)
        elif action == "create":
            step = _apply_creation(current, mode)
        else:
            raise ValueError(f"Unknown ladder action {action!r}.")
        if step is None:
            return None
        step_sign, current = step
        sign *= step_sign
    return sign, current


def _build_sector_basis(norb: int, nelec: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[int, int]]:
    alpha_occ = cistring.gen_occslst(range(norb), nelec[0])
    beta_occ = cistring.gen_occslst(range(norb), nelec[1])

    alpha_strings = tuple(_occ_list_to_bits(occ) for occ in alpha_occ)
    beta_strings = tuple(_occ_list_to_bits(occ) for occ in beta_occ)

    basis_states = []
    basis_index: dict[int, int] = {}
    for ia, alpha_bits in enumerate(alpha_strings):
        for ib, beta_bits in enumerate(beta_strings):
            combined = int(alpha_bits) | (int(beta_bits) << norb)
            index = ia * len(beta_strings) + ib
            basis_states.append(combined)
            basis_index[combined] = index

    return alpha_strings, beta_strings, tuple(basis_states), basis_index


def _compile_sector_fci_problem(hamiltonian, *, n_electrons=None, n_particles=None, atol: float = 1e-10):
    operator = coerce_symbolic_operator(hamiltonian)
    space = operator.space

    if space.num_spins != 2 or tuple(space.spins) != ("up", "down"):
        raise UnsupportedTransformError("PySCFSolver currently requires a two-spin ElectronicSpace.")

    nelec_pair, total_electrons = normalize_electron_count(
        space=space,
        n_electrons=n_electrons,
        n_particles=n_particles,
    )
    if total_electrons > space.num_spin_orbitals:
        raise ValueError(
            f"Electron count {total_electrons} exceeds the {space.num_spin_orbitals} available spin orbitals."
        )

    if any(delta != 0 for delta in operator.particle_number_changes()):
        raise UnsupportedTransformError(
            "PySCFSolver only supports number-conserving one-body and two-body Hamiltonians."
        )
    if any(delta != 0 for delta in operator.spin_changes()):
        raise UnsupportedTransformError(
            "PySCFSolver currently requires Hamiltonians that preserve n_up and n_down."
        )

    norb = space.num_spatial_orbitals
    one_body_terms: list[tuple[int, int, complex]] = []
    two_body_terms: list[tuple[int, int, int, int, complex]] = []

    for term, coeff in operator.iter_terms():
        coeff = complex(coeff)
        if abs(coeff) <= atol:
            continue

        factors = term.factors
        actions = tuple(factor.action for factor in factors)

        if len(factors) == 2 and actions == ("create", "destroy"):
            one_body_terms.append(
                (
                    _physical_spin_orbital_to_block_mode(space, factors[0].mode),
                    _physical_spin_orbital_to_block_mode(space, factors[1].mode),
                    coeff,
                )
            )
            continue

        if len(factors) == 4 and actions == ("create", "create", "destroy", "destroy"):
            two_body_terms.append(
                (
                    _physical_spin_orbital_to_block_mode(space, factors[0].mode),
                    _physical_spin_orbital_to_block_mode(space, factors[1].mode),
                    _physical_spin_orbital_to_block_mode(space, factors[2].mode),
                    _physical_spin_orbital_to_block_mode(space, factors[3].mode),
                    coeff,
                )
            )
            continue

        raise UnsupportedTransformError(
            "PySCFSolver only supports number-conserving one-body and two-body Hamiltonians."
        )

    alpha_strings, beta_strings, basis_states, basis_index = _build_sector_basis(norb, nelec_pair)
    return SpinSectorFCIProblem(
        space=space,
        norb=norb,
        nelec=nelec_pair,
        constant=complex(operator.constant),
        one_body_terms=tuple(one_body_terms),
        two_body_terms=tuple(two_body_terms),
        alpha_strings=alpha_strings,
        beta_strings=beta_strings,
        basis_states=basis_states,
        basis_index=basis_index,
    )


def _build_sector_hamiltonian_matrix(problem: SpinSectorFCIProblem) -> np.ndarray:
    dimension = len(problem.basis_states)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    if abs(problem.constant) > 0.0:
        matrix[np.diag_indices(dimension)] += problem.constant

    for column, state in enumerate(problem.basis_states):
        for p, q, coeff in problem.one_body_terms:
            step = _apply_term(state, (p, q), ("create", "destroy"))
            if step is None:
                continue
            sign, new_state = step
            row = problem.basis_index.get(new_state)
            if row is None:
                continue
            matrix[row, column] += coeff * sign

        for p, q, r, s, coeff in problem.two_body_terms:
            step = _apply_term(state, (p, q, r, s), ("create", "create", "destroy", "destroy"))
            if step is None:
                continue
            sign, new_state = step
            row = problem.basis_index.get(new_state)
            if row is None:
                continue
            matrix[row, column] += coeff * sign

    return matrix


def _expectation_compiled(problem: SpinSectorFCIProblem, state_vector: np.ndarray, compiled: SpinSectorFCIProblem):
    value = complex(compiled.constant)
    for column, state in enumerate(problem.basis_states):
        amplitude = complex(state_vector[column])
        if abs(amplitude) <= 1e-15:
            continue

        for p, q, coeff in compiled.one_body_terms:
            step = _apply_term(state, (p, q), ("create", "destroy"))
            if step is None:
                continue
            sign, new_state = step
            row = problem.basis_index.get(new_state)
            if row is None:
                continue
            value += np.conjugate(state_vector[row]) * coeff * sign * amplitude

        for p, q, r, s, coeff in compiled.two_body_terms:
            step = _apply_term(state, (p, q, r, s), ("create", "create", "destroy", "destroy"))
            if step is None:
                continue
            sign, new_state = step
            row = problem.basis_index.get(new_state)
            if row is None:
                continue
            value += np.conjugate(state_vector[row]) * coeff * sign * amplitude

    return value


def _block_mode_to_interleaved(problem: SpinSectorFCIProblem, mode: int) -> int:
    if mode < problem.norb:
        return 2 * mode
    return 2 * (mode - problem.norb) + 1


def _build_uhf_fci_hamiltonian(hamiltonian, *, n_electrons=None, n_particles=None, atol: float = 1e-10):
    """Backward-compatible alias for the sector FCI compiler.

    Earlier revisions tried to repack `mbkit` operators into PySCF UHF ERI
    blocks. The current implementation instead compiles the exact symbolic
    Hamiltonian directly into the requested `(n_up, n_down)` CI sector, which is
    the robust path for general lattice-model two-body tensors.
    """

    return _compile_sector_fci_problem(
        hamiltonian,
        n_electrons=n_electrons,
        n_particles=n_particles,
        atol=atol,
    )


class PySCFFCIBackend(SolverBackend):
    """Concrete PySCF-backed exact FCI solver for spin-conserving Hamiltonians."""

    solver_family = "qc"
    backend_name = "pyscf_fci"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_rdm1=True,
        supports_native_expectation=False,
        supports_complex=True,
        preferred_problem_type="qc",
    )

    def __init__(self, *args, **kwargs) -> None:
        require_dependency("PySCFSolver", "pyscf", _PYSCF_IMPORT_ERROR)
        self.method = kwargs.pop("method", "fci")
        self.conv_tol = kwargs.pop("conv_tol", 1e-10)
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.atol = kwargs.pop("atol", 1e-10)
        self.legacy_kwargs = dict(kwargs)

        self.problem = None
        self.hamiltonian_matrix = None
        self.energy_value = None
        self.ground_state = None
        self.ci_vector = None
        self.fci_solver = fci.direct_uhf.FCISolver()

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        if self.method != "fci":
            raise NotImplementedError("PySCFSolver currently supports only method='fci'.")

        self.problem = _compile_sector_fci_problem(
            hamiltonian,
            n_electrons=n_electrons,
            n_particles=n_particles,
            atol=self.atol,
        )
        self.hamiltonian_matrix = _build_sector_hamiltonian_matrix(self.problem)

        if not np.allclose(self.hamiltonian_matrix, self.hamiltonian_matrix.conj().T, atol=self.atol):
            raise UnsupportedTransformError(
                "PySCFSolver requires a Hermitian Hamiltonian after symbolic compilation."
            )

        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian_matrix)
        self.energy_value = _real_if_close_scalar(eigenvalues[0], atol=self.atol)
        self.ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
        self.ci_vector = self.ground_state.reshape(self.problem.ci_shape).view(direct_spin1.FCIvector)
        return self

    def _require_solution(self) -> None:
        if self.problem is None or self.ground_state is None or self.ci_vector is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def energy(self):
        self._require_solution()
        return self.energy_value

    def rdm1(self):
        self._require_solution()
        nmode = self.problem.num_spin_orbitals
        block_rdm = np.zeros((nmode, nmode), dtype=np.complex128)
        for p in range(nmode):
            for q in range(nmode):
                compiled = SpinSectorFCIProblem(
                    space=self.problem.space,
                    norb=self.problem.norb,
                    nelec=self.problem.nelec,
                    constant=0.0,
                    one_body_terms=((p, q, 1.0),),
                    two_body_terms=(),
                    alpha_strings=self.problem.alpha_strings,
                    beta_strings=self.problem.beta_strings,
                    basis_states=self.problem.basis_states,
                    basis_index=self.problem.basis_index,
                )
                block_rdm[p, q] = _expectation_compiled(self.problem, self.ground_state, compiled)

        rdm = np.zeros_like(block_rdm)
        for p in range(nmode):
            for q in range(nmode):
                rdm[_block_mode_to_interleaved(self.problem, p), _block_mode_to_interleaved(self.problem, q)] = block_rdm[p, q]
        return np.real_if_close(rdm)

    def docc(self):
        self._require_solution()
        weights = np.abs(self.ground_state) ** 2
        docc = np.zeros(self.problem.norb, dtype=float)
        for index, state in enumerate(self.problem.basis_states):
            weight = float(weights[index])
            if weight <= 1e-15:
                continue
            for orbital in range(self.problem.norb):
                alpha_mask = 1 << orbital
                beta_mask = 1 << (self.problem.norb + orbital)
                if (state & alpha_mask) and (state & beta_mask):
                    docc[orbital] += weight
        return docc

    def s2(self):
        self._require_solution()
        return _real_if_close_scalar(
            _expectation_compiled(
                self.problem,
                self.ground_state,
                _compile_sector_fci_problem(
                    self.problem.space.spin_squared_term(),
                    n_particles=self.problem.nelec,
                    atol=self.atol,
                ),
            ),
            atol=self.atol,
        )
