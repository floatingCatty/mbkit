from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pyscf import fci
    from pyscf.fci import spin_op

    _PYSCF_IMPORT_ERROR = None
except ImportError as exc:
    fci = spin_op = None
    _PYSCF_IMPORT_ERROR = exc

from ..operator.transforms import UnsupportedTransformError
from ._frontend import coerce_electronic_integrals, normalize_electron_count
from ._optional import require_dependency


@dataclass(frozen=True)
class UHFFCIHamiltonian:
    space: object
    norb: int
    nelec: tuple[int, int]
    h1e: tuple[np.ndarray, np.ndarray]
    eri: tuple[np.ndarray, np.ndarray, np.ndarray]
    ecore: float


def _real_scalar(value, *, name: str, atol: float = 1e-10) -> float:
    scalar = complex(value)
    if abs(scalar.imag) > atol:
        raise UnsupportedTransformError(f"PySCFSolver requires a real {name}; got {value!r}.")
    return float(scalar.real)


def _reorder_pair_by_spin(modes, desired):
    current = tuple(mode.spin for mode in modes)
    if current == desired:
        return 1.0, modes
    if current == tuple(reversed(desired)):
        return -1.0, (modes[1], modes[0])
    raise UnsupportedTransformError(
        f"PySCFSolver does not support the spin pattern {current!r} in this tensor block."
    )


def _build_uhf_fci_hamiltonian(hamiltonian, *, n_electrons=None, n_particles=None, atol: float = 1e-10):
    integrals = coerce_electronic_integrals(hamiltonian)
    operator = integrals.to_operator(atol=atol)
    space = operator.space

    if space.num_spins != 2 or tuple(space.spins) != ("up", "down"):
        raise UnsupportedTransformError("PySCFSolver currently requires a two-spin ElectronicSpace.")

    nelec_pair, total_electrons = normalize_electron_count(
        n_electrons=n_electrons,
        n_particles=n_particles,
    )
    if total_electrons > space.num_spin_orbitals:
        raise ValueError(
            f"Electron count {total_electrons} exceeds the {space.num_spin_orbitals} available spin orbitals."
        )

    norb = space.num_spatial_orbitals
    h1e_a = np.zeros((norb, norb), dtype=float)
    h1e_b = np.zeros((norb, norb), dtype=float)
    eri_aa = np.zeros((norb, norb, norb, norb), dtype=float)
    eri_ab = np.zeros((norb, norb, norb, norb), dtype=float)
    eri_bb = np.zeros((norb, norb, norb, norb), dtype=float)

    for term, coeff in operator.iter_terms():
        coeff_real = _real_scalar(coeff, name="coefficient", atol=atol)
        actions = tuple(factor.action for factor in term.factors)

        if len(term.factors) == 2 and actions == ("create", "destroy"):
            left = space.unpack_mode(term.factors[0].mode)
            right = space.unpack_mode(term.factors[1].mode)
            if left.spin != right.spin:
                raise UnsupportedTransformError(
                    "PySCFSolver does not support spin-mixing one-body terms."
                )
            left_spatial = left.index // 2
            right_spatial = right.index // 2
            if left.spin == "up":
                h1e_a[left_spatial, right_spatial] += coeff_real
            else:
                h1e_b[left_spatial, right_spatial] += coeff_real
            continue

        if len(term.factors) == 4 and actions == ("create", "create", "destroy", "destroy"):
            creators = tuple(space.unpack_mode(factor.mode) for factor in term.factors[:2])
            annihilators = tuple(space.unpack_mode(factor.mode) for factor in term.factors[2:])
            creator_spins = tuple(mode.spin for mode in creators)
            annihilator_spins = tuple(mode.spin for mode in annihilators)

            if creator_spins == ("up", "up") and annihilator_spins == ("up", "up"):
                p = creators[0].index // 2
                r = creators[1].index // 2
                s = annihilators[0].index // 2
                q = annihilators[1].index // 2
                eri_aa[p, q, r, s] += 2.0 * coeff_real
                continue

            if creator_spins == ("down", "down") and annihilator_spins == ("down", "down"):
                p = creators[0].index // 2
                r = creators[1].index // 2
                s = annihilators[0].index // 2
                q = annihilators[1].index // 2
                eri_bb[p, q, r, s] += 2.0 * coeff_real
                continue

            if sorted(creator_spins) == ["down", "up"] and sorted(annihilator_spins) == ["down", "up"]:
                sign_c, creators = _reorder_pair_by_spin(creators, ("up", "down"))
                sign_a, annihilators = _reorder_pair_by_spin(annihilators, ("down", "up"))
                p = creators[0].index // 2
                r = creators[1].index // 2
                s = annihilators[0].index // 2
                q = annihilators[1].index // 2
                eri_ab[p, q, r, s] += sign_c * sign_a * coeff_real
                continue

            raise UnsupportedTransformError(
                "PySCFSolver only supports spin-conserving same-spin and alpha-beta two-body terms."
            )

        raise UnsupportedTransformError(
            "PySCFSolver only supports number-conserving one-body and two-body Hamiltonians."
        )

    ecore = _real_scalar(operator.constant, name="constant shift", atol=atol)
    return UHFFCIHamiltonian(
        space=space,
        norb=norb,
        nelec=nelec_pair,
        h1e=(h1e_a, h1e_b),
        eri=(eri_aa, eri_ab, eri_bb),
        ecore=ecore,
    )


class PySCFSolver:
    """Direct PySCF FCI solver for validated UHF-compatible Hamiltonians."""

    def __init__(self, *args, **kwargs) -> None:
        require_dependency("PySCFSolver", "pyscf", _PYSCF_IMPORT_ERROR)
        self.method = kwargs.pop("method", "fci")
        self.conv_tol = kwargs.pop("conv_tol", 1e-10)
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.atol = kwargs.pop("atol", 1e-10)
        self.legacy_kwargs = dict(kwargs)

        self.problem = None
        self.fci_solver = None
        self.ci_vector = None
        self.energy_value = None

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        if self.method != "fci":
            raise NotImplementedError("PySCFSolver currently supports only method='fci'.")

        self.problem = _build_uhf_fci_hamiltonian(
            hamiltonian,
            n_electrons=n_electrons,
            n_particles=n_particles,
            atol=self.atol,
        )
        self.fci_solver = fci.direct_uhf.FCISolver()
        self.fci_solver.conv_tol = self.conv_tol
        self.fci_solver.max_cycle = self.max_cycle

        self.energy_value, self.ci_vector = self.fci_solver.kernel(
            self.problem.h1e,
            self.problem.eri,
            self.problem.norb,
            self.problem.nelec,
            ecore=self.problem.ecore,
        )
        return self

    def _require_solution(self) -> None:
        if self.problem is None or self.fci_solver is None or self.ci_vector is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def energy(self):
        self._require_solution()
        return self.energy_value

    def rdm1(self):
        self._require_solution()
        dm1a, dm1b = self.fci_solver.make_rdm1s(
            self.ci_vector,
            self.problem.norb,
            self.problem.nelec,
        )
        rdm = np.zeros((self.problem.space.num_spin_orbitals, self.problem.space.num_spin_orbitals), dtype=float)
        rdm[0::2, 0::2] = dm1a
        rdm[1::2, 1::2] = dm1b
        return rdm

    def docc(self):
        self._require_solution()
        (_, _), (_, dm2ab, _) = self.fci_solver.make_rdm12s(
            self.ci_vector,
            self.problem.norb,
            self.problem.nelec,
        )
        return np.asarray([dm2ab[i, i, i, i] for i in range(self.problem.norb)], dtype=float)

    def s2(self):
        self._require_solution()
        return spin_op.spin_square(
            self.ci_vector,
            self.problem.norb,
            self.problem.nelec,
            fcisolver=self.fci_solver,
        )[0]


PYSCF_solver = PySCFSolver
