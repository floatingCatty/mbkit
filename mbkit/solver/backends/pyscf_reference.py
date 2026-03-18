from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pyscf import ao2mo, gto, mp, scf

    _PYSCF_IMPORT_ERROR = None
except ImportError as exc:
    ao2mo = gto = mp = scf = None
    _PYSCF_IMPORT_ERROR = exc

from ...operator.transforms import UnsupportedTransformError
from ...utils.dependencies import require_dependency
from ...utils.solver import coerce_symbolic_operator, normalize_electron_count
from ..base import SolverBackend
from ..capabilities import BackendCapabilities


@dataclass(frozen=True)
class PySCFReferenceProblem:
    """Mean-field-compatible PySCF problem extracted from a symbolic Hamiltonian.

    This backend intentionally supports a narrower class than the exact FCI path:
    the two-body interaction must be representable by one spin-independent
    spatial-orbital ERI tensor suitable for UHF/UMP2-style methods.
    """

    space: object
    norb: int
    nelec: tuple[int, int]
    h1e: tuple[np.ndarray, np.ndarray]
    eri: np.ndarray
    ecore: float


def _real_scalar(value, *, name: str, atol: float = 1e-10) -> float:
    scalar = complex(value)
    if abs(scalar.imag) > atol:
        raise UnsupportedTransformError(f"PySCF reference methods require a real {name}; got {value!r}.")
    return float(scalar.real)


def _reorder_pair_by_spin(modes, desired):
    current = tuple(mode.spin for mode in modes)
    if current == desired:
        return 1.0, modes
    if current == tuple(reversed(desired)):
        return -1.0, (modes[1], modes[0])
    raise UnsupportedTransformError(
        f"PySCF reference methods do not support the spin pattern {current!r} in this tensor block."
    )


def _spatial_index(mode) -> int:
    return mode.site * mode.space.num_orbitals_per_site + mode.orbital_index if hasattr(mode, "space") else mode.index // 2


def _mode_spatial(space, mode_index: int) -> int:
    mode = space.unpack_mode(mode_index)
    return mode.site * space.num_orbitals_per_site + mode.orbital_index


def _build_pyscf_reference_problem(hamiltonian, *, n_electrons=None, n_particles=None, atol: float = 1e-10):
    operator = coerce_symbolic_operator(hamiltonian)
    space = operator.space

    if space.num_spins != 2 or tuple(space.spins) != ("up", "down"):
        raise UnsupportedTransformError("PySCF reference methods currently require a two-spin ElectronicSpace.")

    if any(delta != 0 for delta in operator.particle_number_changes()):
        raise UnsupportedTransformError(
            "PySCF reference methods only support number-conserving Hamiltonians."
        )
    if any(delta != 0 for delta in operator.spin_changes()):
        raise UnsupportedTransformError(
            "PySCF reference methods require Hamiltonians that preserve n_up and n_down."
        )

    nelec_pair, total_electrons = normalize_electron_count(
        space=space,
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
    eri = np.zeros((norb, norb, norb, norb), dtype=float)
    same_spin_terms: list[tuple[str, int, int, int, int, float]] = []

    for term, coeff in operator.iter_terms():
        coeff_real = _real_scalar(coeff, name="coefficient", atol=atol)
        actions = tuple(factor.action for factor in term.factors)

        if len(term.factors) == 2 and actions == ("create", "destroy"):
            left = space.unpack_mode(term.factors[0].mode)
            right = space.unpack_mode(term.factors[1].mode)
            if left.spin != right.spin:
                raise UnsupportedTransformError(
                    "PySCF reference methods do not support spin-mixing one-body terms."
                )
            left_spatial = _mode_spatial(space, term.factors[0].mode)
            right_spatial = _mode_spatial(space, term.factors[1].mode)
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

            if sorted(creator_spins) == ["down", "up"] and sorted(annihilator_spins) == ["down", "up"]:
                sign_c, creators = _reorder_pair_by_spin(creators, ("up", "down"))
                sign_a, annihilators = _reorder_pair_by_spin(annihilators, ("down", "up"))
                p = creators[0].site * space.num_orbitals_per_site + creators[0].orbital_index
                r = creators[1].site * space.num_orbitals_per_site + creators[1].orbital_index
                s = annihilators[0].site * space.num_orbitals_per_site + annihilators[0].orbital_index
                q = annihilators[1].site * space.num_orbitals_per_site + annihilators[1].orbital_index
                eri[p, q, r, s] += sign_c * sign_a * coeff_real
                continue

            if creator_spins == ("up", "up") and annihilator_spins == ("up", "up"):
                p = creators[0].site * space.num_orbitals_per_site + creators[0].orbital_index
                q = creators[1].site * space.num_orbitals_per_site + creators[1].orbital_index
                r = annihilators[0].site * space.num_orbitals_per_site + annihilators[0].orbital_index
                s = annihilators[1].site * space.num_orbitals_per_site + annihilators[1].orbital_index
                same_spin_terms.append(("up", p, q, r, s, coeff_real))
                continue

            if creator_spins == ("down", "down") and annihilator_spins == ("down", "down"):
                p = creators[0].site * space.num_orbitals_per_site + creators[0].orbital_index
                q = creators[1].site * space.num_orbitals_per_site + creators[1].orbital_index
                r = annihilators[0].site * space.num_orbitals_per_site + annihilators[0].orbital_index
                s = annihilators[1].site * space.num_orbitals_per_site + annihilators[1].orbital_index
                same_spin_terms.append(("down", p, q, r, s, coeff_real))
                continue

            raise UnsupportedTransformError(
                "PySCF reference methods only support spin-conserving one-body terms and "
                "two-body terms representable by a common spatial-orbital ERI tensor."
            )

        raise UnsupportedTransformError(
            "PySCF reference methods only support number-conserving one-body and two-body Hamiltonians."
        )

    for spin, p, q, r, s, coeff in same_spin_terms:
        predicted = eri[q, r, p, s] - eri[p, r, q, s]
        if abs(predicted - coeff) > atol:
            raise UnsupportedTransformError(
                "PySCF reference methods require the two-body tensor to be compatible "
                "with a spin-independent spatial-orbital ERI representation."
            )

    ecore = _real_scalar(operator.constant, name="constant shift", atol=atol)
    return PySCFReferenceProblem(
        space=space,
        norb=norb,
        nelec=nelec_pair,
        h1e=(h1e_a, h1e_b),
        eri=eri,
        ecore=ecore,
    )


def _interleaved_rdm1_from_spin_blocks(space, dm1a: np.ndarray, dm1b: np.ndarray) -> np.ndarray:
    dtype = np.result_type(dm1a.dtype, dm1b.dtype)
    rdm = np.zeros((space.num_spin_orbitals, space.num_spin_orbitals), dtype=dtype)
    rdm[0::2, 0::2] = dm1a
    rdm[1::2, 1::2] = dm1b
    return np.real_if_close(rdm)


def _build_uhf_reference(problem: PySCFReferenceProblem, *, conv_tol: float, max_cycle: int):
    mol = gto.M(verbose=0)
    mol.nelectron = sum(problem.nelec)
    mol.spin = problem.nelec[0] - problem.nelec[1]
    mol.incore_anyway = True
    mol.nao_nr = lambda *args, **kwargs: problem.norb

    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.chkfile = None
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.init_guess = "1e"
    mf.get_hcore = lambda *args, **kwargs: problem.h1e
    mf.get_ovlp = lambda *args, **kwargs: np.eye(problem.norb)
    mf._eri = ao2mo.restore(8, problem.eri, problem.norb)
    mf.kernel()
    return mf


class PySCFReferenceBackend(SolverBackend):
    """PySCF backend for UHF-based approximate many-body methods.

    The stable public API routes `PySCFSolver(method="uhf")` and
    `PySCFSolver(method="mp2")` through this backend.
    """

    solver_family = "qc"
    backend_name = "pyscf_reference"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_rdm1=True,
        supports_native_expectation=False,
        supports_complex=False,
        preferred_problem_type="qc",
    )

    def __init__(self, *args, **kwargs) -> None:
        require_dependency("PySCFSolver", "pyscf", _PYSCF_IMPORT_ERROR)
        self.method = kwargs.pop("method", "uhf")
        self.conv_tol = kwargs.pop("conv_tol", 1e-10)
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.atol = kwargs.pop("atol", 1e-10)
        self.legacy_kwargs = dict(kwargs)

        self.problem = None
        self.reference = None
        self.post_hf = None
        self.energy_value = None

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        if self.method not in {"uhf", "mp2"}:
            raise NotImplementedError(
                "PySCF reference backend currently supports only method='uhf' or method='mp2'."
            )

        self.problem = _build_pyscf_reference_problem(
            hamiltonian,
            n_electrons=n_electrons,
            n_particles=n_particles,
            atol=self.atol,
        )
        self.reference = _build_uhf_reference(
            self.problem,
            conv_tol=self.conv_tol,
            max_cycle=self.max_cycle,
        )

        if self.method == "uhf":
            self.post_hf = None
            self.energy_value = float(self.reference.e_tot + self.problem.ecore)
            return self

        self.post_hf = mp.UMP2(self.reference)
        self.post_hf.verbose = 0
        corr_energy, _ = self.post_hf.kernel()
        self.energy_value = float(self.reference.e_tot + corr_energy + self.problem.ecore)
        return self

    def _require_solution(self) -> None:
        if self.problem is None or self.reference is None or self.energy_value is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def energy(self):
        self._require_solution()
        return self.energy_value

    def rdm1(self):
        self._require_solution()
        if self.post_hf is None:
            dm1a, dm1b = self.reference.make_rdm1()
        else:
            dm1 = self.post_hf.make_rdm1()
            if isinstance(dm1, tuple):
                dm1a, dm1b = dm1
            else:
                dm1a, dm1b = dm1[0], dm1[1]
        return _interleaved_rdm1_from_spin_blocks(self.problem.space, np.asarray(dm1a), np.asarray(dm1b))

    def docc(self):
        self._require_solution()
        rdm1 = np.asarray(self.rdm1())
        return np.asarray(
            [rdm1[2 * orbital, 2 * orbital] * rdm1[2 * orbital + 1, 2 * orbital + 1] for orbital in range(self.problem.norb)],
            dtype=float,
        )

    def s2(self):
        self._require_solution()
        target = self.post_hf if self.post_hf is not None else self.reference
        if hasattr(target, "spin_square"):
            value = target.spin_square()
            if isinstance(value, tuple):
                return float(value[0])
            return float(value)
        return float(self.reference.spin_square()[0])
