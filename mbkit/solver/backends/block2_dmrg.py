from __future__ import annotations

import numpy as np

try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    _PYBLOCK2_IMPORT_ERROR = None
except ImportError as exc:
    DMRGDriver = None
    SymmetryTypes = None
    _PYBLOCK2_IMPORT_ERROR = exc

from ...operator import to_qc_tensors
from ...utils.dependencies import require_dependency
from ...utils.solver import normalize_electron_count
from ..base import SolverBackend
from ..capabilities import BackendCapabilities
from ..compile import compile_qc_hamiltonian


def _real_scalar(value, *, name: str, atol: float = 1e-10) -> float:
    scalar = complex(value)
    if abs(scalar.imag) > atol:
        raise ValueError(f"DMRGSolver requires a real {name}; got {value!r}.")
    return float(scalar.real)


def _as_driver_array(array: np.ndarray, *, iscomplex: bool, name: str, atol: float = 1e-10) -> np.ndarray:
    arr = np.asarray(array)
    if iscomplex:
        return np.asarray(arr, dtype=np.complex128)
    if np.max(np.abs(np.imag(arr))) > atol:
        raise ValueError(f"DMRGSolver received a complex {name} but `iscomplex=False`.")
    return np.asarray(np.real(arr), dtype=float)


def _compile_direct_dmrg_hamiltonian(hamiltonian):
    return compile_qc_hamiltonian(hamiltonian)


class Block2DMRGBackend(SolverBackend):
    """Concrete DMRG backend implemented with block2 / pyblock2."""

    solver_family = "dmrg"
    backend_name = "block2"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_rdm1=True,
        supports_native_expectation=True,
        supports_complex=True,
        preferred_problem_type="qc",
    )

    def __init__(self, *args, **kwargs) -> None:
        require_dependency("DMRGSolver", "dmrg", _PYBLOCK2_IMPORT_ERROR)
        self.iscomplex = kwargs.pop("iscomplex", False)
        self.scratch_dir = kwargs.pop("scratch_dir", "/tmp/dmrg_tmp")
        self.n_threads = kwargs.pop("n_threads", 1)
        self.mpi = kwargs.pop("mpi", False)
        self.bond_dim = kwargs.pop("bond_dim", 250)
        self.bond_mul = kwargs.pop("bond_mul", 2)
        self.n_sweep = kwargs.pop("n_sweep", 20)
        self.nupdate = kwargs.pop("nupdate", 4)
        self.iprint = kwargs.pop("iprint", 0)
        self.reorder = kwargs.pop("reorder", False)
        self.eig_cutoff = kwargs.pop("eig_cutoff", 1e-7)
        self.legacy_kwargs = dict(kwargs)

        self.driver = None
        self.mpo = None
        self.ket = None
        self.energy_value = None
        self.compiled = None
        self.space = None
        self.n_electrons = None

    def _initialize_driver(self, *, n_spin_orbitals: int, n_electrons: int) -> None:
        symm_type = SymmetryTypes.SGFCPX if self.iscomplex else SymmetryTypes.SGF
        # the symm_type need to be reconsider for efficiency. If the system is spin degenerated, 
        # using more symmetry sectors would be better to ustilize it and reduce the bond dimension.
        self.driver = DMRGDriver(
            scratch=self.scratch_dir,
            symm_type=symm_type,
            n_threads=self.n_threads,
            mpi=self.mpi,
            stack_mem=5368709120,
        )
        self.driver.initialize_system(n_sites=n_spin_orbitals, n_elec=n_electrons)

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        compiled = _compile_direct_dmrg_hamiltonian(hamiltonian)
        (_, total_electrons) = normalize_electron_count(
            space=compiled.space,
            n_electrons=n_electrons,
            n_particles=n_particles,
        )

        self.compiled = compiled
        self.space = compiled.space
        self.n_electrons = total_electrons

        h1e = _as_driver_array(compiled.h1e, iscomplex=self.iscomplex, name="one-body tensor")
        g2e = _as_driver_array(compiled.g2e, iscomplex=self.iscomplex, name="two-body tensor")
        ecore = compiled.constant_shift if self.iscomplex else _real_scalar(
            compiled.constant_shift,
            name="constant shift",
        )

        self._initialize_driver(
            n_spin_orbitals=compiled.space.num_spin_orbitals,
            n_electrons=total_electrons,
        )

        reorder_idx = None
        if self.reorder:
            reorder_idx = self.driver.orbital_reordering(h1e=h1e, g2e=g2e, method="gaopt")

        self.mpo = self.driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            iprint=0,
            reorder=reorder_idx,
        )
        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=self.bond_dim, nroots=1)

        bond_dims = [self.bond_dim] * self.nupdate + [self.bond_mul * self.bond_dim] * self.nupdate
        noises = [1e-4] * self.nupdate + [1e-5] * self.nupdate + [0]
        thrds = [1e-5] * self.nupdate + [1e-7] * self.nupdate

        result = self.driver.dmrg(
            self.mpo,
            self.ket,
            n_sweeps=self.n_sweep,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            cutoff=self.eig_cutoff,
            iprint=self.iprint,
            cached_contraction=True,
            tol=1e-6,
        )
        if result is None:
            result = self.driver.expectation(self.ket, self.mpo, self.ket)
        if isinstance(result, (tuple, list, np.ndarray)):
            result = np.asarray(result).reshape(-1)[0]
        self.energy_value = result
        return self

    def _require_solution(self) -> None:
        if self.driver is None or self.ket is None or self.mpo is None or self.space is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def _expectation(self, operator):
        compiled = to_qc_tensors(operator)
        h1e = _as_driver_array(compiled.h1e, iscomplex=self.iscomplex, name="observable one-body tensor")
        g2e = _as_driver_array(compiled.g2e, iscomplex=self.iscomplex, name="observable two-body tensor")
        ecore = compiled.constant_shift if self.iscomplex else _real_scalar(
            compiled.constant_shift,
            name="observable constant shift",
        )
        mpo = self.driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            iprint=0,
            reorder=self.driver.reorder_idx,
        )
        return self.driver.expectation(self.ket, mpo, self.ket)

    def energy(self):
        self._require_solution()
        if self.energy_value is None:
            self.energy_value = self.driver.expectation(self.ket, self.mpo, self.ket)
        return self.energy_value

    def rdm1(self):
        self._require_solution()
        return np.asarray(self.driver.get_1pdm(self.ket))

    def docc(self):
        self._require_solution()
        values = []
        for site in range(self.space.num_sites):
            for orbital in self.space.orbitals:
                operator = (
                    self.space.number(site, orbital=orbital, spin="up")
                    @ self.space.number(site, orbital=orbital, spin="down")
                )
                values.append(np.real_if_close(self._expectation(operator)).item())
        return np.asarray(values, dtype=float)

    def s2(self):
        self._require_solution()
        return self._expectation(self.space.spin_squared_term())

