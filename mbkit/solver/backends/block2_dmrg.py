from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    _PYBLOCK2_IMPORT_ERROR = None
except ImportError as exc:
    DMRGDriver = None
    SymmetryTypes = None
    _PYBLOCK2_IMPORT_ERROR = exc

from ...operator import to_qc_tensors
from ...operator.transforms import UnsupportedTransformError
from ...utils.dependencies import require_dependency
from ...utils.solver import coerce_symbolic_operator, normalize_electron_count
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


@dataclass(frozen=True)
class _Block2SpatialProblem:
    space: object
    norb: int
    h1e_alpha: np.ndarray
    h1e_beta: np.ndarray
    eri: np.ndarray
    constant_shift: complex


@dataclass(frozen=True)
class _Block2PreparedProblem:
    mode: str
    n_sites: int
    n_electrons: int
    sector: tuple[int, int]
    spin: int
    h1e: object
    g2e: object
    ecore: object
    target_spin_twos: int | None
    target_sz_twos: int | None


def _mode_spatial(space, mode_index: int) -> int:
    mode = space.unpack_mode(mode_index)
    return mode.site * space.num_orbitals_per_site + mode.orbital_index


def _reorder_pair_by_spin(modes, desired):
    current = tuple(mode.spin for mode in modes)
    if current == desired:
        return 1.0, modes
    if current == tuple(reversed(desired)):
        return -1.0, (modes[1], modes[0])
    raise UnsupportedTransformError(
        f"block2 spatial symmetry modes do not support the spin pattern {current!r} in this tensor block."
    )


def _build_block2_spatial_problem(hamiltonian, *, atol: float = 1e-10) -> _Block2SpatialProblem:
    operator = coerce_symbolic_operator(hamiltonian)
    space = operator.space

    if space.num_spins != 2 or tuple(space.spins) != ("up", "down"):
        raise UnsupportedTransformError(
            "block2 spatial symmetry modes currently require a two-spin ElectronicSpace."
        )

    if any(delta != 0 for delta in operator.particle_number_changes()):
        raise UnsupportedTransformError(
            "block2 spatial symmetry modes only support number-conserving Hamiltonians."
        )
    if any(delta != 0 for delta in operator.spin_changes()):
        raise UnsupportedTransformError(
            "block2 spatial symmetry modes require Hamiltonians that preserve n_up and n_down."
        )

    norb = space.num_spatial_orbitals
    h1e_alpha = np.zeros((norb, norb), dtype=np.complex128)
    h1e_beta = np.zeros((norb, norb), dtype=np.complex128)
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    same_spin_terms: list[tuple[str, int, int, int, int, complex]] = []

    for term, coeff in operator.iter_terms():
        coeff = complex(coeff)
        if abs(coeff) <= atol:
            continue

        actions = tuple(factor.action for factor in term.factors)

        if len(term.factors) == 2 and actions == ("create", "destroy"):
            left = space.unpack_mode(term.factors[0].mode)
            right = space.unpack_mode(term.factors[1].mode)
            if left.spin != right.spin:
                raise UnsupportedTransformError(
                    "block2 SZ/SU2 modes do not support spin-mixing one-body terms."
                )
            left_spatial = _mode_spatial(space, term.factors[0].mode)
            right_spatial = _mode_spatial(space, term.factors[1].mode)
            if left.spin == "up":
                h1e_alpha[left_spatial, right_spatial] += coeff
            else:
                h1e_beta[left_spatial, right_spatial] += coeff
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
                eri[p, q, r, s] += sign_c * sign_a * coeff
                continue

            if creator_spins == ("up", "up") and annihilator_spins == ("up", "up"):
                p = creators[0].site * space.num_orbitals_per_site + creators[0].orbital_index
                q = creators[1].site * space.num_orbitals_per_site + creators[1].orbital_index
                r = annihilators[0].site * space.num_orbitals_per_site + annihilators[0].orbital_index
                s = annihilators[1].site * space.num_orbitals_per_site + annihilators[1].orbital_index
                same_spin_terms.append(("up", p, q, r, s, coeff))
                continue

            if creator_spins == ("down", "down") and annihilator_spins == ("down", "down"):
                p = creators[0].site * space.num_orbitals_per_site + creators[0].orbital_index
                q = creators[1].site * space.num_orbitals_per_site + creators[1].orbital_index
                r = annihilators[0].site * space.num_orbitals_per_site + annihilators[0].orbital_index
                s = annihilators[1].site * space.num_orbitals_per_site + annihilators[1].orbital_index
                same_spin_terms.append(("down", p, q, r, s, coeff))
                continue

            raise UnsupportedTransformError(
                "block2 SZ/SU2 modes only support spin-conserving one-body terms and "
                "two-body terms representable by a common spatial-orbital ERI tensor."
            )

        raise UnsupportedTransformError(
            "block2 SZ/SU2 modes only support number-conserving one-body and two-body operators."
        )

    for _spin, p, q, r, s, coeff in same_spin_terms:
        predicted = eri[q, r, p, s] - eri[p, r, q, s]
        if abs(predicted - coeff) > atol:
            raise UnsupportedTransformError(
                "block2 SZ/SU2 modes require the two-body tensor to be compatible "
                "with a common spatial-orbital ERI representation."
            )

    return _Block2SpatialProblem(
        space=space,
        norb=norb,
        h1e_alpha=h1e_alpha,
        h1e_beta=h1e_beta,
        eri=eri,
        constant_shift=complex(operator.constant),
    )


def _interleaved_rdm1_from_spin_blocks(dm1a: np.ndarray, dm1b: np.ndarray) -> np.ndarray:
    dtype = np.result_type(dm1a.dtype, dm1b.dtype)
    n_orb = dm1a.shape[0]
    rdm = np.zeros((2 * n_orb, 2 * n_orb), dtype=dtype)
    rdm[0::2, 0::2] = dm1a
    rdm[1::2, 1::2] = dm1b
    return np.real_if_close(rdm)


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
        self.symmetry_request = str(kwargs.pop("symmetry", "sgf")).lower()
        if self.symmetry_request not in {"sgf", "sz", "su2", "auto"}:
            raise ValueError("DMRGSolver symmetry must be one of 'sgf', 'sz', 'su2', or 'auto'.")
        self.target_spin = kwargs.pop("target_spin", None)
        self.target_sz = kwargs.pop("target_sz", None)
        self.singlet_embedding = kwargs.pop("singlet_embedding", True)
        self.atol = kwargs.pop("atol", 1e-10)
        self.scratch_dir = kwargs.pop("scratch_dir", "/tmp/dmrg_tmp")
        self.n_threads = kwargs.pop("n_threads", 1)
        self.mpi = kwargs.pop("mpi", False)
        self.bond_dim = kwargs.pop("bond_dim", 250)
        self.bond_mul = kwargs.pop("bond_mul", 2)
        self.n_sweep = kwargs.pop("n_sweep", 20)
        self.nupdate = kwargs.pop("nupdate", 4)
        self.iprint = kwargs.pop("iprint", 0)
        self.reorder = kwargs.pop("reorder", False)
        self.reorder_method = kwargs.pop("reorder_method", "gaopt")
        self.eig_cutoff = kwargs.pop("eig_cutoff", 1e-7)
        self.legacy_kwargs = dict(kwargs)

        self.driver = None
        self.mpo = None
        self.ket = None
        self.spin_square_mpo = None
        self.energy_value = None
        self.compiled = None
        self.space = None
        self.n_electrons = None
        self.symmetry_mode = None
        self.target_spin_twos = None
        self.target_sz_twos = None

    def _symmetry_flag(self, mode: str):
        base = {
            "sgf": SymmetryTypes.SGF,
            "sz": SymmetryTypes.SZ,
            "su2": SymmetryTypes.SU2,
        }[mode]
        return (SymmetryTypes.CPX | base) if self.iscomplex else base

    def _explicit_spin_sector_requested(self, *, n_electrons=None, n_particles=None) -> bool:
        spec = n_particles if n_particles is not None else n_electrons
        if isinstance(spec, tuple) and len(spec) == 2:
            return True
        if isinstance(spec, list):
            if len(spec) == 1 and isinstance(spec[0], tuple) and len(spec[0]) == 2:
                return True
            if len(spec) == 2 and all(isinstance(value, (int, np.integer)) for value in spec):
                return True
        return False

    def _resolve_target_sz_twos(
        self,
        sector: tuple[int, int],
        total_electrons: int,
        *,
        explicit_spin_sector: bool,
    ) -> int:
        inferred = int(sector[0] - sector[1])
        if self.target_sz is None:
            return inferred
        target = int(self.target_sz)
        if explicit_spin_sector and target != inferred:
            raise ValueError(
                "DMRGSolver symmetry='sz' received an explicit (n_up, n_down) sector "
                f"{sector} but target_sz={target}, which implies a different 2Sz."
            )
        if abs(target) > total_electrons or (total_electrons + target) % 2 != 0:
            raise ValueError(
                f"target_sz={target} is incompatible with total electron count {total_electrons}."
            )
        return target

    def _resolve_target_spin_twos(self, sector: tuple[int, int], total_electrons: int) -> int:
        min_twos = abs(int(sector[0] - sector[1]))
        target = min_twos if self.target_spin is None else int(self.target_spin)
        if target < min_twos:
            raise ValueError(
                f"target_spin={target} is incompatible with the requested particle sector {sector}."
            )
        if target > total_electrons or (total_electrons - target) % 2 != 0:
            raise ValueError(
                f"target_spin={target} is incompatible with total electron count {total_electrons}."
            )
        return target

    def _prepare_sgf_problem(self, hamiltonian, *, n_electrons=None, n_particles=None) -> _Block2PreparedProblem:
        compiled = _compile_direct_dmrg_hamiltonian(hamiltonian)
        sector, total_electrons = normalize_electron_count(
            space=compiled.space,
            n_electrons=n_electrons,
            n_particles=n_particles,
        )
        h1e = _as_driver_array(compiled.h1e, iscomplex=self.iscomplex, name="one-body tensor")
        g2e = _as_driver_array(compiled.g2e, iscomplex=self.iscomplex, name="two-body tensor")
        ecore = compiled.constant_shift if self.iscomplex else _real_scalar(
            compiled.constant_shift,
            name="constant shift",
        )
        return _Block2PreparedProblem(
            mode="sgf",
            n_sites=compiled.space.num_spin_orbitals,
            n_electrons=total_electrons,
            sector=sector,
            spin=0,
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            target_spin_twos=None,
            target_sz_twos=None,
        )

    def _prepare_spatial_problem(self, hamiltonian, *, n_electrons=None, n_particles=None) -> _Block2PreparedProblem:
        sector, total_electrons = normalize_electron_count(
            space=coerce_symbolic_operator(hamiltonian).space,
            n_electrons=n_electrons,
            n_particles=n_particles,
        )
        spatial = _build_block2_spatial_problem(hamiltonian, atol=self.atol)
        explicit_spin_sector = self._explicit_spin_sector_requested(
            n_electrons=n_electrons,
            n_particles=n_particles,
        )

        mode = self.symmetry_request
        if mode == "auto":
            mode = "su2" if np.allclose(spatial.h1e_alpha, spatial.h1e_beta, atol=self.atol) else "sz"

        ecore = spatial.constant_shift if self.iscomplex else _real_scalar(
            spatial.constant_shift,
            name="constant shift",
        )
        if mode == "sz":
            target_sz_twos = self._resolve_target_sz_twos(
                sector,
                total_electrons,
                explicit_spin_sector=explicit_spin_sector,
            )
            return _Block2PreparedProblem(
                mode="sz",
                n_sites=spatial.norb,
                n_electrons=total_electrons,
                sector=sector,
                spin=target_sz_twos,
                h1e=[
                    _as_driver_array(spatial.h1e_alpha, iscomplex=self.iscomplex, name="alpha one-body tensor"),
                    _as_driver_array(spatial.h1e_beta, iscomplex=self.iscomplex, name="beta one-body tensor"),
                ],
                g2e=[
                    _as_driver_array(spatial.eri, iscomplex=self.iscomplex, name="aa two-body tensor"),
                    _as_driver_array(spatial.eri, iscomplex=self.iscomplex, name="ab two-body tensor"),
                    _as_driver_array(spatial.eri, iscomplex=self.iscomplex, name="bb two-body tensor"),
                ],
                ecore=ecore,
                target_spin_twos=None,
                target_sz_twos=target_sz_twos,
            )

        if not np.allclose(spatial.h1e_alpha, spatial.h1e_beta, atol=self.atol):
            raise UnsupportedTransformError(
                "DMRGSolver symmetry='su2' requires spin-independent one-body terms. "
                "Use symmetry='sz' or symmetry='sgf' for spin-dependent Hamiltonians."
            )
        target_spin_twos = self._resolve_target_spin_twos(sector, total_electrons)
        return _Block2PreparedProblem(
            mode="su2",
            n_sites=spatial.norb,
            n_electrons=total_electrons,
            sector=sector,
            spin=target_spin_twos,
            h1e=_as_driver_array(spatial.h1e_alpha, iscomplex=self.iscomplex, name="one-body tensor"),
            g2e=_as_driver_array(spatial.eri, iscomplex=self.iscomplex, name="two-body tensor"),
            ecore=ecore,
            target_spin_twos=target_spin_twos,
            target_sz_twos=sector[0] - sector[1],
        )

    def _prepare_problem(self, hamiltonian, *, n_electrons=None, n_particles=None) -> _Block2PreparedProblem:
        if self.symmetry_request == "sgf":
            return self._prepare_sgf_problem(hamiltonian, n_electrons=n_electrons, n_particles=n_particles)
        try:
            return self._prepare_spatial_problem(hamiltonian, n_electrons=n_electrons, n_particles=n_particles)
        except UnsupportedTransformError:
            if self.symmetry_request != "auto":
                raise
            return self._prepare_sgf_problem(hamiltonian, n_electrons=n_electrons, n_particles=n_particles)

    def _initialize_driver(self, *, problem: _Block2PreparedProblem) -> None:
        symm_type = self._symmetry_flag(problem.mode)
        # the symm_type need to be reconsider for efficiency. If the system is spin degenerated, 
        # using more symmetry sectors would be better to ustilize it and reduce the bond dimension.
        self.driver = DMRGDriver(
            scratch=self.scratch_dir,
            symm_type=symm_type,
            n_threads=self.n_threads,
            mpi=self.mpi,
            stack_mem=5368709120,
        )
        self.driver.initialize_system(
            n_sites=problem.n_sites,
            n_elec=problem.n_electrons,
            spin=problem.spin,
            singlet_embedding=self.singlet_embedding,
        )
        self.spin_square_mpo = None

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        problem = self._prepare_problem(hamiltonian, n_electrons=n_electrons, n_particles=n_particles)

        self.compiled = problem
        self.space = coerce_symbolic_operator(hamiltonian).space
        self.n_electrons = problem.n_electrons
        self.symmetry_mode = problem.mode
        self.target_spin_twos = problem.target_spin_twos
        self.target_sz_twos = problem.target_sz_twos

        self._initialize_driver(problem=problem)

        reorder_idx = None
        if self.reorder:
            reorder_idx = self.driver.orbital_reordering(
                h1e=problem.h1e,
                g2e=problem.g2e,
                method=self.reorder_method,
            )

        self.mpo = self.driver.get_qc_mpo(
            h1e=problem.h1e,
            g2e=problem.g2e,
            ecore=problem.ecore,
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

    def _build_operator_mpo(self, operator):
        if self.symmetry_mode == "sgf":
            compiled = to_qc_tensors(operator)
            h1e = _as_driver_array(compiled.h1e, iscomplex=self.iscomplex, name="observable one-body tensor")
            g2e = _as_driver_array(compiled.g2e, iscomplex=self.iscomplex, name="observable two-body tensor")
            ecore = compiled.constant_shift if self.iscomplex else _real_scalar(
                compiled.constant_shift,
                name="observable constant shift",
            )
        else:
            compiled = _build_block2_spatial_problem(operator, atol=self.atol)
            ecore = compiled.constant_shift if self.iscomplex else _real_scalar(
                compiled.constant_shift,
                name="observable constant shift",
            )
            if self.symmetry_mode == "sz":
                h1e = [
                    _as_driver_array(compiled.h1e_alpha, iscomplex=self.iscomplex, name="observable alpha one-body tensor"),
                    _as_driver_array(compiled.h1e_beta, iscomplex=self.iscomplex, name="observable beta one-body tensor"),
                ]
                g2e = [
                    _as_driver_array(compiled.eri, iscomplex=self.iscomplex, name="observable aa two-body tensor"),
                    _as_driver_array(compiled.eri, iscomplex=self.iscomplex, name="observable ab two-body tensor"),
                    _as_driver_array(compiled.eri, iscomplex=self.iscomplex, name="observable bb two-body tensor"),
                ]
            else:
                if not np.allclose(compiled.h1e_alpha, compiled.h1e_beta, atol=self.atol):
                    raise UnsupportedTransformError(
                        "Generic expectations in block2 symmetry='su2' require spin-scalar operators. "
                        "Use symmetry='sz' or symmetry='sgf' for spin-resolved observables."
                    )
                h1e = _as_driver_array(compiled.h1e_alpha, iscomplex=self.iscomplex, name="observable one-body tensor")
                g2e = _as_driver_array(compiled.eri, iscomplex=self.iscomplex, name="observable two-body tensor")

        return self.driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            iprint=0,
            reorder=self.driver.reorder_idx,
        )

    def _expectation(self, operator):
        mpo = self._build_operator_mpo(operator)
        return self.driver.expectation(self.ket, mpo, self.ket)

    def energy(self):
        self._require_solution()
        if self.energy_value is None:
            self.energy_value = self.driver.expectation(self.ket, self.mpo, self.ket)
        return self.energy_value

    def rdm1(self, *, kind: str | None = None):
        self._require_solution()
        raw = self.driver.get_1pdm(self.ket)
        if self.symmetry_mode == "sgf":
            if kind not in (None, "spin_orbital"):
                raise ValueError("SGF block2 rdm1() supports only `kind='spin_orbital'`.")
            return np.asarray(raw)

        if self.symmetry_mode == "sz":
            if isinstance(raw, (list, tuple)):
                dm1a, dm1b = raw
            else:
                dm1a, dm1b = np.asarray(raw)[0], np.asarray(raw)[1]
            dm1a = np.asarray(dm1a)
            dm1b = np.asarray(dm1b)
            if kind == "spin_blocks":
                return dm1a, dm1b
            if kind not in (None, "spin_orbital"):
                raise ValueError("SZ block2 rdm1() supports `kind='spin_orbital'` or `kind='spin_blocks'`.")
            return _interleaved_rdm1_from_spin_blocks(dm1a, dm1b)

        spin_traced = np.asarray(raw)
        if kind == "spin_traced":
            return spin_traced
        if kind not in (None, "spin_orbital"):
            raise ValueError("SU2 block2 rdm1() supports `kind='spin_orbital'` or `kind='spin_traced'`.")
        if self.target_spin_twos != 0:
            raise NotImplementedError(
                "SU2 block2 rdm1() can reconstruct the full spin-orbital density only for target_spin=0. "
                "Use `kind='spin_traced'` or choose symmetry='sz'/'sgf' for spin-resolved densities."
            )
        return _interleaved_rdm1_from_spin_blocks(0.5 * spin_traced, 0.5 * spin_traced)

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
        if self.spin_square_mpo is None:
            self.spin_square_mpo = self.driver.get_spin_square_mpo(iprint=0)
        return self.driver.expectation(self.ket, self.spin_square_mpo, self.ket)

    def diagnostics(self) -> dict[str, object]:
        data = super().diagnostics()
        if self.symmetry_mode == "sgf":
            rdm1_kinds: tuple[str, ...] = ("spin_orbital",)
        elif self.symmetry_mode == "sz":
            rdm1_kinds = ("spin_orbital", "spin_blocks")
        elif self.target_spin_twos == 0:
            rdm1_kinds = ("spin_orbital", "spin_traced")
        else:
            rdm1_kinds = ("spin_traced",)
        data.update(
            {
                "n_spin_orbitals": None if self.space is None else self.space.num_spin_orbitals,
                "n_block2_sites": None if self.compiled is None else self.compiled.n_sites,
                "n_electrons": self.n_electrons,
                "bond_dim": self.bond_dim,
                "bond_mul": self.bond_mul,
                "n_sweep": self.n_sweep,
                "reorder": self.reorder,
                "reorder_method": self.reorder_method,
                "has_driver": self.driver is not None,
                "has_mpo": self.mpo is not None,
                "requested_symmetry": self.symmetry_request,
                "resolved_symmetry": self.symmetry_mode,
                "sector": None if self.compiled is None else self.compiled.sector,
                "target_spin_twos": self.target_spin_twos,
                "target_sz_twos": self.target_sz_twos,
                "singlet_embedding": self.singlet_embedding,
                "rdm1_kinds": rdm1_kinds,
            }
        )
        return data
