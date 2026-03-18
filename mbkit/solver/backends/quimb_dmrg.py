from __future__ import annotations

import numpy as np

try:
    import quimb.tensor as qtn

    _QUIMB_IMPORT_ERROR = None
except ImportError as exc:
    qtn = None
    _QUIMB_IMPORT_ERROR = exc

from ...operator.operator import Operator as SymbolicOperator
from ...utils.dependencies import require_dependency
from ...utils.solver import coerce_symbolic_operator, normalize_electron_count
from ..base import SolverBackend
from ..capabilities import BackendCapabilities
from ..compile import compile_local_terms
from ._local_terms_dense import (
    as_real_if_possible as _as_real_if_possible,
    compiled_is_complex as _compiled_is_complex,
    dense_hamiltonian_from_local_terms as _dense_hamiltonian_from_local_terms,
)
from ._number_penalty import (
    compiled_row_sum_bound,
    identity_mpo_tensors,
    local_number_matrix,
    sector_penalty_operator,
    square_sum_mpo_tensors,
    sum_mpo_tensors,
)


def _identity_fill_array(local_dim: int, *, dtype) -> np.ndarray:
    return np.eye(local_dim, dtype=dtype)


def _mpo_from_local_terms(compiled, *, dtype):
    local_dim = compiled.local_basis.local_dim
    fill_array = _identity_fill_array(local_dim, dtype=dtype)
    mpo = None

    for term in compiled.terms:
        dims = [local_dim] * len(term.sites)
        matrix = _as_real_if_possible(term.matrix) if dtype is float else np.asarray(term.matrix, dtype=dtype)
        term_mpo = qtn.MatrixProductOperator.from_dense(
            matrix,
            dims=dims,
            sites=term.sites,
            L=compiled.space.num_sites,
        )
        term_mpo = term_mpo.fill_empty_sites(fill_array=fill_array, inplace=False)
        mpo = term_mpo if mpo is None else mpo.add_MPO(term_mpo)

    if mpo is None:
        mpo = qtn.MPO_identity(
            compiled.space.num_sites,
            phys_dim=local_dim,
            dtype=dtype,
        )
        mpo *= 0.0

    if abs(complex(compiled.constant_shift)) > 1e-15:
        identity = qtn.MPO_identity(
            compiled.space.num_sites,
            phys_dim=local_dim,
            dtype=dtype,
        )
        mpo = mpo.add_MPO(identity * complex(compiled.constant_shift))

    mpo.compress(cutoff=1e-12)
    return mpo


def _quimb_mpo_from_raw_tensors(raw_tensors: list[np.ndarray]):
    """Wrap raw 4-leg MPO tensors into a quimb `MatrixProductOperator`."""
    arrays = []
    num_sites = len(raw_tensors)
    for site, tensor in enumerate(raw_tensors):
        array = np.asarray(tensor)
        if num_sites == 1:
            arrays.append(array[0, 0, :, :])
        elif site == 0:
            arrays.append(array[0, :, :, :])
        elif site == num_sites - 1:
            arrays.append(array[:, 0, :, :])
        else:
            arrays.append(array)
    return qtn.MatrixProductOperator(arrays, L=num_sites, shape="lrud")


def _cast_raw_tensors(raw_tensors: list[np.ndarray], *, dtype):
    if dtype is float:
        return [np.asarray(np.real_if_close(tensor), dtype=float) for tensor in raw_tensors]
    return [np.asarray(tensor, dtype=np.complex128) for tensor in raw_tensors]


def _sector_penalty_mpo(compiled, *, target_pair: tuple[int, int], penalty_scale: float, dtype):
    """Build an exact quimb MPO for the quadratic particle-sector penalty."""
    num_sites = compiled.space.num_sites
    local_dim = compiled.local_basis.local_dim
    penalty = _quimb_mpo_from_raw_tensors(identity_mpo_tensors(local_dim, num_sites, coefficient=0.0))

    for spin_label, target in zip(("up", "down"), target_pair):
        local_q = local_number_matrix(compiled.local_basis, spin=spin_label)
        part = _quimb_mpo_from_raw_tensors(
            _cast_raw_tensors(square_sum_mpo_tensors(local_q, num_sites, coefficient=penalty_scale), dtype=dtype)
        )
        part = part.add_MPO(
            _quimb_mpo_from_raw_tensors(
                _cast_raw_tensors(
                    sum_mpo_tensors(local_q, num_sites, coefficient=-2.0 * penalty_scale * target),
                    dtype=dtype,
                )
            )
        )
        part = part.add_MPO(
            _quimb_mpo_from_raw_tensors(
                _cast_raw_tensors(
                    identity_mpo_tensors(local_dim, num_sites, coefficient=penalty_scale * (target**2)),
                    dtype=dtype,
                )
            )
        )
        penalty = penalty.add_MPO(part)

    penalty.compress(cutoff=1e-12)
    return penalty


def _expectation_from_local_terms(state, compiled, *, info: dict | None = None):
    """Compute a local-terms expectation with quimb's native MPS contractions."""
    if not compiled.terms:
        return complex(compiled.constant_shift)

    terms = {}
    for term in compiled.terms:
        terms[tuple(term.sites)] = np.asarray(term.matrix)

    value = state.compute_local_expectation(
        terms,
        normalized=True,
        return_all=False,
        method="canonical",
        info=info,
        inplace=False,
    )
    return complex(compiled.constant_shift) + value


def _mps_from_dense_state(vector: np.ndarray, *, dims: list[int]):
    """Convert a tiny exact wavefunction into a native quimb MPS."""
    state = qtn.MatrixProductState.from_dense(
        np.asarray(vector, dtype=np.complex128),
        dims=dims,
    )
    state.normalize()
    state.canonize(0, inplace=True)
    return state

class QuimbDMRGBackend(SolverBackend):
    """Concrete tensor-network DMRG backend implemented with quimb.

    The first implementation compiles the Hamiltonian into local site blocks,
    assembles an exact dense operator in the site-local Fock basis, and then
    converts that operator into an MPO with `MatrixProductOperator.from_dense`.
    This is already a real tensor-network backend path, while leaving room for a
    later optimized direct local-term-to-MPO lowering.
    """

    solver_family = "dmrg"
    backend_name = "quimb"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_rdm1=True,
        supports_native_expectation=False,
        supports_complex=True,
        preferred_problem_type="local_terms",
    )

    def __init__(self, *args, **kwargs) -> None:
        require_dependency("QuimbDMRGBackend", "quimb", _QUIMB_IMPORT_ERROR)
        self.bond_dims = kwargs.pop("bond_dims", [16, 32, 64])
        self.cutoffs = kwargs.pop("cutoffs", 1e-9)
        self.which = kwargs.pop("which", "SA")
        self.solve_tol = kwargs.pop("solve_tol", 1e-6)
        self.max_sweeps = kwargs.pop("max_sweeps", 8)
        self.verbosity = kwargs.pop("verbosity", 0)
        self.particle_penalty = kwargs.pop("particle_penalty", None)
        self.legacy_kwargs = dict(kwargs)

        self.operator = None
        self.space = None
        self.compiled = None
        self.mpo = None
        self.dmrg = None
        self.energy_value = None
        self.state = None
        self.state_vector = None
        self.solve_mode = None
        self._expectation_info = None

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        operator = coerce_symbolic_operator(hamiltonian)
        compiled = compile_local_terms(operator)
        target_pair = None
        if n_electrons is not None or n_particles is not None or operator.space.electron_count() is not None:
            target_pair, _ = normalize_electron_count(
                space=operator.space,
                n_electrons=n_electrons,
                n_particles=n_particles,
            )
        dims = [compiled.local_basis.local_dim] * compiled.space.num_sites

        self.operator = operator
        self.space = compiled.space
        self.compiled = compiled
        self.state = None
        self.solve_mode = None
        self._expectation_info = {"cur_orthog": "calc"}
        if compiled.space.num_sites == 1:
            dense = _dense_hamiltonian_from_local_terms(compiled)
            if target_pair is not None:
                penalty_scale = self.particle_penalty
                if penalty_scale is None:
                    penalty_scale = 10.0 * max(1.0, compiled_row_sum_bound(compiled))
                penalty_operator = sector_penalty_operator(
                    operator.space,
                    target_pair,
                    penalty_scale=penalty_scale,
                )
                dense = dense + _dense_hamiltonian_from_local_terms(compile_local_terms(penalty_operator))
            eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(dense, dtype=np.complex128))
            self.mpo = None
            self.dmrg = None
            self.energy_value = eigenvalues[0]
            self.state = _mps_from_dense_state(
                np.asarray(eigenvectors[:, 0], dtype=np.complex128),
                dims=dims,
            )
            self.state_vector = None
            self.solve_mode = "exact_mps"
            return self

        mpo_dtype = np.complex128 if _compiled_is_complex(compiled) else float
        self.mpo = _mpo_from_local_terms(compiled, dtype=mpo_dtype)
        self.solve_mode = "local_terms"
        if target_pair is not None:
            penalty_scale = self.particle_penalty
            if penalty_scale is None:
                penalty_scale = 10.0 * max(1.0, compiled_row_sum_bound(compiled))
            penalty_mpo = _sector_penalty_mpo(
                compiled,
                target_pair=target_pair,
                penalty_scale=penalty_scale,
                dtype=mpo_dtype,
            )
            self.mpo = self.mpo.add_MPO(penalty_mpo)
            self.mpo.compress(cutoff=1e-12)
            self.solve_mode = "local_terms_penalty_mpo"
        self.dmrg = qtn.DMRG2(
            self.mpo,
            bond_dims=self.bond_dims,
            cutoffs=self.cutoffs,
            which=self.which,
        )
        self.dmrg.solve(
            tol=self.solve_tol,
            max_sweeps=self.max_sweeps,
            verbosity=self.verbosity,
        )
        self.energy_value = self.dmrg.energy
        self.state = self.dmrg.state
        self.state_vector = None
        return self

    def _require_solution(self) -> None:
        if (self.state is None and self.state_vector is None) or self.space is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def energy(self):
        self._require_solution()
        return self.energy_value

    def _expectation(self, operator: SymbolicOperator):
        compiled = compile_local_terms(operator)
        if self.state is not None:
            return _expectation_from_local_terms(self.state, compiled, info=self._expectation_info)
        dense = _dense_hamiltonian_from_local_terms(compiled)
        vector = np.asarray(self.state_vector, dtype=np.complex128).reshape(-1)
        return np.vdot(vector, dense @ vector)

    def rdm1(self):
        self._require_solution()
        nmode = self.space.num_spin_orbitals
        rdm = np.zeros((nmode, nmode), dtype=np.complex128)
        for p in range(nmode):
            left = self.space.unpack_mode(p)
            for q in range(nmode):
                right = self.space.unpack_mode(q)
                op = self.space.create(left.site, orbital=left.orbital, spin=left.spin) @ self.space.destroy(
                    right.site,
                    orbital=right.orbital,
                    spin=right.spin,
                )
                rdm[p, q] = self._expectation(op)
        return np.real_if_close(rdm)

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
