"""TeNPy DMRG backend built on the local-term compiler family.

The TeNPy path now builds MPOs term by term from the backend-neutral compiled
local blocks, so the intermediate object size is tied to each term support
rather than the full Hilbert space. A dense full-Hamiltonian fallback remains
only for the current particle-sector penalty path and for tiny systems where
direct diagonalization is cheaper than launching DMRG.
"""

from __future__ import annotations

import numpy as np

try:
    import tenpy.linalg.np_conserved as npc
    from tenpy.algorithms import dmrg
    from tenpy.linalg.charges import LegCharge
    from tenpy.models.lattice import Chain
    from tenpy.models.model import MPOModel
    from tenpy.networks.mpo import MPO
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import Site

    _TENPY_IMPORT_ERROR = None
except ImportError as exc:
    npc = None
    dmrg = None
    LegCharge = None
    Chain = None
    MPOModel = None
    MPO = None
    MPS = None
    Site = None
    _TENPY_IMPORT_ERROR = exc

from ...operator.operator import Operator as SymbolicOperator
from ...utils.dependencies import require_dependency
from ...utils.solver import coerce_symbolic_operator, normalize_electron_count
from ..base import SolverBackend
from ..capabilities import BackendCapabilities
from ..compile import compile_local_terms
from ._local_terms_dense import (
    dense_hamiltonian_from_local_terms,
    vector_expectation,
)
from ._number_penalty import (
    compiled_row_sum_bound,
    identity_mpo_tensors,
    local_number_matrix,
    sector_penalty_operator,
    square_sum_mpo_tensors,
    sum_mpo_tensors,
)


def _site_from_local_basis(basis) -> Site:
    """Construct one TeNPy local site matching one `ElectronicSpace` site block."""
    num_local_modes = basis.num_local_modes
    state_labels = tuple(format(index, f"0{num_local_modes}b") for index in range(basis.local_dim))
    site = Site(
        LegCharge.from_trivial(basis.local_dim),
        state_labels=state_labels,
        JW=np.asarray(basis.parity, dtype=np.complex128),
    )
    for local_index in range(num_local_modes):
        site.add_op(
            f"C{local_index}",
            np.asarray(basis.create_ops[local_index], dtype=np.complex128),
            need_JW=True,
            permute_dense=False,
        )
        site.add_op(
            f"D{local_index}",
            np.asarray(basis.destroy_ops[local_index], dtype=np.complex128),
            need_JW=True,
            permute_dense=False,
        )
    return site


def _dense_to_mpo(matrix: np.ndarray, *, sites: list[Site], svd_cutoff: float) -> MPO:
    """Convert a full dense operator into an exact finite MPO by sequential SVD."""
    local_dim = sites[0].dim
    num_sites = len(sites)
    tensor = np.asarray(matrix, dtype=np.complex128).reshape([local_dim] * num_sites + [local_dim] * num_sites)

    # MPO tensors use per-site operator legs `(p_i, p_i*)`, so interleave bra/ket
    # site dimensions before performing the tensor-train factorization.
    permutation = []
    for site in range(num_sites):
        permutation.extend([site, site + num_sites])
    rest = np.transpose(tensor, permutation)

    left_rank = 1
    tensors = []
    for _site in range(num_sites - 1):
        rest = rest.reshape(left_rank * local_dim * local_dim, -1)
        u, singular_values, vh = np.linalg.svd(rest, full_matrices=False)
        keep = int(np.sum(singular_values > svd_cutoff))
        if keep == 0:
            keep = 1
        u = u[:, :keep]
        singular_values = singular_values[:keep]
        vh = vh[:keep, :]

        tensor_site = u.reshape(left_rank, local_dim, local_dim, keep).transpose(0, 3, 1, 2)
        tensors.append(
            npc.Array.from_ndarray_trivial(
                tensor_site,
                labels=["wL", "wR", "p", "p*"],
            )
        )
        rest = singular_values[:, None] * vh
        left_rank = keep

    tensors.append(
        npc.Array.from_ndarray_trivial(
            rest.reshape(left_rank, 1, local_dim, local_dim),
            labels=["wL", "wR", "p", "p*"],
        )
    )
    return MPO(
        sites,
        tensors,
        bc="finite",
        IdL=[0] + [None] * num_sites,
        IdR=[None] * num_sites + [0],
        mps_unit_cell_width=1,
    )


def _identity_tensor(local_dim: int, *, coefficient: complex = 1.0):
    tensor = np.zeros((1, 1, local_dim, local_dim), dtype=np.complex128)
    tensor[0, 0, :, :] = coefficient * np.eye(local_dim, dtype=np.complex128)
    return npc.Array.from_ndarray_trivial(tensor, labels=["wL", "wR", "p", "p*"])


def _support_dense_to_mpo_tensors(matrix: np.ndarray, *, support_size: int, local_dim: int, svd_cutoff: float):
    """Factor a dense operator on a contiguous support block into MPO tensors."""
    tensor = np.asarray(matrix, dtype=np.complex128).reshape(
        [local_dim] * support_size + [local_dim] * support_size
    )
    permutation = []
    for site in range(support_size):
        permutation.extend([site, site + support_size])
    rest = np.transpose(tensor, permutation)

    left_rank = 1
    tensors = []
    for _site in range(support_size - 1):
        rest = rest.reshape(left_rank * local_dim * local_dim, -1)
        u, singular_values, vh = np.linalg.svd(rest, full_matrices=False)
        keep = int(np.sum(singular_values > svd_cutoff))
        if keep == 0:
            keep = 1
        u = u[:, :keep]
        singular_values = singular_values[:keep]
        vh = vh[:keep, :]

        tensor_site = u.reshape(left_rank, local_dim, local_dim, keep).transpose(0, 3, 1, 2)
        tensors.append(npc.Array.from_ndarray_trivial(tensor_site, labels=["wL", "wR", "p", "p*"]))
        rest = singular_values[:, None] * vh
        left_rank = keep

    tensors.append(
        npc.Array.from_ndarray_trivial(
            rest.reshape(left_rank, 1, local_dim, local_dim),
            labels=["wL", "wR", "p", "p*"],
        )
    )
    return tensors


def _term_to_global_mpo(term, *, sites: list[Site], svd_cutoff: float) -> MPO:
    """Convert one compiled local-block term into a global finite MPO."""
    local_dim = sites[0].dim
    num_sites = len(sites)
    start = term.sites[0]
    support_size = len(term.sites)
    support_tensors = _support_dense_to_mpo_tensors(
        term.matrix,
        support_size=support_size,
        local_dim=local_dim,
        svd_cutoff=svd_cutoff,
    )

    tensors = []
    support_index = 0
    for site in range(num_sites):
        if start <= site < start + support_size:
            tensors.append(support_tensors[support_index])
            support_index += 1
        else:
            tensors.append(_identity_tensor(local_dim))

    return MPO(
        sites,
        tensors,
        bc="finite",
        IdL=[0] + [None] * num_sites,
        IdR=[None] * num_sites + [0],
        mps_unit_cell_width=1,
    )


def _identity_mpo(sites: list[Site], *, coefficient: complex = 1.0) -> MPO:
    """Build a rank-1 identity MPO with an optional scalar prefactor."""
    local_dim = sites[0].dim
    tensors = [_identity_tensor(local_dim, coefficient=coefficient)] + [
        _identity_tensor(local_dim) for _ in range(len(sites) - 1)
    ]
    return MPO(
        sites,
        tensors,
        bc="finite",
        IdL=[0] + [None] * len(sites),
        IdR=[None] * len(sites) + [0],
        mps_unit_cell_width=1,
    )


def _mpo_from_local_terms(compiled, *, sites: list[Site], svd_cutoff: float) -> MPO:
    """Build a TeNPy MPO directly from compiled local support blocks."""
    mpo = None
    for term in compiled.terms:
        term_mpo = _term_to_global_mpo(term, sites=sites, svd_cutoff=svd_cutoff)
        mpo = term_mpo if mpo is None else mpo + term_mpo

    if mpo is None:
        mpo = _identity_mpo(sites, coefficient=0.0)

    if abs(complex(compiled.constant_shift)) > 1e-15:
        mpo = mpo + _identity_mpo(sites, coefficient=complex(compiled.constant_shift))
    return mpo


def _tenpy_mpo_from_raw_tensors(raw_tensors: list[np.ndarray], *, sites: list[Site]) -> MPO:
    """Wrap raw 4-leg MPO tensors into a TeNPy `MPO`."""
    tensors = [
        npc.Array.from_ndarray_trivial(
            np.asarray(tensor, dtype=np.complex128),
            labels=["wL", "wR", "p", "p*"],
        )
        for tensor in raw_tensors
    ]
    num_sites = len(sites)
    return MPO(
        sites,
        tensors,
        bc="finite",
        IdL=[0] + [None] * num_sites,
        IdR=[None] * num_sites + [0],
        mps_unit_cell_width=1,
    )


def _sector_penalty_mpo(compiled, *, sites: list[Site], target_pair: tuple[int, int], penalty_scale: float) -> MPO:
    """Build an exact MPO for the quadratic particle-sector penalty."""
    num_sites = compiled.space.num_sites
    local_dim = compiled.local_basis.local_dim
    penalty = _tenpy_mpo_from_raw_tensors(
        identity_mpo_tensors(local_dim, num_sites, coefficient=0.0),
        sites=sites,
    )

    for spin_label, target in zip(("up", "down"), target_pair):
        local_q = local_number_matrix(compiled.local_basis, spin=spin_label)
        part = _tenpy_mpo_from_raw_tensors(
            square_sum_mpo_tensors(local_q, num_sites, coefficient=penalty_scale),
            sites=sites,
        )
        part = part + _tenpy_mpo_from_raw_tensors(
            sum_mpo_tensors(local_q, num_sites, coefficient=-2.0 * penalty_scale * target),
            sites=sites,
        )
        part = part + _tenpy_mpo_from_raw_tensors(
            identity_mpo_tensors(local_dim, num_sites, coefficient=penalty_scale * (target**2)),
            sites=sites,
        )
        penalty = penalty + part

    return penalty


def _product_state_for_target(space, target_pair: tuple[int, int] | None) -> list[int]:
    """Build a simple site-product initial state matching a target particle sector."""
    if target_pair is None:
        return [0] * space.num_sites

    remaining = {
        "up": int(target_pair[0]),
        "down": int(target_pair[1]),
    }
    local_states = [0] * space.num_sites

    for spin in space.spins:
        for site in range(space.num_sites):
            for orbital_index, _orbital in enumerate(space.orbitals):
                if remaining[spin] <= 0:
                    break
                local_mode = orbital_index * space.num_spins + space.spin_index(spin)
                local_states[site] |= 1 << local_mode
                remaining[spin] -= 1
            if remaining[spin] <= 0:
                break

    if remaining["up"] != 0 or remaining["down"] != 0:
        raise ValueError(
            "Could not build a TeNPy initial product state for the requested electron sector "
            f"{target_pair!r} on space {space!r}."
        )
    return local_states


def _mps_from_dense_state(vector: np.ndarray, *, sites: list[Site], local_dim: int) -> MPS:
    """Convert a tiny exact wavefunction into a native TeNPy MPS."""
    num_sites = len(sites)
    psi_tensor = npc.Array.from_ndarray_trivial(
        np.asarray(vector, dtype=np.complex128).reshape([local_dim] * num_sites),
        labels=[f"p{site}" for site in range(num_sites)],
    )
    return MPS.from_full(
        sites,
        psi_tensor,
        bc="finite",
        unit_cell_width=1,
    )


class TeNPyDMRGBackend(SolverBackend):
    """Concrete tensor-network DMRG backend implemented with TeNPy."""

    solver_family = "dmrg"
    backend_name = "tenpy"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_rdm1=True,
        supports_native_expectation=True,
        supports_complex=True,
        preferred_problem_type="local_terms",
    )

    def __init__(self, *args, **kwargs) -> None:
        require_dependency("TeNPyDMRGBackend", "tenpy", _TENPY_IMPORT_ERROR)
        self.chi_max = kwargs.pop("chi_max", 64)
        self.svd_min = kwargs.pop("svd_min", 1e-12)
        self.max_sweeps = kwargs.pop("max_sweeps", 8)
        self.max_E_err = kwargs.pop("max_E_err", 1e-8)
        self.combine = kwargs.pop("combine", False)
        self.mixer = kwargs.pop("mixer", None)
        self.particle_penalty = kwargs.pop("particle_penalty", None)
        self.mpo_svd_cutoff = kwargs.pop("mpo_svd_cutoff", 1e-12)
        self.legacy_kwargs = dict(kwargs)

        self.operator = None
        self.space = None
        self.compiled = None
        self.sites = None
        self.mpo = None
        self.model = None
        self.psi = None
        self.info = None
        self.energy_value = None
        self.state_vector = None
        self.mpo_mode = None

    def _make_mpo_from_dense(self, dense: np.ndarray, *, space, compiled):
        site = _site_from_local_basis(compiled.local_basis)
        sites = [site] * space.num_sites
        mpo = _dense_to_mpo(dense, sites=sites, svd_cutoff=self.mpo_svd_cutoff)
        return sites, mpo

    def _make_mpo_from_local_terms(self, *, compiled):
        site = _site_from_local_basis(compiled.local_basis)
        sites = [site] * compiled.space.num_sites
        mpo = _mpo_from_local_terms(compiled, sites=sites, svd_cutoff=self.mpo_svd_cutoff)
        return sites, mpo

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

        dense = None

        self.operator = operator
        self.space = compiled.space
        self.compiled = compiled
        self.psi = None
        self.info = None
        self.state_vector = None
        self.mpo_mode = None

        if compiled.space.num_sites <= 2:
            dense = dense_hamiltonian_from_local_terms(compiled)
            if target_pair is not None:
                penalty_scale = self.particle_penalty
                if penalty_scale is None:
                    penalty_scale = 10.0 * max(1.0, compiled_row_sum_bound(compiled))
                penalty_operator = sector_penalty_operator(
                    operator.space,
                    target_pair,
                    penalty_scale=penalty_scale,
                )
                dense = dense + dense_hamiltonian_from_local_terms(compile_local_terms(penalty_operator))
            eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(dense, dtype=np.complex128))
            self.sites = None
            self.mpo = None
            self.model = None
            self.energy_value = eigenvalues[0]
            site = _site_from_local_basis(compiled.local_basis)
            self.sites = [site] * compiled.space.num_sites
            self.psi = _mps_from_dense_state(
                np.asarray(eigenvectors[:, 0], dtype=np.complex128),
                sites=self.sites,
                local_dim=compiled.local_basis.local_dim,
            )
            self.state_vector = None
            self.mpo_mode = "exact_mps"
            return self

        self.sites, self.mpo = self._make_mpo_from_local_terms(compiled=compiled)
        self.mpo_mode = "local_terms"
        if target_pair is not None:
            penalty_scale = self.particle_penalty
            if penalty_scale is None:
                penalty_scale = 10.0 * max(1.0, compiled_row_sum_bound(compiled))
            penalty_mpo = _sector_penalty_mpo(
                compiled,
                sites=self.sites,
                target_pair=target_pair,
                penalty_scale=penalty_scale,
            )
            self.mpo = self.mpo + penalty_mpo
        lattice = Chain(compiled.space.num_sites, self.sites[0], bc="open")
        self.model = MPOModel(lattice, self.mpo)
        self.psi = MPS.from_product_state(
            lattice.mps_sites(),
            _product_state_for_target(compiled.space, target_pair),
            bc="finite",
            unit_cell_width=1,
        )

        options = {
            "trunc_params": {
                "chi_max": self.chi_max,
                "svd_min": self.svd_min,
            },
            "max_sweeps": self.max_sweeps,
            "max_E_err": self.max_E_err,
            "combine": self.combine,
        }
        if self.mixer is not None:
            options["mixer"] = self.mixer

        self.info = dmrg.run(self.psi, self.model, options)
        self.energy_value = self.info["E"]
        return self

    def _require_solution(self) -> None:
        if self.space is None or (self.psi is None and self.state_vector is None):
            raise RuntimeError("Call solve() before requesting observables.")

    def _expectation(self, operator: SymbolicOperator):
        compiled = compile_local_terms(operator)
        if self.state_vector is not None:
            dense = dense_hamiltonian_from_local_terms(compiled)
            return vector_expectation(self.state_vector, dense)
        mpo = _mpo_from_local_terms(compiled, sites=self.sites, svd_cutoff=self.mpo_svd_cutoff)
        return mpo.expectation_value(self.psi)

    def energy(self):
        self._require_solution()
        if self.energy_value is None:
            if self.state_vector is not None:
                self.energy_value = self._expectation(self.operator)
            else:
                self.energy_value = self.mpo.expectation_value(self.psi)
        return self.energy_value

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

    def diagnostics(self) -> dict[str, object]:
        data = super().diagnostics()
        data.update(
            {
                "mpo_mode": self.mpo_mode,
                "has_mpo": self.mpo is not None,
                "has_psi": self.psi is not None,
                "chi_max": self.chi_max,
                "max_sweeps": self.max_sweeps,
            }
        )
        return data
