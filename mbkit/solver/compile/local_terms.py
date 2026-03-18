"""Local-block Hamiltonian compilation for tensor-network backends.

This module lowers a symbolic fermionic Hamiltonian into dense operators acting
on contiguous site blocks. The representation is backend-neutral and designed as
the shared tensor-network bridge for packages such as quimb and TeNPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...operator.operator import Operator as SymbolicOperator
from ...operator.transforms import UnsupportedTransformError
from ...utils.solver import coerce_symbolic_operator


def _kron_all(matrices: list[np.ndarray]) -> np.ndarray:
    out = np.asarray([[1.0]], dtype=np.complex128)
    for matrix in matrices:
        out = np.kron(out, matrix)
    return out


def _bit_parity(value: int) -> int:
    return int(value).bit_count() % 2


@dataclass(frozen=True)
class SiteLocalBasis:
    """Local Fock-space data for one lattice site."""

    space: object
    local_dim: int
    mode_labels: tuple[tuple[str, str], ...]
    identity: np.ndarray
    parity: np.ndarray
    create_ops: tuple[np.ndarray, ...]
    destroy_ops: tuple[np.ndarray, ...]

    @property
    def num_local_modes(self) -> int:
        return len(self.mode_labels)


@dataclass(frozen=True)
class LocalBlockTerm:
    """One dense local operator acting on a contiguous block of sites."""

    sites: tuple[int, ...]
    matrix: np.ndarray

    @property
    def support_size(self) -> int:
        return len(self.sites)


@dataclass(frozen=True)
class LocalTermsCompilation:
    """Dense local-block Hamiltonian representation for tensor-network backends."""

    space: object
    constant_shift: complex
    local_basis: SiteLocalBasis
    terms: tuple[LocalBlockTerm, ...]

    def term_map(self) -> dict[tuple[int, ...], np.ndarray]:
        """Return the compiled local terms as a support-to-matrix mapping."""
        return {term.sites: np.asarray(term.matrix).copy() for term in self.terms}

    def max_support_size(self) -> int:
        """Return the largest contiguous site support appearing in the Hamiltonian."""
        return max((term.support_size for term in self.terms), default=0)


def build_site_local_basis(space) -> SiteLocalBasis:
    """Construct site-local fermionic operator matrices for one `ElectronicSpace` site."""
    mode_labels = tuple((orbital, spin) for orbital in space.orbitals for spin in space.spins)
    nlocal = len(mode_labels)
    local_dim = 1 << nlocal

    identity = np.eye(local_dim, dtype=np.complex128)
    parity = np.zeros((local_dim, local_dim), dtype=np.complex128)
    create_ops = []
    destroy_ops = []

    for mode in range(nlocal):
        create = np.zeros((local_dim, local_dim), dtype=np.complex128)
        destroy = np.zeros((local_dim, local_dim), dtype=np.complex128)

        for state in range(local_dim):
            occupied = (state >> mode) & 1
            sign = -1.0 if _bit_parity(state & ((1 << mode) - 1)) else 1.0

            if occupied == 0:
                new_state = state | (1 << mode)
                create[new_state, state] = sign
            else:
                new_state = state & ~(1 << mode)
                destroy[new_state, state] = sign

            parity[state, state] = -1.0 if _bit_parity(state) else 1.0

        create_ops.append(create)
        destroy_ops.append(destroy)

    return SiteLocalBasis(
        space=space,
        local_dim=local_dim,
        mode_labels=mode_labels,
        identity=identity,
        parity=parity,
        create_ops=tuple(create_ops),
        destroy_ops=tuple(destroy_ops),
    )


def _local_mode_index(mode, *, num_spins: int) -> int:
    return int(mode.orbital_index * num_spins + mode.spin_index)


def _compile_one_term(operator: SymbolicOperator, basis: SiteLocalBasis, term, coeff: complex) -> LocalBlockTerm:
    modes = tuple(operator.space.unpack_mode(ladder.mode) for ladder in term.factors)
    sites_with_ops = tuple(mode.site for mode in modes)
    min_site = min(sites_with_ops)
    max_site = max(sites_with_ops)
    sites = tuple(range(min_site, max_site + 1))

    if len(term.factors) % 2 != 0:
        raise UnsupportedTransformError(
            "Local-term compilation only supports even-fermion-parity terms. "
            "Encountered a term with odd fermion parity."
        )

    block_dim = basis.local_dim ** len(sites)
    block_matrix = np.eye(block_dim, dtype=np.complex128)

    for ladder, mode in zip(term.factors, modes):
        site_matrices = []
        local_index = _local_mode_index(mode, num_spins=operator.space.num_spins)
        local_op = basis.create_ops[local_index] if ladder.action == "create" else basis.destroy_ops[local_index]
        for site in sites:
            if site < mode.site:
                site_matrices.append(basis.parity)
            elif site == mode.site:
                site_matrices.append(local_op)
            else:
                site_matrices.append(basis.identity)
        block_matrix = block_matrix @ _kron_all(site_matrices)

    return LocalBlockTerm(sites=sites, matrix=np.asarray(coeff, dtype=np.complex128) * block_matrix)


def compile_local_terms(hamiltonian) -> LocalTermsCompilation:
    """Compile a symbolic Hamiltonian into dense contiguous site-block operators.

    Each compiled term acts on the smallest contiguous block of sites spanning
    the symbolic term support. Jordan-Wigner parity strings are inserted on any
    intermediate sites so the resulting matrices are exact in the blocked
    site-local Fock basis.
    """
    operator = coerce_symbolic_operator(hamiltonian)
    basis = build_site_local_basis(operator.space)
    aggregated: dict[tuple[int, ...], np.ndarray] = {}

    for term, coeff in operator.iter_terms():
        compiled = _compile_one_term(operator, basis, term, coeff)
        support = compiled.sites
        if support in aggregated:
            aggregated[support] = aggregated[support] + compiled.matrix
        else:
            aggregated[support] = compiled.matrix

    terms = tuple(
        LocalBlockTerm(sites=support, matrix=matrix)
        for support, matrix in sorted(aggregated.items(), key=lambda item: (len(item[0]), item[0]))
        if np.max(np.abs(matrix)) > 1e-15
    )
    return LocalTermsCompilation(
        space=operator.space,
        constant_shift=operator.constant,
        local_basis=basis,
        terms=terms,
    )
