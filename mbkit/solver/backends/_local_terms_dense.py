"""Shared dense helper utilities for local-term tensor-network backends.

These helpers keep backend modules focused on the solver-specific bridge while
sharing the exact same dense embedding rules for compiled local Hamiltonians.
"""

from __future__ import annotations

import numpy as np

from ..compile import compile_local_terms


def dense_hamiltonian_from_local_terms(compiled) -> np.ndarray:
    """Embed a local-term compilation into the full site-local Fock basis."""
    dims = [compiled.local_basis.local_dim] * compiled.space.num_sites
    total_dim = int(np.prod(dims, dtype=int))
    dense = np.zeros((total_dim, total_dim), dtype=np.complex128)

    for term in compiled.terms:
        left_dim = compiled.local_basis.local_dim ** term.sites[0]
        right_sites = compiled.space.num_sites - term.sites[-1] - 1
        right_dim = compiled.local_basis.local_dim ** right_sites

        embedded = np.kron(
            np.eye(left_dim, dtype=np.complex128),
            np.kron(term.matrix, np.eye(right_dim, dtype=np.complex128)),
        )
        if embedded.shape != (total_dim, total_dim):
            raise ValueError(
                "Internal local-term embedding produced an inconsistent matrix shape: "
                f"expected {(total_dim, total_dim)!r}, got {embedded.shape!r}."
            )
        dense += embedded

    if abs(complex(compiled.constant_shift)) > 1e-15:
        dense += complex(compiled.constant_shift) * np.eye(total_dim, dtype=np.complex128)
    return dense


def as_real_if_possible(array: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Drop negligible imaginary parts to keep backend tensors real when possible."""
    if np.max(np.abs(np.imag(array))) <= atol:
        return np.asarray(np.real(array), dtype=float)
    return np.asarray(array, dtype=np.complex128)


def vector_expectation(state_vector: np.ndarray, matrix: np.ndarray):
    """Evaluate ``<psi|matrix|psi>`` for a dense state vector."""
    vector = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    return np.vdot(vector, matrix @ vector)


def compiled_is_complex(compiled, *, atol: float = 1e-12) -> bool:
    """Return whether a compiled local Hamiltonian contains essential phases."""
    if abs(complex(compiled.constant_shift).imag) > atol:
        return True
    return any(np.max(np.abs(np.imag(term.matrix))) > atol for term in compiled.terms)


def sector_penalty_dense(space, target_pair: tuple[int, int], *, penalty_scale: float) -> np.ndarray:
    """Build a quadratic penalty enforcing a target ``(n_up, n_down)`` sector."""
    up_number = dense_hamiltonian_from_local_terms(compile_local_terms(space.number_term(spin="up")))
    down_number = dense_hamiltonian_from_local_terms(compile_local_terms(space.number_term(spin="down")))
    full_dim = up_number.shape[0]
    eye = np.eye(full_dim, dtype=np.complex128)
    target_up, target_down = target_pair
    return penalty_scale * (
        (up_number - target_up * eye) @ (up_number - target_up * eye)
        + (down_number - target_down * eye) @ (down_number - target_down * eye)
    )
