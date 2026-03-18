"""Shared helpers for particle-number penalty MPOs.

These helpers avoid materializing dense full-Hilbert-space penalty operators
for backends that can work directly with MPOs. The core idea is to build small
bond-dimension MPOs for ``sum_i q_i`` and ``(sum_i q_i)^2`` where ``q_i`` is a
local onsite number operator.
"""

from __future__ import annotations

import numpy as np


def sector_penalty_operator(space, target_pair: tuple[int, int], *, penalty_scale: float):
    """Build the symbolic quadratic penalty for a target ``(n_up, n_down)`` sector."""
    up = space.number_term(spin="up")
    down = space.number_term(spin="down")
    delta_up = up - target_pair[0]
    delta_down = down - target_pair[1]
    return penalty_scale * ((delta_up @ delta_up) + (delta_down @ delta_down))


def local_number_matrix(basis, *, spin: str) -> np.ndarray:
    """Return the onsite number operator for one spin species on one site block."""
    matrix = np.zeros((basis.local_dim, basis.local_dim), dtype=np.complex128)
    for local_index, (_orbital, spin_label) in enumerate(basis.mode_labels):
        if spin_label != spin:
            continue
        matrix = matrix + basis.create_ops[local_index] @ basis.destroy_ops[local_index]
    return matrix


def compiled_row_sum_bound(compiled) -> float:
    """Return a cheap operator-norm upper bound from compiled local terms."""
    bound = float(abs(complex(compiled.constant_shift)))
    for term in compiled.terms:
        bound += float(np.max(np.sum(np.abs(term.matrix), axis=1)))
    return bound


def identity_mpo_tensors(local_dim: int, num_sites: int, *, coefficient: complex = 1.0) -> list[np.ndarray]:
    """Return rank-1 MPO tensors for ``coefficient * I`` in 4-leg form."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")
    tensors = []
    for site in range(num_sites):
        tensor = np.zeros((1, 1, local_dim, local_dim), dtype=np.complex128)
        prefactor = coefficient if site == 0 else 1.0
        tensor[0, 0, :, :] = prefactor * np.eye(local_dim, dtype=np.complex128)
        tensors.append(tensor)
    return tensors


def sum_mpo_tensors(local_op: np.ndarray, num_sites: int, *, coefficient: complex = 1.0) -> list[np.ndarray]:
    """Return MPO tensors for ``coefficient * sum_i local_op(i)`` in 4-leg form."""
    local_op = np.asarray(local_op, dtype=np.complex128)
    local_dim = local_op.shape[0]
    identity = np.eye(local_dim, dtype=np.complex128)

    if num_sites == 1:
        tensor = np.zeros((1, 1, local_dim, local_dim), dtype=np.complex128)
        tensor[0, 0, :, :] = coefficient * local_op
        return [tensor]

    tensors = []

    first = np.zeros((1, 2, local_dim, local_dim), dtype=np.complex128)
    first[0, 0, :, :] = coefficient * identity
    first[0, 1, :, :] = coefficient * local_op
    tensors.append(first)

    bulk = np.zeros((2, 2, local_dim, local_dim), dtype=np.complex128)
    bulk[0, 0, :, :] = identity
    bulk[0, 1, :, :] = local_op
    bulk[1, 1, :, :] = identity
    for _site in range(num_sites - 2):
        tensors.append(np.array(bulk, copy=True))

    last = np.zeros((2, 1, local_dim, local_dim), dtype=np.complex128)
    last[0, 0, :, :] = local_op
    last[1, 0, :, :] = identity
    tensors.append(last)

    return tensors


def square_sum_mpo_tensors(local_op: np.ndarray, num_sites: int, *, coefficient: complex = 1.0) -> list[np.ndarray]:
    """Return MPO tensors for ``coefficient * (sum_i local_op(i))^2`` in 4-leg form."""
    local_op = np.asarray(local_op, dtype=np.complex128)
    local_dim = local_op.shape[0]
    identity = np.eye(local_dim, dtype=np.complex128)
    local_op_sq = local_op @ local_op

    if num_sites == 1:
        tensor = np.zeros((1, 1, local_dim, local_dim), dtype=np.complex128)
        tensor[0, 0, :, :] = coefficient * local_op_sq
        return [tensor]

    tensors = []

    first = np.zeros((1, 3, local_dim, local_dim), dtype=np.complex128)
    first[0, 0, :, :] = coefficient * identity
    first[0, 1, :, :] = coefficient * local_op
    first[0, 2, :, :] = coefficient * local_op_sq
    tensors.append(first)

    bulk = np.zeros((3, 3, local_dim, local_dim), dtype=np.complex128)
    bulk[0, 0, :, :] = identity
    bulk[0, 1, :, :] = local_op
    bulk[0, 2, :, :] = local_op_sq
    bulk[1, 1, :, :] = identity
    bulk[1, 2, :, :] = 2.0 * local_op
    bulk[2, 2, :, :] = identity
    for _site in range(num_sites - 2):
        tensors.append(np.array(bulk, copy=True))

    last = np.zeros((3, 1, local_dim, local_dim), dtype=np.complex128)
    last[0, 0, :, :] = local_op_sq
    last[1, 0, :, :] = 2.0 * local_op
    last[2, 0, :, :] = identity
    tensors.append(last)

    return tensors
