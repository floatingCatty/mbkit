from __future__ import annotations

import numpy as np

from ..soc import get_soc_matrix_cubic_basis
from .local_terms import _coerce_orbitals, _coerce_sites, _site_value


def soc(
    space,
    *,
    strength=1.0,
    matrix=None,
    orbital_type: str | None = None,
    sites="all",
    orbitals="all",
):
    selected_sites = _coerce_sites(space, sites)
    selected_orbitals = tuple(_coerce_orbitals(space, orbitals))

    if matrix is None:
        if orbital_type is None:
            raise ValueError("soc() requires either an explicit matrix or an orbital_type.")
        matrix = get_soc_matrix_cubic_basis(orbital_type)

    matrix = np.asarray(matrix, dtype=np.complex128)
    n_orbitals = len(selected_orbitals)
    expected_shape = (2 * n_orbitals, 2 * n_orbitals)
    if matrix.shape != expected_shape:
        raise ValueError(f"soc matrix must have shape {expected_shape}, got {matrix.shape}.")

    basis = [(orbital, "up") for orbital in selected_orbitals] + [
        (orbital, "down") for orbital in selected_orbitals
    ]

    H = 0
    for site in selected_sites:
        site_strength = _site_value(strength, site)
        if abs(site_strength) == 0:
            continue
        for row, (left_orbital, left_spin) in enumerate(basis):
            for col, (right_orbital, right_spin) in enumerate(basis):
                coeff = site_strength * matrix[row, col]
                if abs(coeff) == 0:
                    continue
                H += coeff * (
                    space.create(site, orbital=left_orbital, spin=left_spin)
                    @ space.destroy(site, orbital=right_orbital, spin=right_spin)
                )
    return H
