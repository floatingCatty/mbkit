from __future__ import annotations

from .local_terms import _bond_sequence, _bond_value, _coerce_orbitals


def hopping(
    space,
    *,
    coeff,
    bonds="all",
    spin: str = "both",
    orbitals="all",
    orbital_pairs=None,
    plus_hc: bool = True,
):
    if orbital_pairs is None:
        selected_orbitals = tuple(_coerce_orbitals(space, orbitals))
        orbital_pairs = tuple((orbital, orbital) for orbital in selected_orbitals)
    else:
        orbital_pairs = tuple((left, right) for left, right in orbital_pairs)

    H = 0
    for bond in _bond_sequence(space, bonds):
        bond_coeff = _bond_value(coeff, bond) * bond.weight
        if abs(bond_coeff) == 0:
            continue
        for left_orbital, right_orbital in orbital_pairs:
            H += space.hopping(
                bond.left,
                bond.right,
                left_orbital=left_orbital,
                right_orbital=right_orbital,
                spin=spin,
                coeff=bond_coeff,
                plus_hc=plus_hc,
            )
    return H
