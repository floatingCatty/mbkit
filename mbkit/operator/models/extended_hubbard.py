from __future__ import annotations

from .hopping import hopping as hopping_term
from .local_terms import _bond_sequence, _bond_value
from .observables import number


def extended_hubbard(
    space,
    *,
    hopping,
    onsite_U,
    intersite_V=None,
    bonds="all",
):
    H = -hopping_term(space, coeff=hopping, bonds=bonds, spin="both", orbitals="all", plus_hc=True)

    if abs(onsite_U) != 0:
        from .observables import double_occupancy

        H += onsite_U * double_occupancy(space)

    if intersite_V is None:
        return H

    for bond in _bond_sequence(space, bonds):
        coeff = _bond_value(intersite_V, bond)
        if abs(coeff) == 0:
            continue
        H += coeff * (
            number(space, sites=bond.left, spin="both")
            @ number(space, sites=bond.right, spin="both")
        )
    return H
