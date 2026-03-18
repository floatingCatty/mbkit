from __future__ import annotations

from .hopping import hopping
from .local_terms import _coerce_orbitals, chemical_potential


def hubbard(space, *, bonds="all", t=1.0, U=0.0, mu=None):
    H = -hopping(space, coeff=t, bonds=bonds, spin="both", orbitals="all", plus_hc=True)

    if abs(U) != 0:
        for site in range(space.num_sites):
            for orbital in _coerce_orbitals(space, "all"):
                H += U * (
                    space.number(site, orbital=orbital, spin="up")
                    @ space.number(site, orbital=orbital, spin="down")
                )

    if mu is not None:
        H += chemical_potential(space, mu=mu)
    return H
