from __future__ import annotations

from .local_terms import _coerce_orbitals, _coerce_sites, exchange, pair_hopping


def slater_kanamori(
    space,
    *,
    sites="all",
    orbitals="all",
    U=3.0,
    Up=0.0,
    J=0.0,
    Jp=0.0,
):
    selected_sites = _coerce_sites(space, sites)
    selected_orbitals = tuple(_coerce_orbitals(space, orbitals))

    H = 0
    for site in selected_sites:
        for orbital in selected_orbitals:
            H += U * (
                space.number(site, orbital=orbital, spin="up")
                @ space.number(site, orbital=orbital, spin="down")
            )

        for i, left in enumerate(selected_orbitals):
            for right in selected_orbitals[i + 1:]:
                if abs(Up) != 0:
                    for spin_left in ("up", "down"):
                        for spin_right in ("up", "down"):
                            H += Up * (
                                space.number(site, orbital=left, spin=spin_left)
                                @ space.number(site, orbital=right, spin=spin_right)
                            )
                if abs(J) != 0:
                    H += -J * exchange(space, site=site, left_orbital=left, right_orbital=right, coeff=1.0)
                if abs(Jp) != 0:
                    H += -Jp * pair_hopping(space, site=site, left_orbital=left, right_orbital=right, coeff=1.0)
    return H
