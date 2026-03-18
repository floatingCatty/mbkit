from __future__ import annotations

from .local_terms import _coerce_orbitals, _coerce_sites


def number(space, *, sites="all", orbitals="all", spin: str = "both"):
    H = 0
    for site in _coerce_sites(space, sites):
        for orbital in _coerce_orbitals(space, orbitals):
            H += space.number(site, orbital=orbital, spin=spin)
    return H


def double_occupancy(space, *, sites="all", orbitals="all"):
    H = 0
    for site in _coerce_sites(space, sites):
        for orbital in _coerce_orbitals(space, orbitals):
            H += (
                space.number(site, orbital=orbital, spin="up")
                @ space.number(site, orbital=orbital, spin="down")
            )
    return H


def spin_z(space, *, sites="all", orbitals="all"):
    H = 0
    for site in _coerce_sites(space, sites):
        for orbital in _coerce_orbitals(space, orbitals):
            H += space.spin_z(site, orbital=orbital)
    return H


def spin_plus(space, *, sites="all", orbitals="all"):
    H = 0
    for site in _coerce_sites(space, sites):
        for orbital in _coerce_orbitals(space, orbitals):
            H += space.spin_plus(site, orbital=orbital)
    return H


def spin_minus(space, *, sites="all", orbitals="all"):
    H = 0
    for site in _coerce_sites(space, sites):
        for orbital in _coerce_orbitals(space, orbitals):
            H += space.spin_minus(site, orbital=orbital)
    return H


def spin_squared(space, *, sites="all", orbitals="all"):
    s_minus = spin_minus(space, sites=sites, orbitals=orbitals)
    s_plus = spin_plus(space, sites=sites, orbitals=orbitals)
    s_z = spin_z(space, sites=sites, orbitals=orbitals)
    return s_minus @ s_plus + s_z @ s_z + s_z
