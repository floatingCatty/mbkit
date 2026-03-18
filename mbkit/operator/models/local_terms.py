from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..lattice import Bond


def _coerce_sites(space, sites):
    if sites == "all":
        return tuple(range(space.num_sites))
    if isinstance(sites, int):
        return (sites,)
    return tuple(int(site) for site in sites)


def _coerce_orbitals(space, orbitals):
    if orbitals == "all":
        return tuple(space.orbitals)
    if isinstance(orbitals, (int, str)):
        return (orbitals,)
    return tuple(orbitals)


def _bond_sequence(space, bonds):
    if space.lattice is None:
        raise ValueError("This builder requires an ElectronicSpace with a lattice.")
    if bonds == "all":
        return space.lattice.bonds()
    if isinstance(bonds, str):
        return space.lattice.bonds(bonds)
    bonds = tuple(bonds)
    if bonds and isinstance(bonds[0], Bond):
        return bonds
    selected = []
    for kind in bonds:
        selected.extend(space.lattice.bonds(kind))
    return tuple(selected)


def _site_value(value, site):
    if isinstance(value, Mapping):
        return value.get(site, 0.0)
    return value


def _bond_value(value, bond):
    if isinstance(value, Mapping):
        return value.get(bond.kind, 0.0)
    return value


def chemical_potential(
    space,
    *,
    mu,
    sites="all",
    orbitals="all",
    spin: str = "both",
):
    H = 0
    for site in _coerce_sites(space, sites):
        mu_site = _site_value(mu, site)
        if abs(mu_site) == 0:
            continue
        for orbital in _coerce_orbitals(space, orbitals):
            H += mu_site * space.number(site, orbital=orbital, spin=spin)
    return H


def density_density(
    space,
    *,
    coeff,
    left_site: int,
    right_site: int,
    left_orbital: str | int = 0,
    right_orbital: str | int = 0,
):
    return coeff * (
        space.number(left_site, orbital=left_orbital, spin="both")
        @ space.number(right_site, orbital=right_orbital, spin="both")
    )


def exchange(space, *, site: int, left_orbital, right_orbital, coeff):
    up_flip = (
        space.create(site, orbital=left_orbital, spin="up")
        @ space.destroy(site, orbital=left_orbital, spin="down")
        @ space.create(site, orbital=right_orbital, spin="down")
        @ space.destroy(site, orbital=right_orbital, spin="up")
    )
    down_flip = (
        space.create(site, orbital=left_orbital, spin="down")
        @ space.destroy(site, orbital=left_orbital, spin="up")
        @ space.create(site, orbital=right_orbital, spin="up")
        @ space.destroy(site, orbital=right_orbital, spin="down")
    )
    return coeff * (up_flip + down_flip)


def pair_hopping(space, *, site: int, left_orbital, right_orbital, coeff):
    transfer = (
        space.create(site, orbital=left_orbital, spin="up")
        @ space.create(site, orbital=left_orbital, spin="down")
        @ space.destroy(site, orbital=right_orbital, spin="up")
        @ space.destroy(site, orbital=right_orbital, spin="down")
    )
    return coeff * (transfer + transfer.adjoint())
