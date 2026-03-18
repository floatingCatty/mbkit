from __future__ import annotations

from collections.abc import Mapping

from ..operator.lattice import Bond


def coerce_shells(shells) -> tuple[int, ...] | None:
    if shells is None or shells == "all":
        return None
    if isinstance(shells, int):
        values = (int(shells),)
    else:
        values = tuple(int(value) for value in shells)
    if not values:
        return tuple()
    if min(values) < 1:
        raise ValueError("shells must contain positive integers.")
    return tuple(sorted(set(values)))


def coerce_sites(space, sites) -> tuple[int, ...]:
    if sites == "all":
        selected = tuple(range(space.num_sites))
    elif isinstance(sites, slice):
        selected = tuple(range(space.num_sites))[sites]
    elif isinstance(sites, range):
        selected = tuple(int(site) for site in sites)
    elif isinstance(sites, int):
        selected = (int(sites),)
    else:
        selected = tuple(int(site) for site in sites)

    for site in selected:
        if not (0 <= site < space.num_sites):
            raise ValueError(f"Site index {site} outside 0..{space.num_sites - 1}.")
    return selected


def coerce_orbitals(space, orbitals) -> tuple[str | int, ...]:
    if orbitals == "all":
        selected = tuple(space.orbitals)
    elif isinstance(orbitals, slice):
        selected = tuple(space.orbitals[orbitals])
    elif isinstance(orbitals, (str, int)):
        selected = (orbitals,)
    else:
        selected = tuple(orbitals)
    return tuple(space.orbitals[space.orbital_index(orbital)] for orbital in selected)


def _filter_bonds_by_shells(bonds: tuple[Bond, ...], shells) -> tuple[Bond, ...]:
    normalized_shells = coerce_shells(shells)
    if normalized_shells is None:
        return bonds
    return tuple(bond for bond in bonds if bond.shell in normalized_shells)


def coerce_bonds(space, bonds, shells=None) -> tuple[Bond, ...]:
    if space.lattice is None:
        raise ValueError("This builder requires an ElectronicSpace with a lattice.")
    all_bonds = space.lattice.bonds(shells=shells)
    if bonds == "all":
        return all_bonds
    if isinstance(bonds, slice):
        return _filter_bonds_by_shells(all_bonds[bonds], shells)
    if isinstance(bonds, Bond):
        return _filter_bonds_by_shells((bonds,), shells)
    if isinstance(bonds, str):
        return space.lattice.bonds(bonds, shells=shells)

    bonds = tuple(bonds)
    if not bonds:
        return tuple()
    if isinstance(bonds[0], Bond):
        return _filter_bonds_by_shells(tuple(bonds), shells)

    selected: list[Bond] = []
    for kind in bonds:
        selected.extend(space.lattice.bonds(str(kind), shells=shells))
    return tuple(selected)


def onsite_value(value, site: int, orbital: str | int | None = None):
    if isinstance(value, Mapping):
        if orbital is not None and (site, orbital) in value:
            return value[(site, orbital)]
        if site in value:
            return value[site]
        if orbital is not None and orbital in value:
            return value[orbital]
        return 0.0
    return value


def bond_value(value, bond: Bond):
    if isinstance(value, Mapping):
        if bond in value:
            return value[bond]
        if bond.shell is not None and (bond.shell, bond.kind) in value:
            return value[(bond.shell, bond.kind)]
        if bond.shell is not None and (bond.kind, bond.shell) in value:
            return value[(bond.kind, bond.shell)]
        if bond.shell is not None and bond.shell in value:
            return value[bond.shell]
        if bond.kind in value:
            return value[bond.kind]
        if (bond.left, bond.right) in value:
            return value[(bond.left, bond.right)]
        if (bond.right, bond.left) in value:
            return value[(bond.right, bond.left)]
        return 0.0
    return value


def orbital_pair_value(value, site: int, left_orbital, right_orbital):
    if isinstance(value, Mapping):
        keys = (
            (site, left_orbital, right_orbital),
            (site, right_orbital, left_orbital),
            (left_orbital, right_orbital),
            (right_orbital, left_orbital),
            site,
        )
        for key in keys:
            if key in value:
                return value[key]
        return 0.0
    return value


def coerce_orbital_pairs(space, orbitals="all", orbital_pairs=None) -> tuple[tuple[str | int, str | int], ...]:
    if orbital_pairs is not None:
        return tuple(
            (
                space.orbitals[space.orbital_index(left)],
                space.orbitals[space.orbital_index(right)],
            )
            for left, right in orbital_pairs
        )
    selected = coerce_orbitals(space, orbitals)
    return tuple((orbital, orbital) for orbital in selected)
