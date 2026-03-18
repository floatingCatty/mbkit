from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Bond:
    left: int
    right: int
    kind: str = "default"
    weight: complex = 1.0
    displacement: tuple[int, ...] | None = None


class Lattice:
    def __init__(
        self,
        *,
        num_sites: int,
        bonds: Sequence[Bond],
        site_positions: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self._num_sites = int(num_sites)
        self._bonds = tuple(self._validate_bond(bond) for bond in bonds)
        self._site_positions = tuple(tuple(position) for position in site_positions) if site_positions is not None else None

    @property
    def num_sites(self) -> int:
        return self._num_sites

    @property
    def site_positions(self) -> tuple[tuple[float, ...], ...] | None:
        return self._site_positions

    def bonds(self, kind: str | None = None) -> tuple[Bond, ...]:
        if kind is None:
            return self._bonds
        return tuple(bond for bond in self._bonds if bond.kind == kind)

    def adjacency(self, kind: str | None = None) -> np.ndarray:
        adj = np.zeros((self.num_sites, self.num_sites), dtype=np.complex128)
        for bond in self.bonds(kind):
            adj[bond.left, bond.right] += bond.weight
            adj[bond.right, bond.left] += np.conjugate(bond.weight)
        if np.allclose(adj.imag, 0.0):
            return adj.real
        return adj

    def _validate_bond(self, bond: Bond) -> Bond:
        if not (0 <= bond.left < self.num_sites and 0 <= bond.right < self.num_sites):
            raise ValueError(f"Bond {bond!r} references a site outside 0..{self.num_sites - 1}.")
        if bond.left == bond.right:
            raise ValueError(f"Bond {bond!r} must connect two distinct sites.")
        return bond


class GeneralLattice(Lattice):
    def __init__(
        self,
        *,
        num_sites: int,
        bonds: Sequence[Bond],
        site_positions: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=site_positions)


class LineLattice(Lattice):
    def __init__(self, length: int, *, boundary: str = "open") -> None:
        if length < 1:
            raise ValueError("LineLattice length must be positive.")
        if boundary not in {"open", "periodic"}:
            raise ValueError("LineLattice boundary must be 'open' or 'periodic'.")

        bonds = [Bond(site, site + 1, kind="nn") for site in range(length - 1)]
        if boundary == "periodic" and length > 2:
            bonds.append(Bond(length - 1, 0, kind="nn"))
        site_positions = [(float(site),) for site in range(length)]
        super().__init__(num_sites=length, bonds=bonds, site_positions=site_positions)


class SquareLattice(Lattice):
    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        boundary: tuple[str, str] = ("open", "open"),
        include_diagonals: bool = False,
    ) -> None:
        if nx < 1 or ny < 1:
            raise ValueError("SquareLattice dimensions must be positive.")
        bx, by = boundary
        if bx not in {"open", "periodic"} or by not in {"open", "periodic"}:
            raise ValueError("SquareLattice boundaries must be 'open' or 'periodic'.")

        bonds: list[Bond] = []
        site_positions = []

        def site_index(ix: int, iy: int) -> int:
            return ix * ny + iy

        for ix in range(nx):
            for iy in range(ny):
                site_positions.append((float(ix), float(iy)))
                if ix + 1 < nx:
                    bonds.append(Bond(site_index(ix, iy), site_index(ix + 1, iy), kind="horizontal"))
                elif bx == "periodic" and nx > 2:
                    bonds.append(Bond(site_index(ix, iy), site_index(0, iy), kind="horizontal", displacement=(1, 0)))

                if iy + 1 < ny:
                    bonds.append(Bond(site_index(ix, iy), site_index(ix, iy + 1), kind="vertical"))
                elif by == "periodic" and ny > 2:
                    bonds.append(Bond(site_index(ix, iy), site_index(ix, 0), kind="vertical", displacement=(0, 1)))

                if include_diagonals:
                    if ix + 1 < nx and iy + 1 < ny:
                        bonds.append(Bond(site_index(ix, iy), site_index(ix + 1, iy + 1), kind="diagonal"))
                    elif bx == "periodic" and by == "periodic" and nx > 2 and ny > 2:
                        bonds.append(Bond(site_index(ix, iy), site_index((ix + 1) % nx, (iy + 1) % ny), kind="diagonal", displacement=(1, 1)))

                    if ix + 1 < nx and iy - 1 >= 0:
                        bonds.append(Bond(site_index(ix, iy), site_index(ix + 1, iy - 1), kind="diagonal"))
                    elif bx == "periodic" and by == "periodic" and nx > 2 and ny > 2:
                        bonds.append(Bond(site_index(ix, iy), site_index((ix + 1) % nx, (iy - 1) % ny), kind="diagonal", displacement=(1, -1)))

        super().__init__(num_sites=nx * ny, bonds=bonds, site_positions=site_positions)
