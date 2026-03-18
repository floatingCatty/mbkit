"""Geometry primitives used by the operator-building layer."""

from __future__ import annotations

from itertools import product
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Bond:
    """A directed bond record used by space-bound model builders.

    `kind` and `weight` let high-level constructors select or scale subsets of
    bonds without exposing backend-specific neighbor tables to users.
    """

    left: int
    right: int
    kind: str = "default"
    weight: complex = 1.0
    displacement: tuple[int, ...] | None = None
    shell: int | None = None
    distance: float | None = None


@dataclass(frozen=True)
class _CellBondSpec:
    """Internal bond recipe for lattices defined from a unit cell."""

    left_basis: int
    right_basis: int
    shift: tuple[int, ...]
    kind: str
    weight: complex = 1.0


@dataclass(frozen=True)
class _RelativeBond:
    """Internal shell-classified relative bond on the infinite lattice."""

    left_basis: int
    right_basis: int
    shift: tuple[int, ...]
    vector: tuple[float, ...]
    distance: float
    hint_kind: str | None = None
    weight: complex = 1.0


class Lattice:
    """Minimal lattice/graph container for operator construction.

    The class intentionally stays lightweight: it stores sites, typed bonds, and
    optional coordinates. More advanced physics lives in `ElectronicSpace` and
    model constructors rather than in the geometry container itself.
    """

    def __init__(
        self,
        *,
        num_sites: int,
        bonds: Sequence[Bond],
        site_positions: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self._num_sites = int(num_sites)
        self._bonds = tuple(self._validate_bond(self._coerce_bond(bond)) for bond in bonds)
        self._site_positions = tuple(tuple(position) for position in site_positions) if site_positions is not None else None

    @property
    def num_sites(self) -> int:
        return self._num_sites

    @property
    def site_positions(self) -> tuple[tuple[float, ...], ...] | None:
        return self._site_positions

    def bonds(self, kind: str | Sequence[str] | None = None, shells=None) -> tuple[Bond, ...]:
        """Return stored bonds filtered by kind and/or neighbor shell."""
        selected = self._bonds
        if kind is not None:
            kinds = (str(kind),) if isinstance(kind, str) else tuple(str(entry) for entry in kind)
            selected = tuple(bond for bond in selected if bond.kind in kinds)
        normalized_shells = _normalize_shells(shells)
        if normalized_shells is not None:
            selected = tuple(bond for bond in selected if bond.shell in normalized_shells)
        return selected

    def bond_kinds(self, *, shells=None) -> tuple[str, ...]:
        return tuple(sorted({bond.kind for bond in self.bonds(shells=shells)}))

    def available_shells(self) -> tuple[int, ...]:
        return tuple(sorted({bond.shell for bond in self._bonds if bond.shell is not None}))

    def bond_summary(self, *, shells=None) -> dict[int | None, dict[str, int]]:
        """Return a shell-first count summary of the stored bond families."""
        summary: dict[int | None, dict[str, int]] = {}
        for bond in self.bonds(shells=shells):
            shell_summary = summary.setdefault(bond.shell, {})
            shell_summary[bond.kind] = shell_summary.get(bond.kind, 0) + 1
        return summary

    def adjacency(self, kind: str | Sequence[str] | None = None, shells=None) -> np.ndarray:
        """Return the Hermitian adjacency matrix implied by the stored bonds."""
        adj = np.zeros((self.num_sites, self.num_sites), dtype=np.complex128)
        for bond in self.bonds(kind=kind, shells=shells):
            adj[bond.left, bond.right] += bond.weight
            adj[bond.right, bond.left] += np.conjugate(bond.weight)
        if np.allclose(adj.imag, 0.0):
            return adj.real
        return adj

    def _coerce_bond(self, bond) -> Bond:
        if isinstance(bond, Bond):
            return bond
        if isinstance(bond, Sequence) and not isinstance(bond, (str, bytes)):
            pieces = tuple(bond)
            if len(pieces) == 2:
                return Bond(int(pieces[0]), int(pieces[1]))
            if len(pieces) == 3:
                return Bond(int(pieces[0]), int(pieces[1]), kind=str(pieces[2]))
            if len(pieces) == 4:
                return Bond(int(pieces[0]), int(pieces[1]), kind=str(pieces[2]), weight=pieces[3])
            if len(pieces) == 5:
                displacement = None if pieces[4] is None else tuple(int(value) for value in pieces[4])
                return Bond(
                    int(pieces[0]),
                    int(pieces[1]),
                    kind=str(pieces[2]),
                    weight=pieces[3],
                    displacement=displacement,
                )
            if len(pieces) == 6:
                displacement = None if pieces[4] is None else tuple(int(value) for value in pieces[4])
                shell = None if pieces[5] is None else int(pieces[5])
                return Bond(
                    int(pieces[0]),
                    int(pieces[1]),
                    kind=str(pieces[2]),
                    weight=pieces[3],
                    displacement=displacement,
                    shell=shell,
                )
            if len(pieces) == 7:
                displacement = None if pieces[4] is None else tuple(int(value) for value in pieces[4])
                shell = None if pieces[5] is None else int(pieces[5])
                distance = None if pieces[6] is None else float(pieces[6])
                return Bond(
                    int(pieces[0]),
                    int(pieces[1]),
                    kind=str(pieces[2]),
                    weight=pieces[3],
                    displacement=displacement,
                    shell=shell,
                    distance=distance,
                )
        raise TypeError(
            "Bonds must be Bond objects or tuples like "
            "(left, right), (left, right, kind), (left, right, kind, weight[, displacement]), "
            "or include shell/distance metadata as extra tuple entries."
        )

    def _validate_bond(self, bond: Bond) -> Bond:
        if not (0 <= bond.left < self.num_sites and 0 <= bond.right < self.num_sites):
            raise ValueError(f"Bond {bond!r} references a site outside 0..{self.num_sites - 1}.")
        if bond.left == bond.right:
            raise ValueError(f"Bond {bond!r} must connect two distinct sites.")
        return bond


class GeneralLattice(Lattice):
    """Lattice built directly from explicit bond data."""

    def __init__(
        self,
        *,
        num_sites: int,
        bonds: Sequence[Bond],
        site_positions: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=site_positions)


def _normalize_shells(shells) -> tuple[int, ...] | None:
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


def _canonical_relative_key(left_basis: int, right_basis: int, shift: tuple[int, ...]) -> tuple[tuple[int, int, tuple[int, ...]], int]:
    reverse = (right_basis, left_basis, tuple(-entry for entry in shift))
    forward = (left_basis, right_basis, shift)
    if forward <= reverse:
        return forward, 1
    return reverse, -1


def _collect_shell_distances(candidates: Sequence[_RelativeBond], *, tol: float = 1e-9) -> list[float]:
    distances: list[float] = []
    for distance in sorted(candidate.distance for candidate in candidates):
        if not distances or not math.isclose(distance, distances[-1], rel_tol=tol, abs_tol=tol):
            distances.append(distance)
    return distances


def _shell_index(distance: float, shell_distances: Sequence[float], *, tol: float = 1e-9) -> int | None:
    for index, reference in enumerate(shell_distances, start=1):
        if math.isclose(distance, reference, rel_tol=tol, abs_tol=tol):
            return index
    return None


def _coerce_boundaries(boundary, *, ndim: int, lattice_name: str) -> tuple[str, ...]:
    if ndim == 1 and isinstance(boundary, str):
        values = (boundary,)
    else:
        values = tuple(str(entry) for entry in boundary)
    if len(values) != ndim:
        raise ValueError(f"{lattice_name} expects {ndim} boundary entries.")
    for value in values:
        if value not in {"open", "periodic"}:
            raise ValueError(f"{lattice_name} boundaries must be 'open' or 'periodic'.")
    return values


def _build_unit_cell_lattice(
    *,
    shape: Sequence[int],
    basis_positions: Sequence[Sequence[float]],
    preferred_bonds: Sequence[_CellBondSpec],
    boundary,
    lattice_name: str,
    lattice_vectors: Sequence[Sequence[float]] | None = None,
    max_shell: int = 1,
    kind_resolver: Callable[[int, _RelativeBond], str | None] | None = None,
) -> tuple[int, list[Bond], list[tuple[float, ...]]]:
    """Construct bonds and positions from a Bravais lattice with a finite basis."""

    shape = tuple(int(length) for length in shape)
    if any(length < 1 for length in shape):
        raise ValueError(f"{lattice_name} dimensions must be positive.")
    if max_shell < 1:
        raise ValueError(f"{lattice_name} max_shell must be positive.")
    ndim = len(shape)
    boundaries = _coerce_boundaries(boundary, ndim=ndim, lattice_name=lattice_name)

    if lattice_vectors is None:
        lattice_vectors = tuple(
            tuple(1.0 if axis == vector_axis else 0.0 for axis in range(ndim)) for vector_axis in range(ndim)
        )
    lattice_vectors = tuple(tuple(float(value) for value in vector) for vector in lattice_vectors)
    if len(lattice_vectors) != ndim:
        raise ValueError(f"{lattice_name} needs {ndim} lattice vectors.")

    basis_positions = tuple(tuple(float(value) for value in position) for position in basis_positions)
    num_basis = len(basis_positions)
    if num_basis < 1:
        raise ValueError(f"{lattice_name} needs at least one basis position.")

    position_dim = max(
        max((len(vector) for vector in lattice_vectors), default=ndim),
        max((len(position) for position in basis_positions), default=ndim),
    )

    normalized_vectors = tuple(vector + (0.0,) * (position_dim - len(vector)) for vector in lattice_vectors)
    normalized_basis = tuple(position + (0.0,) * (position_dim - len(position)) for position in basis_positions)

    strides = [int(np.prod(shape[index + 1 :], dtype=int)) if index + 1 < ndim else 1 for index in range(ndim)]

    def linear_cell_index(cell: tuple[int, ...]) -> int:
        return sum(cell[axis] * strides[axis] for axis in range(ndim))

    def site_index(cell: tuple[int, ...], basis: int) -> int:
        return linear_cell_index(cell) * num_basis + basis

    positions: list[tuple[float, ...]] = []
    for cell in np.ndindex(shape):
        cell_origin = [0.0] * position_dim
        for axis, coordinate in enumerate(cell):
            vector = normalized_vectors[axis]
            for position_axis, value in enumerate(vector):
                cell_origin[position_axis] += coordinate * value
        for basis_position in normalized_basis:
            positions.append(
                tuple(cell_origin[position_axis] + basis_position[position_axis] for position_axis in range(position_dim))
            )

    preferred_map: dict[tuple[int, int, tuple[int, ...]], tuple[str, complex]] = {}
    for spec in preferred_bonds:
        shift = tuple(int(entry) for entry in spec.shift)
        preferred_map[(spec.left_basis, spec.right_basis, shift)] = (spec.kind, spec.weight)
        preferred_map[(spec.right_basis, spec.left_basis, tuple(-entry for entry in shift))] = (spec.kind, spec.weight)

    search_radius = max(1, 2 * max_shell)
    relative_candidates: dict[tuple[int, int, tuple[int, ...]], _RelativeBond] = {}
    shifts = product(range(-search_radius, search_radius + 1), repeat=ndim)
    for shift_entries in shifts:
        shift = tuple(int(entry) for entry in shift_entries)
        for left_basis in range(num_basis):
            for right_basis in range(num_basis):
                if left_basis == right_basis and all(entry == 0 for entry in shift):
                    continue

                displacement_vector = [0.0] * position_dim
                for axis, cell_shift in enumerate(shift):
                    vector = normalized_vectors[axis]
                    for position_axis, value in enumerate(vector):
                        displacement_vector[position_axis] += cell_shift * value
                for position_axis in range(position_dim):
                    displacement_vector[position_axis] += (
                        normalized_basis[right_basis][position_axis] - normalized_basis[left_basis][position_axis]
                    )

                distance = math.sqrt(sum(value * value for value in displacement_vector))
                if distance <= 1e-12:
                    continue

                canonical_key, sign = _canonical_relative_key(left_basis, right_basis, shift)
                canonical_vector = tuple(sign * value for value in displacement_vector)
                preferred = preferred_map.get((left_basis, right_basis, shift))
                candidate = _RelativeBond(
                    left_basis=canonical_key[0],
                    right_basis=canonical_key[1],
                    shift=canonical_key[2],
                    vector=canonical_vector,
                    distance=distance,
                    hint_kind=None if preferred is None else preferred[0],
                    weight=1.0 if preferred is None else preferred[1],
                )
                existing = relative_candidates.get(canonical_key)
                if existing is None or (existing.hint_kind is None and candidate.hint_kind is not None):
                    relative_candidates[canonical_key] = candidate

    shell_distances = _collect_shell_distances(tuple(relative_candidates.values()))
    if len(shell_distances) < max_shell:
        raise ValueError(f"{lattice_name} could not classify {max_shell} neighbor shells.")
    shell_distances = shell_distances[:max_shell]

    selected_candidates: list[tuple[int, _RelativeBond, str]] = []
    for candidate in relative_candidates.values():
        shell = _shell_index(candidate.distance, shell_distances)
        if shell is None:
            continue
        kind = candidate.hint_kind
        if kind is None and kind_resolver is not None:
            kind = kind_resolver(shell, candidate)
        if kind is None:
            kind = f"shell_{shell}"
        selected_candidates.append((shell, candidate, kind))
    selected_candidates.sort(key=lambda item: (item[0], item[1].distance, item[2], item[1].shift))

    bonds: list[Bond] = []
    seen: set[tuple[int, int, str]] = set()
    for cell in np.ndindex(shape):
        cell_tuple = tuple(int(entry) for entry in cell)
        for shell, candidate, kind in selected_candidates:
            target = list(cell_tuple)
            wrapped = False
            valid = True
            for axis, delta in enumerate(candidate.shift):
                coordinate = target[axis] + delta
                if 0 <= coordinate < shape[axis]:
                    target[axis] = coordinate
                    continue
                if boundaries[axis] == "periodic" and shape[axis] > 1:
                    target[axis] = coordinate % shape[axis]
                    wrapped = True
                else:
                    valid = False
                    break
            if not valid:
                continue

            left = site_index(cell_tuple, candidate.left_basis)
            right = site_index(tuple(target), candidate.right_basis)
            if left == right:
                continue

            key = (min(left, right), max(left, right), kind)
            if key in seen:
                continue
            seen.add(key)

            bonds.append(
                Bond(
                    left,
                    right,
                    kind=kind,
                    weight=candidate.weight,
                    displacement=candidate.shift if wrapped else None,
                    shell=shell,
                    distance=candidate.distance,
                )
            )
    return int(np.prod(shape, dtype=int)) * num_basis, bonds, positions


class LineLattice(Lattice):
    """1D nearest-neighbor chain lattice."""

    def __init__(self, length: int, *, boundary: str = "open", max_shell: int = 1) -> None:
        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(length,),
            basis_positions=((0.0,),),
            preferred_bonds=(_CellBondSpec(0, 0, (1,), "nn"),),
            boundary=boundary,
            lattice_name="LineLattice",
            max_shell=max_shell,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)


class LadderLattice(Lattice):
    """Multi-leg ladder with `leg` and `rung` bond labels.

    The common ladder use case is a two-leg system, but the `legs` parameter
    keeps the class flexible without changing the public one-line helper style.
    """

    def __init__(self, length: int, *, legs: int = 2, boundary: str = "open", max_shell: int = 1) -> None:
        if legs < 2:
            raise ValueError("LadderLattice needs at least two legs.")

        preferred_bonds = tuple(
            [_CellBondSpec(leg, leg, (1,), "leg") for leg in range(legs)]
            + [_CellBondSpec(leg, leg + 1, (0,), "rung") for leg in range(legs - 1)]
        )

        def ladder_kind(shell: int, candidate: _RelativeBond) -> str | None:
            if candidate.left_basis == candidate.right_basis:
                return "leg"
            if candidate.shift == (0,):
                return "rung"
            return None

        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(length,),
            basis_positions=tuple((0.0, float(leg)) for leg in range(legs)),
            preferred_bonds=preferred_bonds,
            boundary=boundary,
            lattice_name="LadderLattice",
            lattice_vectors=((1.0, 0.0),),
            max_shell=max_shell,
            kind_resolver=ladder_kind,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)


class SquareLattice(Lattice):
    """2D square lattice with optional periodic boundaries and diagonals."""

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        boundary: tuple[str, str] = ("open", "open"),
        include_diagonals: bool = False,
        max_shell: int = 1,
    ) -> None:
        effective_max_shell = max(max_shell, 2 if include_diagonals else 1)

        def square_kind(shell: int, candidate: _RelativeBond) -> str | None:
            shift_x, shift_y = candidate.shift
            if shift_y == 0 and shift_x != 0:
                return "horizontal"
            if shift_x == 0 and shift_y != 0:
                return "vertical"
            if abs(shift_x) == abs(shift_y) and shift_x != 0:
                return "diagonal"
            return None

        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(nx, ny),
            basis_positions=((0.0, 0.0),),
            preferred_bonds=(
                _CellBondSpec(0, 0, (1, 0), "horizontal"),
                _CellBondSpec(0, 0, (0, 1), "vertical"),
                _CellBondSpec(0, 0, (1, 1), "diagonal"),
                _CellBondSpec(0, 0, (1, -1), "diagonal"),
            ),
            boundary=boundary,
            lattice_name="SquareLattice",
            max_shell=effective_max_shell,
            kind_resolver=square_kind,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)


class RectangularLattice(SquareLattice):
    """Alias of :class:`SquareLattice` with a more explicit rectangular name."""


class TriangularLattice(Lattice):
    """2D triangular lattice with three nearest-neighbor bond directions."""

    def __init__(self, nx: int, ny: int, *, boundary: tuple[str, str] = ("open", "open"), max_shell: int = 1) -> None:
        def triangular_kind(shell: int, candidate: _RelativeBond) -> str | None:
            shift_x, shift_y = candidate.shift
            if shift_y == 0 and shift_x != 0:
                return "horizontal"
            if shift_x == 0 and shift_y != 0:
                return "up_right"
            if shift_x == -shift_y and shift_x != 0:
                return "down_right"
            return None

        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(nx, ny),
            basis_positions=((0.0, 0.0),),
            preferred_bonds=(
                _CellBondSpec(0, 0, (1, 0), "horizontal"),
                _CellBondSpec(0, 0, (0, 1), "up_right"),
                _CellBondSpec(0, 0, (1, -1), "down_right"),
            ),
            boundary=boundary,
            lattice_name="TriangularLattice",
            lattice_vectors=((1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)),
            max_shell=max_shell,
            kind_resolver=triangular_kind,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)


class HoneycombLattice(Lattice):
    """2D honeycomb lattice with a two-site basis and three bond directions."""

    def __init__(self, nx: int, ny: int, *, boundary: tuple[str, str] = ("open", "open"), max_shell: int = 1) -> None:
        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(nx, ny),
            basis_positions=((0.0, 0.0), (0.0, 1.0 / math.sqrt(3.0))),
            preferred_bonds=(
                _CellBondSpec(0, 1, (0, 0), "vertical"),
                _CellBondSpec(0, 1, (0, -1), "down_left"),
                _CellBondSpec(0, 1, (1, -1), "down_right"),
            ),
            boundary=boundary,
            lattice_name="HoneycombLattice",
            lattice_vectors=((1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)),
            max_shell=max_shell,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)


class KagomeLattice(Lattice):
    """2D kagome lattice with a three-site basis on a triangular Bravais lattice."""

    def __init__(self, nx: int, ny: int, *, boundary: tuple[str, str] = ("open", "open"), max_shell: int = 1) -> None:
        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(nx, ny),
            basis_positions=((0.0, 0.0), (0.5, 0.0), (0.25, math.sqrt(3.0) / 4.0)),
            preferred_bonds=(
                _CellBondSpec(0, 1, (0, 0), "ab"),
                _CellBondSpec(0, 2, (0, 0), "ac"),
                _CellBondSpec(1, 2, (0, 0), "bc"),
                _CellBondSpec(0, 1, (-1, 0), "ab"),
                _CellBondSpec(0, 2, (0, -1), "ac"),
                _CellBondSpec(1, 2, (1, -1), "bc"),
            ),
            boundary=boundary,
            lattice_name="KagomeLattice",
            lattice_vectors=((1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)),
            max_shell=max_shell,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)


class CubicLattice(Lattice):
    """3D simple cubic lattice with nearest-neighbor x/y/z bonds."""

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        *,
        boundary: tuple[str, str, str] = ("open", "open", "open"),
        max_shell: int = 1,
    ) -> None:
        def cubic_kind(shell: int, candidate: _RelativeBond) -> str | None:
            components = tuple(abs(entry) for entry in candidate.shift)
            if sum(1 for entry in components if entry != 0) == 1:
                axis = components.index(next(entry for entry in components if entry != 0))
                return ("x", "y", "z")[axis]
            return None

        num_sites, bonds, positions = _build_unit_cell_lattice(
            shape=(nx, ny, nz),
            basis_positions=((0.0, 0.0, 0.0),),
            preferred_bonds=(
                _CellBondSpec(0, 0, (1, 0, 0), "x"),
                _CellBondSpec(0, 0, (0, 1, 0), "y"),
                _CellBondSpec(0, 0, (0, 0, 1), "z"),
            ),
            boundary=boundary,
            lattice_name="CubicLattice",
            max_shell=max_shell,
            kind_resolver=cubic_kind,
        )
        super().__init__(num_sites=num_sites, bonds=bonds, site_positions=positions)
