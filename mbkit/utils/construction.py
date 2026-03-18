"""Internal construction utilities behind the public `ElectronicSpace` API.

The public design intentionally avoids exposing a builder object. This module
still centralizes the repeated coupling/onsite assembly logic so the
space-bound methods stay short and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..operator.operator import Operator
from .selection import bond_value, coerce_bonds, coerce_orbital_pairs, coerce_orbitals, coerce_sites, onsite_value


@dataclass(frozen=True)
class ConstructionRecord:
    """Debug record describing one construction step performed by the engine."""

    kind: str
    category: str | None
    details: dict[str, object]


class ConstructionEngine:
    """Internal coupling-style construction engine for space-bound model methods.

    New developers should treat this as implementation machinery, not part of the
    public API. Its job is to keep high-level model methods consistent while the
    user-facing surface remains the much simpler `space.method(...)` style.
    """

    def __init__(self, space) -> None:
        self.space = space
        self._constant = 0.0 + 0.0j
        self._operators: list[Operator] = []
        self._records: list[ConstructionRecord] = []

    @property
    def records(self) -> tuple[ConstructionRecord, ...]:
        return tuple(self._records)

    def store(self, operator, *, kind: str, category: str | None, details: dict[str, object]) -> None:
        """Accumulate either a scalar shift or a symbolic operator term."""
        if isinstance(operator, (int, float, complex, np.number)):
            value = complex(operator)
            if abs(value) <= 1e-15:
                return
            self._constant += value
        else:
            simplified = operator.simplify()
            if abs(simplified.constant) <= 1e-15 and not simplified.terms:
                return
            self._operators.append(simplified)
        self._records.append(ConstructionRecord(kind=kind, category=category, details=dict(details)))

    def add_constant(self, value, category: str | None = None, **details) -> None:
        self.store(value, kind="constant", category=category, details=details)

    def _onsite_operator(self, opname, *, site: int, orbital, spin: str):
        if callable(opname):
            return opname(self.space, site=site, orbital=orbital, spin=spin)
        if opname == "number":
            return self.space.number(site, orbital=orbital, spin=spin)
        if opname == "spin_z":
            return self.space.spin_z(site, orbital=orbital)
        if opname == "spin_plus":
            return self.space.spin_plus(site, orbital=orbital)
        if opname == "spin_minus":
            return self.space.spin_minus(site, orbital=orbital)
        if opname == "double_occupancy":
            return self.space.number(site, orbital=orbital, spin="up") @ self.space.number(
                site,
                orbital=orbital,
                spin="down",
            )
        raise ValueError(f"Unsupported onsite operator name {opname!r}.")

    def add_onsite(
        self,
        coeff,
        opname,
        *,
        sites="all",
        orbitals="all",
        spin: str = "both",
        category: str | None = None,
        plus_hc: bool = False,
    ) -> None:
        """Expand a site-local term over selected sites/orbitals."""
        for site in coerce_sites(self.space, sites):
            for orbital in coerce_orbitals(self.space, orbitals):
                value = onsite_value(coeff, site, orbital)
                if abs(complex(value)) <= 1e-15:
                    continue
                term = self._onsite_operator(opname, site=site, orbital=orbital, spin=spin)
                if plus_hc:
                    term = term + term.adjoint()
                self.store(
                    value * term,
                    kind="onsite",
                    category=category,
                    details={"site": site, "orbital": orbital, "spin": spin, "opname": getattr(opname, "__name__", opname)},
                )

    def add_coupling(
        self,
        coeff,
        coupling,
        *,
        bonds="all",
        shells=None,
        orbitals="all",
        orbital_pairs=None,
        spin: str = "both",
        category: str | None = None,
        plus_hc: bool = False,
    ) -> None:
        """Expand a two-site term over selected bonds and orbital pairs."""
        orbital_pairs = coerce_orbital_pairs(self.space, orbitals=orbitals, orbital_pairs=orbital_pairs)
        for bond in coerce_bonds(self.space, bonds, shells=shells):
            value = bond_value(coeff, bond) * bond.weight
            if abs(complex(value)) <= 1e-15:
                continue
            for left_orbital, right_orbital in orbital_pairs:
                if callable(coupling):
                    term = coupling(
                        self.space,
                        bond=bond,
                        left_orbital=left_orbital,
                        right_orbital=right_orbital,
                        spin=spin,
                    )
                elif coupling == "hopping":
                    term = self.space.hopping(
                        bond.left,
                        bond.right,
                        left_orbital=left_orbital,
                        right_orbital=right_orbital,
                        spin=spin,
                        coeff=1.0,
                        plus_hc=plus_hc,
                    )
                elif coupling == "density_density":
                    term = self.space.number(bond.left, orbital=left_orbital, spin=spin) @ self.space.number(
                        bond.right,
                        orbital=right_orbital,
                        spin=spin,
                    )
                    if plus_hc:
                        term = term + term.adjoint()
                elif coupling == "pairing":
                    term = self.space.create(bond.left, orbital=left_orbital, spin="up") @ self.space.create(
                        bond.right,
                        orbital=right_orbital,
                        spin="down",
                    )
                    if plus_hc:
                        term = term + term.adjoint()
                elif coupling == "current":
                    forward = self.space.hopping(
                        bond.left,
                        bond.right,
                        left_orbital=left_orbital,
                        right_orbital=right_orbital,
                        spin=spin,
                        coeff=1.0,
                        plus_hc=False,
                    )
                    term = 1.0j * (forward - forward.adjoint())
                else:
                    raise ValueError(f"Unsupported coupling kind {coupling!r}.")
                self.store(
                    value * term,
                    kind="coupling",
                    category=category,
                    details={
                        "bond": (bond.left, bond.right),
                        "bond_kind": bond.kind,
                        "bond_shell": bond.shell,
                        "left_orbital": left_orbital,
                        "right_orbital": right_orbital,
                        "spin": spin,
                        "coupling": getattr(coupling, "__name__", coupling),
                    },
                )

    def add_multi_coupling(
        self,
        coeff,
        coupling,
        *,
        sites="all",
        orbital_pairs,
        category: str | None = None,
        plus_hc: bool = False,
    ) -> None:
        """Expand a local multi-orbital quartic term over selected sites."""
        for site in coerce_sites(self.space, sites):
            value = onsite_value(coeff, site)
            if abs(complex(value)) <= 1e-15:
                continue
            for left_orbital, right_orbital in tuple(orbital_pairs):
                if callable(coupling):
                    term = coupling(
                        self.space,
                        site=site,
                        left_orbital=left_orbital,
                        right_orbital=right_orbital,
                    )
                elif coupling == "exchange":
                    up_flip = (
                        self.space.create(site, orbital=left_orbital, spin="up")
                        @ self.space.destroy(site, orbital=left_orbital, spin="down")
                        @ self.space.create(site, orbital=right_orbital, spin="down")
                        @ self.space.destroy(site, orbital=right_orbital, spin="up")
                    )
                    down_flip = (
                        self.space.create(site, orbital=left_orbital, spin="down")
                        @ self.space.destroy(site, orbital=left_orbital, spin="up")
                        @ self.space.create(site, orbital=right_orbital, spin="up")
                        @ self.space.destroy(site, orbital=right_orbital, spin="down")
                    )
                    term = up_flip + down_flip
                elif coupling == "pair_hopping":
                    transfer = (
                        self.space.create(site, orbital=left_orbital, spin="up")
                        @ self.space.create(site, orbital=left_orbital, spin="down")
                        @ self.space.destroy(site, orbital=right_orbital, spin="up")
                        @ self.space.destroy(site, orbital=right_orbital, spin="down")
                    )
                    term = transfer + transfer.adjoint()
                else:
                    raise ValueError(f"Unsupported multi-coupling kind {coupling!r}.")
                if plus_hc:
                    term = term + term.adjoint()
                self.store(
                    value * term,
                    kind="multi_coupling",
                    category=category,
                    details={
                        "site": site,
                        "left_orbital": left_orbital,
                        "right_orbital": right_orbital,
                        "coupling": getattr(coupling, "__name__", coupling),
                    },
                )

    def add_local_matrix(
        self,
        matrix,
        *,
        strength=1.0,
        sites="all",
        orbitals="all",
        spins: tuple[str, ...] = ("up", "down"),
        category: str | None = None,
        plus_hc: bool = False,
    ) -> None:
        """Expand a local single-particle matrix in orbital-spin space."""
        spins = _coerce_spins(self.space, spins)
        matrix = np.asarray(matrix, dtype=np.complex128)
        selected_orbitals = tuple(coerce_orbitals(self.space, orbitals))
        basis = [(orbital, spin) for orbital in selected_orbitals for spin in spins]
        if matrix.shape != (len(basis), len(basis)):
            raise ValueError(f"local matrix must have shape {(len(basis), len(basis))}, got {matrix.shape}.")

        for site in coerce_sites(self.space, sites):
            site_strength = onsite_value(strength, site)
            if abs(complex(site_strength)) <= 1e-15:
                continue
            for row, (left_orbital, left_spin) in enumerate(basis):
                for col, (right_orbital, right_spin) in enumerate(basis):
                    coeff = site_strength * matrix[row, col]
                    if abs(coeff) <= 1e-15:
                        continue
                    term = self.space.create(site, orbital=left_orbital, spin=left_spin) @ self.space.destroy(
                        site,
                        orbital=right_orbital,
                        spin=right_spin,
                    )
                    if plus_hc:
                        term = term + term.adjoint()
                    self.store(
                        coeff * term,
                        kind="local_matrix",
                        category=category,
                        details={
                            "site": site,
                            "left_orbital": left_orbital,
                            "left_spin": left_spin,
                            "right_orbital": right_orbital,
                            "right_spin": right_spin,
                        },
                    )

    def add_local_interaction_tensor(
        self,
        tensor,
        *,
        strength=1.0,
        sites="all",
        orbitals="all",
        spins: tuple[str, ...] | str = ("up", "down"),
        category: str | None = None,
        plus_hc: bool = False,
    ) -> None:
        """Expand an onsite four-index interaction tensor in orbital space.

        The tensor convention follows the symbolic operator layer directly:
        ``tensor[p, q, r, s]`` multiplies
        ``c_p^dagger c_q^dagger c_r c_s`` on each selected site, summed over the
        chosen spin pairs ``(sigma, tau)`` as
        ``c_{p,sigma}^dagger c_{q,tau}^dagger c_{r,tau} c_{s,sigma}``.
        """
        spins = _coerce_spins(self.space, spins)
        tensor = np.asarray(tensor, dtype=np.complex128)
        selected_orbitals = tuple(coerce_orbitals(self.space, orbitals))
        shape = (len(selected_orbitals),) * 4
        if tensor.shape != shape:
            raise ValueError(f"local interaction tensor must have shape {shape}, got {tensor.shape}.")

        for site in coerce_sites(self.space, sites):
            site_strength = onsite_value(strength, site)
            if abs(complex(site_strength)) <= 1e-15:
                continue
            for p_index, p_orbital in enumerate(selected_orbitals):
                for q_index, q_orbital in enumerate(selected_orbitals):
                    for r_index, r_orbital in enumerate(selected_orbitals):
                        for s_index, s_orbital in enumerate(selected_orbitals):
                            coeff = site_strength * tensor[p_index, q_index, r_index, s_index]
                            if abs(coeff) <= 1e-15:
                                continue
                            for left_spin in spins:
                                for right_spin in spins:
                                    term = (
                                        self.space.create(site, orbital=p_orbital, spin=left_spin)
                                        @ self.space.create(site, orbital=q_orbital, spin=right_spin)
                                        @ self.space.destroy(site, orbital=r_orbital, spin=right_spin)
                                        @ self.space.destroy(site, orbital=s_orbital, spin=left_spin)
                                    )
                                    if plus_hc:
                                        term = term + term.adjoint()
                                    self.store(
                                        coeff * term,
                                        kind="local_interaction_tensor",
                                        category=category,
                                        details={
                                            "site": site,
                                            "p_orbital": p_orbital,
                                            "q_orbital": q_orbital,
                                            "r_orbital": r_orbital,
                                            "s_orbital": s_orbital,
                                            "left_spin": left_spin,
                                            "right_spin": right_spin,
                                        },
                                    )

    def build(self, *, simplify: bool = True) -> Operator:
        """Return the assembled symbolic operator."""
        operator = Operator.identity(self.space, self._constant)
        for term in self._operators:
            operator = operator + term
        return operator.simplify() if simplify else operator


def _coerce_spins(space, spins) -> tuple[str, ...]:
    """Normalize spin selectors used by local matrix/tensor builders."""
    if isinstance(spins, str):
        if spins == "both":
            normalized = tuple(space.spins)
        else:
            normalized = (str(spins),)
    else:
        normalized = tuple(str(spin) for spin in spins)
    if not normalized:
        raise ValueError("spins must select at least one spin label.")
    invalid = tuple(spin for spin in normalized if spin not in space.spins)
    if invalid:
        raise ValueError(f"Unknown spin labels in local builder: {invalid!r}.")
    return normalized
