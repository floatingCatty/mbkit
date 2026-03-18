from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .lattice import Lattice


@dataclass(frozen=True)
class Mode:
    index: int
    site: int
    orbital: str
    orbital_index: int
    spin: str
    spin_index: int


class ElectronicSpace:
    def __init__(
        self,
        lattice: Lattice | None = None,
        *,
        num_sites: int | None = None,
        orbitals: Sequence[str] | int = ("orb0",),
        spins: Sequence[str] = ("up", "down"),
    ) -> None:
        if lattice is None and num_sites is None:
            raise ValueError("ElectronicSpace requires either a lattice or num_sites.")
        if lattice is not None and num_sites is not None and lattice.num_sites != int(num_sites):
            raise ValueError("num_sites does not match the supplied lattice.")
        self.lattice = lattice
        self._num_sites = lattice.num_sites if lattice is not None else int(num_sites)

        if isinstance(orbitals, int):
            if orbitals < 1:
                raise ValueError("orbitals must be positive.")
            self._orbitals = tuple(f"orb{i}" for i in range(orbitals))
        else:
            self._orbitals = tuple(str(orbital) for orbital in orbitals)
        if not self._orbitals:
            raise ValueError("ElectronicSpace needs at least one orbital.")

        self._spins = tuple(str(spin) for spin in spins)
        if self._spins != ("up", "down"):
            raise ValueError("v1 ElectronicSpace only supports spins ('up', 'down').")

        self._orbital_to_index = {label: index for index, label in enumerate(self._orbitals)}
        self._spin_to_index = {label: index for index, label in enumerate(self._spins)}

    @property
    def num_sites(self) -> int:
        return self._num_sites

    @property
    def orbitals(self) -> tuple[str, ...]:
        return self._orbitals

    @property
    def spins(self) -> tuple[str, ...]:
        return self._spins

    @property
    def num_orbitals_per_site(self) -> int:
        return len(self._orbitals)

    @property
    def num_spins(self) -> int:
        return len(self._spins)

    @property
    def num_spatial_orbitals(self) -> int:
        return self.num_sites * self.num_orbitals_per_site

    @property
    def num_spin_orbitals(self) -> int:
        return self.num_spatial_orbitals * self.num_spins

    def orbital_index(self, orbital: str | int) -> int:
        if isinstance(orbital, int):
            if not (0 <= orbital < self.num_orbitals_per_site):
                raise ValueError(f"Orbital index {orbital} outside 0..{self.num_orbitals_per_site - 1}.")
            return orbital
        try:
            return self._orbital_to_index[str(orbital)]
        except KeyError as exc:
            raise ValueError(f"Unknown orbital label {orbital!r}.") from exc

    def spin_index(self, spin: str) -> int:
        try:
            return self._spin_to_index[spin]
        except KeyError as exc:
            raise ValueError(f"Unknown spin label {spin!r}.") from exc

    def mode_index(self, site: int, *, orbital: str | int = 0, spin: str = "up") -> int:
        if not (0 <= site < self.num_sites):
            raise ValueError(f"Site index {site} outside 0..{self.num_sites - 1}.")
        orbital_index = self.orbital_index(orbital)
        spin_index = self.spin_index(spin)
        return ((site * self.num_orbitals_per_site) + orbital_index) * self.num_spins + spin_index

    def mode(self, site: int, *, orbital: str | int = 0, spin: str = "up") -> Mode:
        orbital_index = self.orbital_index(orbital)
        spin_index = self.spin_index(spin)
        index = self.mode_index(site, orbital=orbital_index, spin=spin)
        return Mode(
            index=index,
            site=int(site),
            orbital=self._orbitals[orbital_index],
            orbital_index=orbital_index,
            spin=spin,
            spin_index=spin_index,
        )

    def unpack_mode(self, mode_index: int) -> Mode:
        if not (0 <= mode_index < self.num_spin_orbitals):
            raise ValueError(f"Mode index {mode_index} outside 0..{self.num_spin_orbitals - 1}.")
        spatial, spin_index = divmod(int(mode_index), self.num_spins)
        site, orbital_index = divmod(spatial, self.num_orbitals_per_site)
        return Mode(
            index=int(mode_index),
            site=site,
            orbital=self._orbitals[orbital_index],
            orbital_index=orbital_index,
            spin=self._spins[spin_index],
            spin_index=spin_index,
        )

    def create(self, site: int, *, orbital: str | int = 0, spin: str = "up"):
        from .operator import Ladder, Operator, Term

        mode = self.mode(site, orbital=orbital, spin=spin)
        return Operator(self, terms={Term((Ladder(mode.index, "create"),)): 1.0})

    def destroy(self, site: int, *, orbital: str | int = 0, spin: str = "up"):
        from .operator import Ladder, Operator, Term

        mode = self.mode(site, orbital=orbital, spin=spin)
        return Operator(self, terms={Term((Ladder(mode.index, "destroy"),)): 1.0})

    def number(self, site: int, *, orbital: str | int = 0, spin: str | None = None):
        if spin in {None, "both"}:
            return self.number(site, orbital=orbital, spin="up") + self.number(site, orbital=orbital, spin="down")
        return self.create(site, orbital=orbital, spin=spin) @ self.destroy(site, orbital=orbital, spin=spin)

    def spin_z(self, site: int, *, orbital: str | int = 0):
        return 0.5 * self.number(site, orbital=orbital, spin="up") - 0.5 * self.number(site, orbital=orbital, spin="down")

    def spin_plus(self, site: int, *, orbital: str | int = 0):
        return self.create(site, orbital=orbital, spin="up") @ self.destroy(site, orbital=orbital, spin="down")

    def spin_minus(self, site: int, *, orbital: str | int = 0):
        return self.create(site, orbital=orbital, spin="down") @ self.destroy(site, orbital=orbital, spin="up")

    def hopping(
        self,
        left_site: int,
        right_site: int,
        *,
        left_orbital: str | int = 0,
        right_orbital: str | int = 0,
        spin: str = "both",
        coeff: complex = 1.0,
        plus_hc: bool = False,
    ):
        from .operator import Operator

        if spin == "both":
            operator = self.hopping(
                left_site,
                right_site,
                left_orbital=left_orbital,
                right_orbital=right_orbital,
                spin="up",
                coeff=coeff,
                plus_hc=plus_hc,
            )
            operator += self.hopping(
                left_site,
                right_site,
                left_orbital=left_orbital,
                right_orbital=right_orbital,
                spin="down",
                coeff=coeff,
                plus_hc=plus_hc,
            )
            return operator

        operator = coeff * (
            self.create(left_site, orbital=left_orbital, spin=spin)
            @ self.destroy(right_site, orbital=right_orbital, spin=spin)
        )
        if plus_hc:
            operator += operator.adjoint()
        return operator
