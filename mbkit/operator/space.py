"""User-facing modeling context for lattice fermion Hamiltonians.

The design choice in `mbkit` is that most users should stay at the `ElectronicSpace`
level: build a space from a lattice helper, call model methods directly on that
space, and only interact with lower-level symbolic objects when they need custom
algebra or backend transforms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from numpy.typing import ArrayLike

from .lattice import Bond, Lattice
from ..utils.construction import ConstructionEngine
from ..utils.selection import coerce_bonds, coerce_orbitals, coerce_sites, orbital_pair_value

if TYPE_CHECKING:
    from .operator import Operator


ScalarLike: TypeAlias = int | float | complex
CoefficientMap: TypeAlias = Mapping[object, ScalarLike]
CoefficientLike: TypeAlias = ScalarLike | CoefficientMap
OrbitalLabel: TypeAlias = str | int
SpinLabel: TypeAlias = Literal["up", "down"]
SpinSelection: TypeAlias = SpinLabel | Literal["both"] | None
LocalSpinSelection: TypeAlias = SpinLabel | Literal["both"] | Sequence[SpinLabel]
SiteSelection: TypeAlias = Literal["all"] | int | slice | range | Sequence[int]
OrbitalSelection: TypeAlias = Literal["all"] | OrbitalLabel | slice | Sequence[OrbitalLabel]
ShellSelection: TypeAlias = Literal["all"] | int | Sequence[int] | None
BondSelection: TypeAlias = Literal["all"] | str | slice | Bond | Sequence[str | Bond]
OrbitalPairSelection: TypeAlias = Sequence[tuple[OrbitalLabel, OrbitalLabel]] | None


def _negate_coefficients(value):
    """Negate either a scalar coefficient or a mapping of coefficients.

    Several model builders accept bond-/site-resolved dictionaries. Keeping this
    helper local makes the sign convention in Hubbard-style constructors explicit
    without complicating the public API.
    """
    if isinstance(value, dict):
        return {key: -item for key, item in value.items()}
    return -value


@dataclass(frozen=True)
class Mode:
    """Resolved spin-orbital label used by the symbolic operator layer."""

    index: int
    site: int
    orbital: str
    orbital_index: int
    spin: str
    spin_index: int


class ElectronicSpace:
    """User-facing modeling context for electronic lattice Hamiltonians.

    This class is the main public entry point for operator construction.
    It owns three pieces of information:

    - geometry: via an optional :class:`~mbkit.operator.lattice.Lattice`
    - labeling: site / orbital / spin ordering for the symbolic layer
    - optional default particle counts for solver convenience

    The intended workflow is:

    1. create a space with :func:`chain`, :func:`square`, :func:`general`,
       or an explicit :class:`~mbkit.operator.lattice.Lattice`
    2. construct Hamiltonians directly with methods such as
       :meth:`hubbard`, :meth:`extended_hubbard`, or :meth:`slater_kanamori`
    3. pass the resulting symbolic :class:`~mbkit.operator.operator.Operator`
       into a solver or transform
    """

    def __init__(
        self,
        lattice: Lattice | None = None,
        *,
        num_sites: int | None = None,
        orbitals: Sequence[str] | int = ("orb0",),
        spins: Sequence[str] = ("up", "down"),
        n_electrons: int | None = None,
        n_electrons_per_spin: tuple[int, int] | None = None,
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

        if n_electrons is not None and n_electrons_per_spin is not None:
            raise ValueError("Specify only one of n_electrons or n_electrons_per_spin.")
        if n_electrons_per_spin is not None:
            n_electrons_per_spin = (int(n_electrons_per_spin[0]), int(n_electrons_per_spin[1]))
            if min(n_electrons_per_spin) < 0:
                raise ValueError("n_electrons_per_spin cannot contain negative entries.")
            if sum(n_electrons_per_spin) > self.num_spin_orbitals:
                raise ValueError("n_electrons_per_spin exceeds the size of the ElectronicSpace.")
            self._n_electrons_per_spin = n_electrons_per_spin
            self._n_electrons = sum(n_electrons_per_spin)
        else:
            self._n_electrons_per_spin = None
            self._n_electrons = None if n_electrons is None else int(n_electrons)
            if self._n_electrons is not None:
                if self._n_electrons < 0:
                    raise ValueError("n_electrons cannot be negative.")
                if self._n_electrons > self.num_spin_orbitals:
                    raise ValueError("n_electrons exceeds the size of the ElectronicSpace.")

    def __repr__(self) -> str:
        lattice_name = type(self.lattice).__name__ if self.lattice is not None else "None"
        return (
            f"ElectronicSpace(num_sites={self.num_sites}, orbitals={self.orbitals}, "
            f"spins={self.spins}, lattice={lattice_name})"
        )

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

    @property
    def n_electrons(self) -> int | None:
        return self._n_electrons

    @property
    def n_electrons_per_spin(self) -> tuple[int, int] | None:
        return self._n_electrons_per_spin

    def electron_count(self):
        """Return the preferred electron-count specification for solvers."""
        if self._n_electrons_per_spin is not None:
            return self._n_electrons_per_spin
        return self._n_electrons

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
            return self._spin_to_index[str(spin)]
        except KeyError as exc:
            raise ValueError(f"Unknown spin label {spin!r}.") from exc

    def mode_index(self, site: int, *, orbital: str | int = 0, spin: str = "up") -> int:
        """Return the canonical spin-orbital index used throughout `mbkit`.

        `mbkit` uses interleaved `(site, orbital, spin)` ordering. All backend
        reorderings are handled in transforms, so this method defines the single
        source of truth for the public operator layer.
        """
        if not (0 <= int(site) < self.num_sites):
            raise ValueError(f"Site index {site} outside 0..{self.num_sites - 1}.")
        orbital_index = self.orbital_index(orbital)
        spin_index = self.spin_index(spin)
        return ((int(site) * self.num_orbitals_per_site) + orbital_index) * self.num_spins + spin_index

    def mode(self, site: int, *, orbital: str | int = 0, spin: str = "up") -> Mode:
        """Return a resolved :class:`Mode` record for a site/orbital/spin label."""
        orbital_index = self.orbital_index(orbital)
        spin_index = self.spin_index(spin)
        return Mode(
            index=self.mode_index(site, orbital=orbital_index, spin=spin),
            site=int(site),
            orbital=self._orbitals[orbital_index],
            orbital_index=orbital_index,
            spin=self._spins[spin_index],
            spin_index=spin_index,
        )

    def unpack_mode(self, mode_index: int) -> Mode:
        """Invert :meth:`mode_index` and recover structured labels from a mode id."""
        if not (0 <= int(mode_index) < self.num_spin_orbitals):
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

    def select_sites(self, sites: SiteSelection = "all") -> tuple[int, ...]:
        """Normalize site selectors accepted by the public builder methods."""
        return coerce_sites(self, sites)

    def select_orbitals(self, orbitals: OrbitalSelection = "all") -> tuple[str, ...]:
        """Normalize orbital selectors accepted by the public builder methods."""
        return coerce_orbitals(self, orbitals)

    def available_shells(self) -> tuple[int, ...]:
        """Return the neighbor shells available on the attached lattice."""
        if self.lattice is None:
            return tuple()
        return self.lattice.available_shells()

    def bond_summary(self, *, shells: ShellSelection = None) -> dict[int | None, dict[str, int]]:
        """Return a shell-first summary of the available bond families."""
        if self.lattice is None:
            return {}
        return self.lattice.bond_summary(shells=shells)

    def select_bonds(
        self,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
    ) -> tuple[Bond, ...]:
        """Normalize bond selectors accepted by the public builder methods.

        `bonds` filters geometric families such as `horizontal` or `rung`,
        while `shells` filters neighbor order such as first, second, or third.
        """
        return coerce_bonds(self, bonds, shells=shells)

    def _orbital_pairs(
        self,
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
    ) -> tuple[tuple[str, str], ...]:
        if orbital_pairs is not None:
            return tuple(
                (
                    self.orbitals[self.orbital_index(left)],
                    self.orbitals[self.orbital_index(right)],
                )
                for left, right in orbital_pairs
            )
        selected = tuple(self.select_orbitals(orbitals))
        return tuple(
            (selected[left_index], selected[right_index])
            for left_index in range(len(selected))
            for right_index in range(left_index + 1, len(selected))
        )

    def create(self, site: int, *, orbital: OrbitalLabel = 0, spin: SpinLabel = "up") -> Operator:
        """Return a single creation operator for the requested mode."""
        from .operator import Ladder, Operator, Term

        mode = self.mode(site, orbital=orbital, spin=spin)
        return Operator(self, terms={Term((Ladder(mode.index, "create"),)): 1.0})

    def destroy(self, site: int, *, orbital: OrbitalLabel = 0, spin: SpinLabel = "up") -> Operator:
        """Return a single annihilation operator for the requested mode."""
        from .operator import Ladder, Operator, Term

        mode = self.mode(site, orbital=orbital, spin=spin)
        return Operator(self, terms={Term((Ladder(mode.index, "destroy"),)): 1.0})

    def number(
        self,
        site: int,
        *,
        orbital: OrbitalLabel = 0,
        spin: SpinSelection = None,
    ) -> Operator:
        """Return a number operator for one spin or the spin-summed density."""
        if spin in {None, "both"}:
            return self.number(site, orbital=orbital, spin="up") + self.number(site, orbital=orbital, spin="down")
        return self.create(site, orbital=orbital, spin=spin) @ self.destroy(site, orbital=orbital, spin=spin)

    def spin_z(self, site: int, *, orbital: OrbitalLabel = 0) -> Operator:
        """Return the onsite spin-z operator for one orbital."""
        return 0.5 * self.number(site, orbital=orbital, spin="up") - 0.5 * self.number(site, orbital=orbital, spin="down")

    def spin_plus(self, site: int, *, orbital: OrbitalLabel = 0) -> Operator:
        """Return the onsite spin-raising operator for one orbital."""
        return self.create(site, orbital=orbital, spin="up") @ self.destroy(site, orbital=orbital, spin="down")

    def spin_minus(self, site: int, *, orbital: OrbitalLabel = 0) -> Operator:
        """Return the onsite spin-lowering operator for one orbital."""
        return self.create(site, orbital=orbital, spin="down") @ self.destroy(site, orbital=orbital, spin="up")

    def hopping(
        self,
        left_site: int,
        right_site: int,
        *,
        left_orbital: OrbitalLabel = 0,
        right_orbital: OrbitalLabel = 0,
        spin: SpinLabel | Literal["both"] = "both",
        coeff: complex = 1.0,
        plus_hc: bool = False,
    ) -> Operator:
        """Return a one-body hopping term between two lattice sites.

        This is the low-level primitive used by the higher-level `hopping_term`
        and Hubbard-family builders.
        """
        from .operator import Operator

        if spin == "both":
            operator = Operator.zero(self)
            for spin_label in self.spins:
                operator = operator + self.hopping(
                    left_site,
                    right_site,
                    left_orbital=left_orbital,
                    right_orbital=right_orbital,
                    spin=spin_label,
                    coeff=coeff,
                    plus_hc=plus_hc,
                )
            return operator

        operator = coeff * (
            self.create(left_site, orbital=left_orbital, spin=spin)
            @ self.destroy(right_site, orbital=right_orbital, spin=spin)
        )
        if plus_hc:
            operator = operator + operator.adjoint()
        return operator

    # Observable builders.
    def number_term(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        spin: SpinLabel | Literal["both"] = "both",
    ) -> Operator:
        """Build the total density operator over the selected sites/orbitals."""
        engine = ConstructionEngine(self)
        engine.add_onsite(1.0, "number", sites=sites, orbitals=orbitals, spin=spin, category="observable")
        return engine.build()

    def double_occupancy_term(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build the onsite doublon-counting operator over the selection."""
        engine = ConstructionEngine(self)
        engine.add_onsite(1.0, "double_occupancy", sites=sites, orbitals=orbitals, category="observable")
        return engine.build()

    def spin_z_term(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build the total spin-z operator over the selected sites/orbitals."""
        engine = ConstructionEngine(self)
        engine.add_onsite(1.0, "spin_z", sites=sites, orbitals=orbitals, category="observable")
        return engine.build()

    def spin_plus_term(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build the total spin-raising operator over the selected sites/orbitals."""
        engine = ConstructionEngine(self)
        engine.add_onsite(1.0, "spin_plus", sites=sites, orbitals=orbitals, category="observable")
        return engine.build()

    def spin_minus_term(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build the total spin-lowering operator over the selected sites/orbitals."""
        engine = ConstructionEngine(self)
        engine.add_onsite(1.0, "spin_minus", sites=sites, orbitals=orbitals, category="observable")
        return engine.build()

    def spin_squared_term(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build the total-spin operator ``S^2`` over the selected sites/orbitals."""
        s_minus = self.spin_minus_term(sites=sites, orbitals=orbitals)
        s_plus = self.spin_plus_term(sites=sites, orbitals=orbitals)
        s_z = self.spin_z_term(sites=sites, orbitals=orbitals)
        return s_minus @ s_plus + s_z @ s_z + s_z

    # Generic lattice-term builders.
    def hopping_term(
        self,
        *,
        coeff: CoefficientLike,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
        spin: SpinLabel | Literal["both"] = "both",
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
        plus_hc: bool = True,
    ) -> Operator:
        """Build a sum of hopping terms over selected bonds.

        `coeff` may be a scalar or a mapping keyed by bond metadata. The most
        useful mapping keys are shell numbers like `1` or `2`, bond-family names
        like `horizontal`, explicit tuples like `(shell, kind)`, or exact bond
        endpoints `(i, j)`.

        `shells` is the primary physics selector for first-, second-, or third-
        neighbor terms; `bonds` optionally refines that to a geometric family.
        If `orbital_pairs` is omitted, the builder uses diagonal pairs such as
        `("dxz", "dxz")` for each selected orbital.
        """
        engine = ConstructionEngine(self)
        engine.add_coupling(
            coeff,
            "hopping",
            bonds=bonds,
            shells=shells,
            orbitals=orbitals,
            orbital_pairs=orbital_pairs,
            spin=spin,
            category="hopping",
            plus_hc=plus_hc,
        )
        return engine.build()

    def chemical_potential_term(
        self,
        *,
        mu: CoefficientLike,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        spin: SpinLabel | Literal["both"] = "both",
    ) -> Operator:
        """Build a chemical-potential term ``mu * n`` over the selected sites/orbitals.

        `mu` may be a scalar or a mapping keyed by site index, orbital label, or
        `(site, orbital)` for site- and orbital-resolved shifts.
        """
        engine = ConstructionEngine(self)
        engine.add_onsite(mu, "number", sites=sites, orbitals=orbitals, spin=spin, category="chemical_potential")
        return engine.build()

    def density_density_term(
        self,
        *,
        coeff: CoefficientLike,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
        spin: SpinLabel | Literal["both"] = "both",
    ) -> Operator:
        """Build intersite density-density couplings over selected bonds.

        The selector semantics match :meth:`hopping_term`. When `spin="both"`,
        each site density is the spin-summed number operator; otherwise the term
        is resolved for the requested spin channel only.
        """
        engine = ConstructionEngine(self)
        engine.add_coupling(
            coeff,
            "density_density",
            bonds=bonds,
            shells=shells,
            orbitals=orbitals,
            orbital_pairs=orbital_pairs,
            spin=spin,
            category="density_density",
        )
        return engine.build()

    def exchange_term(
        self,
        *,
        coeff: CoefficientLike,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
    ) -> Operator:
        """Build onsite Hund-like spin-exchange terms between orbital pairs.

        If `orbital_pairs` is omitted, all distinct pairs inside the selected
        orbital set are used on each selected site.
        """
        engine = ConstructionEngine(self)
        engine.add_multi_coupling(
            coeff,
            "exchange",
            sites=sites,
            orbital_pairs=self._orbital_pairs(orbitals=orbitals, orbital_pairs=orbital_pairs),
            category="exchange",
        )
        return engine.build()

    def pair_hopping_term(
        self,
        *,
        coeff: CoefficientLike,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
    ) -> Operator:
        """Build onsite pair-hopping terms between orbital pairs."""
        engine = ConstructionEngine(self)
        engine.add_multi_coupling(
            coeff,
            "pair_hopping",
            sites=sites,
            orbital_pairs=self._orbital_pairs(orbitals=orbitals, orbital_pairs=orbital_pairs),
            category="pair_hopping",
        )
        return engine.build()

    def soc_term(
        self,
        *,
        strength: CoefficientLike = 1.0,
        matrix: ArrayLike | None = None,
        orbital_type: str | None = None,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build a local spin-orbit coupling term.

        Users can either pass an explicit local matrix or request a built-in
        cubic-basis matrix via `orbital_type`. The local matrix is interpreted
        in the same `(orbital, spin)` order documented by
        :meth:`local_matrix_term`.
        """
        from .soc import get_soc_matrix_cubic_basis

        selected_orbitals = tuple(self.select_orbitals(orbitals))
        if matrix is None:
            if orbital_type is None:
                raise ValueError("soc_term() requires either an explicit matrix or an orbital_type.")
            matrix = get_soc_matrix_cubic_basis(orbital_type)

        engine = ConstructionEngine(self)
        engine.add_local_matrix(
            matrix,
            strength=strength,
            sites=sites,
            orbitals=selected_orbitals,
            category="soc",
        )
        return engine.build()

    def local_matrix_term(
        self,
        *,
        matrix: ArrayLike,
        strength: CoefficientLike = 1.0,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        spins: LocalSpinSelection = ("up", "down"),
        plus_hc: bool = False,
    ) -> Operator:
        """Build a generic onsite one-body term from a local matrix.

        The matrix basis follows the selected local spin-orbital order
        ``[(orbital_0, spin_0), (orbital_0, spin_1), ..., (orbital_n, spin_m)]``.
        In other words, orbitals vary slowest and spins vary fastest, matching
        the public ``(site, orbital, spin)`` mode ordering inside one site.

        `strength` may be a scalar or a site-resolved mapping. Set `plus_hc=True`
        when you want the supplied matrix to represent only the forward part of a
        non-Hermitian local operator.
        """
        engine = ConstructionEngine(self)
        engine.add_local_matrix(
            matrix,
            strength=strength,
            sites=sites,
            orbitals=orbitals,
            spins=spins,
            category="local_matrix",
            plus_hc=plus_hc,
        )
        return engine.build()

    def local_interaction_tensor_term(
        self,
        *,
        tensor: ArrayLike,
        strength: CoefficientLike = 1.0,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        spins: LocalSpinSelection = ("up", "down"),
        plus_hc: bool = False,
    ) -> Operator:
        """Build a generic onsite two-body term from an orbital-space tensor.

        The tensor convention is ``tensor[p, q, r, s] * c_p^dagger c_q^dagger c_r c_s``
        on each selected site. For every chosen spin pair ``(sigma, tau)``, the
        method expands the corresponding term as
        ``c_{p,sigma}^dagger c_{q,tau}^dagger c_{r,tau} c_{s,sigma}``.

        Tensor indices follow the explicit `orbitals=` selection order. As with
        every symbolic `Operator` in `mbkit`, the stored internal terms are then
        canonically normal-ordered, so backend tensor round-trips reflect that
        canonical fermionic ordering rather than the raw input word order.

        `strength` may be a scalar or a site-resolved mapping. This helper is
        the generic onsite quartic builder to reach for when a model does not
        fit into the specialized Slater-Kanamori or SOC convenience APIs.
        """
        engine = ConstructionEngine(self)
        engine.add_local_interaction_tensor(
            tensor,
            strength=strength,
            sites=sites,
            orbitals=orbitals,
            spins=spins,
            category="local_interaction",
            plus_hc=plus_hc,
        )
        return engine.build()

    def local_two_body_tensor_term(
        self,
        *,
        tensor: ArrayLike,
        strength: CoefficientLike = 1.0,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        spins: LocalSpinSelection = ("up", "down"),
        plus_hc: bool = False,
    ) -> Operator:
        """Alias for :meth:`local_interaction_tensor_term`.

        This name is kept because some users naturally search for a "two-body
        tensor" constructor before they discover the more physics-oriented
        "interaction tensor" wording.
        """
        return self.local_interaction_tensor_term(
            tensor=tensor,
            strength=strength,
            sites=sites,
            orbitals=orbitals,
            spins=spins,
            plus_hc=plus_hc,
        )

    def crystal_field_term(
        self,
        *,
        values: CoefficientLike,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        spin: SpinLabel | Literal["both"] = "both",
    ) -> Operator:
        """Build onsite orbital-energy shifts over the selected sites/orbitals.

        `values` may be a scalar or a mapping keyed by site, orbital, or
        `(site, orbital)` to express crystal-field splittings.
        """
        engine = ConstructionEngine(self)
        engine.add_onsite(values, "number", sites=sites, orbitals=orbitals, spin=spin, category="crystal_field")
        return engine.build()

    def zeeman_term(
        self,
        *,
        hz: CoefficientLike,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Build a local Zeeman field ``h_z S_z`` over the selection."""
        engine = ConstructionEngine(self)
        engine.add_onsite(hz, "spin_z", sites=sites, orbitals=orbitals, category="zeeman")
        return engine.build()

    def pairing_term(
        self,
        *,
        coeff: CoefficientLike,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
        plus_hc: bool = True,
    ) -> Operator:
        """Build singlet-like bond pairing terms over selected bonds.

        The generated forward term is
        ``c_{i,up}^dagger c_{j,down}^dagger`` for the chosen orbital pair, with
        `plus_hc=True` adding the Hermitian conjugate by default.
        """
        engine = ConstructionEngine(self)
        engine.add_coupling(
            coeff,
            "pairing",
            bonds=bonds,
            shells=shells,
            orbitals=orbitals,
            orbital_pairs=orbital_pairs,
            spin="both",
            category="pairing",
            plus_hc=plus_hc,
        )
        return engine.build()

    def current_term(
        self,
        *,
        coeff: CoefficientLike,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
        orbitals: OrbitalSelection = "all",
        orbital_pairs: OrbitalPairSelection = None,
        spin: SpinLabel | Literal["both"] = "both",
    ) -> Operator:
        """Build bond-current operators over selected bonds and orbital pairs."""
        engine = ConstructionEngine(self)
        engine.add_coupling(
            coeff,
            "current",
            bonds=bonds,
            shells=shells,
            orbitals=orbitals,
            orbital_pairs=orbital_pairs,
            spin=spin,
            category="current",
        )
        return engine.build()

    # High-level model constructors.
    def hubbard(
        self,
        *,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
        t: CoefficientLike = 1.0,
        U: CoefficientLike = 0.0,
        mu: CoefficientLike | None = None,
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Construct a spinful Hubbard Hamiltonian on this space.

        The sign convention follows the common condensed-matter form
        `-t (c_i^dagger c_j + h.c.) + U n_{i,up} n_{i,down}`.

        `shells` is the primary way to target first-, second-, or third-neighbor
        terms; `bonds` optionally refines that selection to a geometric family.
        `t` can be shell-resolved or bond-kind-resolved via a mapping, and `mu`
        can be site- or orbital-resolved using the same selectors accepted by
        :meth:`chemical_potential_term`.
        """
        engine = ConstructionEngine(self)
        engine.add_coupling(
            _negate_coefficients(t),
            "hopping",
            bonds=bonds,
            shells=shells,
            orbitals=orbitals,
            spin="both",
            category="kinetic",
            plus_hc=True,
        )
        engine.add_onsite(U, "double_occupancy", sites="all", orbitals=orbitals, category="interaction")
        if mu is not None:
            engine.add_onsite(mu, "number", sites="all", orbitals=orbitals, spin="both", category="chemical_potential")
        return engine.build()

    def extended_hubbard(
        self,
        *,
        hopping: CoefficientLike,
        onsite_U: CoefficientLike,
        intersite_V: CoefficientLike | None = None,
        bonds: BondSelection = "all",
        shells: ShellSelection = None,
        orbitals: OrbitalSelection = "all",
    ) -> Operator:
        """Construct an extended Hubbard Hamiltonian with intersite density terms.

        `shells` filters neighbor order, while `bonds` optionally refines the
        chosen shell(s) to a specific geometric family. `hopping` and
        `intersite_V` both accept the same bond metadata mappings documented by
        :meth:`hopping_term`.
        """
        engine = ConstructionEngine(self)
        engine.add_coupling(
            _negate_coefficients(hopping),
            "hopping",
            bonds=bonds,
            shells=shells,
            orbitals=orbitals,
            spin="both",
            category="kinetic",
            plus_hc=True,
        )
        engine.add_onsite(onsite_U, "double_occupancy", sites="all", orbitals=orbitals, category="onsite_U")
        if intersite_V is not None:
            engine.add_coupling(
                intersite_V,
                "density_density",
                bonds=bonds,
                shells=shells,
                orbitals=orbitals,
                spin="both",
                category="intersite_V",
            )
        return engine.build()

    def slater_kanamori(
        self,
        *,
        sites: SiteSelection = "all",
        orbitals: OrbitalSelection = "all",
        U: CoefficientLike = 3.0,
        Up: CoefficientLike = 0.0,
        J: CoefficientLike = 0.0,
        Jp: CoefficientLike = 0.0,
    ) -> Operator:
        """Construct the onsite Slater-Kanamori interaction for multi-orbital sites.

        The symbolic `Operator` layer stays generic; this method is where the
        domain-specific orbital structure is assembled into quartic terms.
        `U` is the intra-orbital doublon penalty, `Up` is the inter-orbital
        density interaction, `J` is the spin-exchange term, and `Jp` is the
        pair-hopping amplitude.
        """
        selected_sites = tuple(self.select_sites(sites))
        orbital_pairs = self._orbital_pairs(orbitals=orbitals)

        engine = ConstructionEngine(self)
        engine.add_onsite(U, "double_occupancy", sites=selected_sites, orbitals=orbitals, category="U")

        for site in selected_sites:
            for left_orbital, right_orbital in orbital_pairs:
                value = orbital_pair_value(Up, site, left_orbital, right_orbital)
                if abs(complex(value)) <= 1e-15:
                    continue
                term = self.number(site, orbital=left_orbital, spin="both") @ self.number(
                    site,
                    orbital=right_orbital,
                    spin="both",
                )
                engine.store(
                    value * term,
                    kind="multi_coupling",
                    category="Up",
                    details={
                        "site": site,
                        "left_orbital": left_orbital,
                        "right_orbital": right_orbital,
                    },
                )

        engine.add_multi_coupling(
            _negate_coefficients(J),
            "exchange",
            sites=selected_sites,
            orbital_pairs=orbital_pairs,
            category="J",
        )
        engine.add_multi_coupling(
            _negate_coefficients(Jp),
            "pair_hopping",
            sites=selected_sites,
            orbital_pairs=orbital_pairs,
            category="Jp",
        )
        return engine.build()


def chain(
    length: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: str = "open",
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a chain `ElectronicSpace` ready for model construction.

    Use `max_shell` to generate beyond-nearest-neighbor bonds on the lattice.
    """
    from .lattice import LineLattice

    return ElectronicSpace(
        LineLattice(length, boundary=boundary, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def ladder(
    length: int,
    *,
    legs: int = 2,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: str = "open",
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a ladder `ElectronicSpace` with `leg` and `rung` bonds.

    Use `max_shell` to include longer-range leg, rung, and diagonal-like bonds.
    """
    from .lattice import LadderLattice

    return ElectronicSpace(
        LadderLattice(length, legs=legs, boundary=boundary, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def square(
    nx: int,
    ny: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: tuple[str, str] = ("open", "open"),
    include_diagonals: bool = False,
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a square-lattice `ElectronicSpace` ready for model construction.

    `max_shell` generates longer-range bonds. `include_diagonals=True` is kept
    as a compatibility shortcut that forces at least second-neighbor bonds.
    """
    from .lattice import SquareLattice

    return ElectronicSpace(
        SquareLattice(nx, ny, boundary=boundary, include_diagonals=include_diagonals, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def rectangular(
    nx: int,
    ny: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: tuple[str, str] = ("open", "open"),
    include_diagonals: bool = False,
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a rectangular-lattice `ElectronicSpace` ready for model construction."""
    from .lattice import RectangularLattice

    return ElectronicSpace(
        RectangularLattice(nx, ny, boundary=boundary, include_diagonals=include_diagonals, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def triangular(
    nx: int,
    ny: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: tuple[str, str] = ("open", "open"),
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a triangular-lattice `ElectronicSpace` ready for model construction."""
    from .lattice import TriangularLattice

    return ElectronicSpace(
        TriangularLattice(nx, ny, boundary=boundary, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def honeycomb(
    nx: int,
    ny: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: tuple[str, str] = ("open", "open"),
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a honeycomb-lattice `ElectronicSpace` ready for model construction."""
    from .lattice import HoneycombLattice

    return ElectronicSpace(
        HoneycombLattice(nx, ny, boundary=boundary, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def kagome(
    nx: int,
    ny: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: tuple[str, str] = ("open", "open"),
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a kagome-lattice `ElectronicSpace` ready for model construction."""
    from .lattice import KagomeLattice

    return ElectronicSpace(
        KagomeLattice(nx, ny, boundary=boundary, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def cubic(
    nx: int,
    ny: int,
    nz: int,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    boundary: tuple[str, str, str] = ("open", "open", "open"),
    max_shell: int = 1,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return a simple-cubic `ElectronicSpace` ready for model construction."""
    from .lattice import CubicLattice

    return ElectronicSpace(
        CubicLattice(nx, ny, nz, boundary=boundary, max_shell=max_shell),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )


def general(
    num_sites: int,
    bonds,
    *,
    orbitals: Sequence[str] | int = ("orb0",),
    site_positions: Sequence[Sequence[float]] | None = None,
    n_electrons: int | None = None,
    n_electrons_per_spin: tuple[int, int] | None = None,
) -> ElectronicSpace:
    """Return an `ElectronicSpace` from explicit bond data.

    This helper keeps the common path short while still allowing custom graph
    geometries without instantiating a lattice manually first.
    """
    from .lattice import GeneralLattice

    return ElectronicSpace(
        GeneralLattice(num_sites=num_sites, bonds=bonds, site_positions=site_positions),
        orbitals=orbitals,
        n_electrons=n_electrons,
        n_electrons_per_spin=n_electrons_per_spin,
    )
