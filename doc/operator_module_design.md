# mbkit Operator Module Design

## Status

As of March 18, 2026, the direct symbolic/integrals frontend described in this document is the
active `mbkit` operator stack. Direct QuSpin and QC-tensor lowerings are in place, the
`mbkit.operator.legacy` namespace has been retired, and the supported solver targets are the direct
ED, DMRG, and PySCF adapters.

## Summary

This document defines the target architecture and staged development plan for the `mbkit.operator`
submodule. The redesign is a clean break from the current QuSpin-shaped API and makes the operator
layer:

- backend-neutral
- lattice-first and user-friendly
- explicit about site/orbital/spin structure
- able to compile to solver-facing formats such as QuSpin, structured electronic integrals,
  Block2-style tensors, Quantax, and later OpenFermion

The current `mbkit.operator.Operator` is the public frontend abstraction; QuSpin and QC tensors are
now compilation targets rather than user-facing construction formats.

The first milestone prioritizes electronic lattice/materials workflows while preserving a clean
export path to integral-based solver backends.

## Goals

- Provide a direct, intuitive API for constructing electronic many-body Hamiltonians.
- Make lattice, graph, site, orbital, and spin structure first-class.
- Separate symbolic operator construction from backend export and solver execution.
- Support a clean path from user-facing Hamiltonian construction to:
  - legacy QuSpin-backed workflows
  - structured one-/two-body integral containers
  - ED/DMRG/PySCF/NQS compiler targets
- Make common condensed-matter model construction concise:
  - Hubbard
  - Slater-Kanamori
  - extended Hubbard
  - local potentials
  - density-density interactions
  - exchange and pair hopping
  - SOC-like local couplings
- Make the redesign implementation-ready enough that an engineer can build it without inventing
  missing interfaces.

## Non-Goals

- Bosonic or mixed electron-boson operators in v1.
- Time-dependent operators in v1.
- Symbolic parameter expressions in v1. Coefficients are numeric only.
- Exact chemistry-first frontend parity with Qiskit Nature or PySCF in v1.
- Reusing QuSpin op-strings as the public construction language.
- Keeping the current public helper API as the main documented API.

## Current mbkit Status And Limitations

The current operator layer is functional as a backend-oriented prototype, but it is not a good
public construction API.

### Current strengths

- Small, working fermionic operator algebra already exists.
- Existing helpers can build Hubbard-like and Slater-Kanamori-like Hamiltonians.
- The solver stack already consumes the current representation.
- The current module can export to solver-facing tensors in some important cases.

### Current limitations

- `mbkit/operator/operator.py` stores operators directly in QuSpin static-list format.
- `mbkit/operator/operator.py` mixes symbolic representation, basis creation, sparse matrix
  construction, diagonalization, and integral export in one class.
- `mbkit/operator/operator.py` has no first-class constant/identity term support.
- `mbkit/operator/operator.py` multiplies operators by concatenating op strings and indices; this is
  not a complete fermionic symbolic algebra layer.
- `mbkit/operator/site_operator.py` requires users to pass `nsites` and raw integer indices into
  every site-level constructor.
- `mbkit/operator/common_operators.py` expects low-level neighbor lists and raw hopping matrices
  rather than a reusable lattice object.
- `mbkit/operator/extended_hubbard/extended_hubbard.py` mixes generic operator construction with a
  highly specialized hard-coded parameter set and physical preset.
- Backend-specific export logic is embedded in the current public `Operator` abstraction.
- The current public API exposes backend details such as QuSpin op strings, current spin-orbital
  ordering, and solver-shaped tensor conversion rules.

### Consequence

The current module is usable by someone who already understands the internal solver conventions,
but it does not yet satisfy the project goal from `doc/devplan.md`: an easy, unified, intuitive
Hamiltonian-construction toolkit.

## Reference Review

The redesign should borrow specific ideas from the local mirrors in `extern/` rather than copying
any one package wholesale.

### OpenFermion

Reviewed sources:

- `extern/OpenFermion/src/openfermion/ops/operators/fermion_operator.py`
- `extern/OpenFermion/src/openfermion/ops/representations/polynomial_tensor.py`
- `extern/OpenFermion/src/openfermion/ops/representations/interaction_operator.py`
- `extern/OpenFermion/src/openfermion/ops/representations/quadratic_hamiltonian.py`

Borrow:

- separation between a general symbolic fermionic operator and specialized structured Hamiltonian
  containers
- explicit normal ordering and fermionic sign handling
- body-order-aware representations
- numeric tensor containers for solver-efficient workflows

Do not borrow:

- string syntax as the primary end-user API in `mbkit`

### Qiskit Nature

Reviewed sources:

- `extern/qiskit-nature/qiskit_nature/second_q/operators/fermionic_op.py`
- `extern/qiskit-nature/qiskit_nature/second_q/operators/electronic_integrals.py`
- `extern/qiskit-nature/qiskit_nature/second_q/hamiltonians/lattice_model.py`
- `extern/qiskit-nature/qiskit_nature/second_q/hamiltonians/fermi_hubbard_model.py`
- `extern/qiskit-nature/qiskit_nature/second_q/hamiltonians/lattices/lattice.py`

Borrow:

- strong split between sparse symbolic operators and structured integral containers
- lattice Hamiltonians as a separate layer above operator internals
- explicit validation of register length and tensor keys
- conversion-driven design: symbolic and tensor forms coexist

Do not borrow:

- the full framework weight or the string-label DSL as the primary public construction mode

### TeNPy

Reviewed sources:

- `extern/tenpy/tenpy/models/model.py`
- `extern/tenpy/tenpy/models/lattice.py`
- `extern/tenpy/tenpy/networks/site.py`

Borrow:

- layered decomposition into site, lattice, and model
- typed lattice/bond taxonomy
- reusable predefined lattices
- explicit separation between geometry and Hamiltonian terms

Do not borrow:

- matrix-local operator storage as the core symbolic representation

### NetKet

Reviewed sources:

- `extern/netket/netket/hilbert/spin_orbital_fermions.py`
- `extern/netket/netket/_src/operator/particle_number_conserving_fermionic/fermihubbard.py`
- `extern/netket/netket/operator/_graph_operator.py`

Borrow:

- hilbert/space abstractions that know particle and spin structure
- graph-based operator ergonomics
- explicit subspace and constraint awareness

Do not borrow:

- graph-local matrix storage as the primary universal representation

### PySCF

Reviewed sources:

- `extern/pyscf/pyscf/fci/direct_spin1.py`
- `extern/pyscf/pyscf/ao2mo/*`
- `extern/pyscf/pyscf/tools/fcidump.py`

Borrow:

- integral conventions and the reality that solver backends consume structured numeric tensors
- FCIDUMP/integral export as an important interoperability target

Do not borrow:

- PySCF's solver-facing data consumption style as the public symbolic construction API

### QuSpin

Reviewed sources:

- `extern/QuSpin/src/quspin/operators/hamiltonian_core.py`
- `extern/QuSpin/src/quspin/operators/quantum_operator_core.py`
- `extern/QuSpin/examples/notebooks/FHM.py`

Borrow:

- QuSpin remains a useful backend/compiler target
- its parameterized operator ideas are good future reference

Do not borrow:

- QuSpin static lists or op strings as the primary public `mbkit` operator abstraction

## Design Principles

- User-facing construction must speak in terms of sites, orbitals, spins, bonds, and model terms.
- Internal symbolic algebra must be backend-neutral.
- Compilers, not the frontend `Operator`, own backend-specific lowering.
- Structured integral containers must exist alongside the general symbolic operator.
- The public API should be explicit and typed, not "magic string plus backend convention".
- The v1 design should favor correctness and clarity over extreme generality.

## Target Architecture

### Top-level package layout

```text
mbkit/operator/
    __init__.py
    legacy/
        __init__.py
        operator.py
        site_operator.py
        common_operators.py
        extended_hubbard/
            __init__.py
            extended_hubbard.py
            lattice.py
    space.py
    lattice.py
    operator.py
    integrals.py
    models/
        __init__.py
        hopping.py
        hubbard.py
        slater_kanamori.py
        extended_hubbard.py
        observables.py
        local_terms.py
    transforms/
        __init__.py
        quspin.py
        integrals.py
        block2.py
        quantax.py
        openfermion.py
```

The current implementation files are moved under `mbkit.operator.legacy` once the redesign starts.

### Layer responsibilities

- `space.py`
  - site/orbital/spin labels
  - canonical mode indexing
  - typed local operator constructors
- `lattice.py`
  - graph/lattice geometry
  - bond groups and bond metadata
  - predefined lattices
- `operator.py`
  - backend-neutral symbolic fermionic operator algebra
  - normal ordering
  - simplification
  - metadata and validation
- `integrals.py`
  - structured one-/two-body electronic coefficient container
  - conversion from and to the symbolic operator when possible
- `models/`
  - ergonomic Hamiltonian builders
  - high-level lattice/materials model constructors
- `transforms/`
  - lower the new symbolic operator or integrals into backend-specific formats
- `legacy/`
  - temporary home of the current QuSpin-shaped implementation

## Public Interfaces

### 1. `ElectronicSpace`

`ElectronicSpace` is the public entrypoint for indexed fermionic modes.

Responsibilities:

- own the canonical spin-orbital ordering
- translate site/orbital/spin labels into mode indices
- expose low-level local operator constructors
- expose small convenience builders for common single-particle terms

#### Canonical ordering

The new public operator layer uses:

- site-major
- orbital-major within site
- spin-major within orbital

That is, mode index is:

```text
mode = ((site_index * n_orbitals_per_site) + orbital_index) * n_spins + spin_index
```

with `spin_index = 0` for `"up"` and `1` for `"down"` in v1.

This is intentionally **not** the current QuSpin backend ordering. The QuSpin compiler will
translate from the public canonical ordering into QuSpin's backend ordering.

#### API sketch

```python
class ElectronicSpace:
    def __init__(
        self,
        lattice: Lattice | None = None,
        *,
        num_sites: int | None = None,
        orbitals: Sequence[str] | int = ("orb0",),
        spins: Sequence[str] = ("up", "down"),
    ) -> None: ...

    @property
    def num_sites(self) -> int: ...

    @property
    def num_orbitals_per_site(self) -> int: ...

    @property
    def num_spin_orbitals(self) -> int: ...

    def mode(
        self,
        site: int,
        *,
        orbital: str | int = 0,
        spin: str = "up",
    ) -> Mode: ...

    def create(self, site: int, *, orbital: str | int = 0, spin: str = "up") -> Operator: ...
    def destroy(self, site: int, *, orbital: str | int = 0, spin: str = "up") -> Operator: ...
    def number(self, site: int, *, orbital: str | int = 0, spin: str | None = None) -> Operator: ...
    def spin_z(self, site: int, *, orbital: str | int = 0) -> Operator: ...
    def spin_plus(self, site: int, *, orbital: str | int = 0) -> Operator: ...
    def spin_minus(self, site: int, *, orbital: str | int = 0) -> Operator: ...
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
    ) -> Operator: ...
```

#### Selector conventions

The operator/model APIs use these shared conventions:

- `spin="up" | "down" | "both"`
- `sites="all" | int | Sequence[int]`
- `orbitals="all" | str | int | Sequence[str | int]`
- `bonds="all" | str | Sequence[str] | Sequence[Bond]`

These conventions should be implemented consistently across the new frontend.

### 2. `Lattice`

`Lattice` owns geometry and bond typing. It does not own the Hilbert space or operator algebra.

#### Required classes

```python
@dataclass(frozen=True)
class Bond:
    left: int
    right: int
    kind: str = "default"
    weight: complex = 1.0
    displacement: tuple[int, ...] | None = None


class Lattice:
    @property
    def num_sites(self) -> int: ...
    def bonds(self, kind: str | None = None) -> tuple[Bond, ...]: ...
    def adjacency(self, kind: str | None = None) -> np.ndarray: ...


class GeneralLattice(Lattice):
    def __init__(
        self,
        *,
        num_sites: int,
        bonds: Sequence[Bond],
        site_positions: Sequence[Sequence[float]] | None = None,
    ) -> None: ...


class LineLattice(Lattice):
    def __init__(self, length: int, *, boundary: str = "open") -> None: ...


class SquareLattice(Lattice):
    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        boundary: tuple[str, str] = ("open", "open"),
        include_diagonals: bool = False,
    ) -> None: ...
```

#### Bond taxonomy rules

- Every bond has a `kind`.
- Predefined lattices assign stable default kinds such as `nn`, `nnn`, `horizontal`, `vertical`,
  or `diagonal`.
- `GeneralLattice` trusts the user-supplied `kind`.
- Model builders select bonds by `kind` rather than raw edge lists when possible.

### 3. New backend-neutral `Operator`

The new `Operator` is the symbolic fermionic algebra object.

#### Representation

```python
@dataclass(frozen=True)
class Ladder:
    mode: int
    action: Literal["create", "destroy"]


@dataclass(frozen=True)
class Term:
    factors: tuple[Ladder, ...]


class Operator:
    def __init__(
        self,
        space: ElectronicSpace,
        *,
        constant: complex = 0.0,
        terms: Mapping[Term, complex] | None = None,
    ) -> None: ...
```

#### Representation rules

- `constant` is first-class and represents `constant * I`.
- Non-constant terms are stored in canonical normal order.
- Coefficients are numeric scalars only in v1.
- Terms with coefficient magnitude below a configured tolerance are dropped during simplify.
- Metadata lives on the operator, not in backend-specific strings.

#### Normal-ordering convention

The symbolic layer follows an OpenFermion-style canonical order:

- all creation operators before annihilation operators
- within the same action class, factors are sorted by descending mode index

Every algebraic operation producing new terms must canonicalize to this normal form.

#### Algebra API

```python
class Operator:
    @property
    def space(self) -> ElectronicSpace: ...

    @property
    def constant(self) -> complex: ...

    @property
    def terms(self) -> Mapping[Term, complex]: ...

    def iter_terms(self) -> Iterator[tuple[Term, complex]]: ...
    def simplify(self, *, atol: float = 1e-12) -> Operator: ...
    def adjoint(self) -> Operator: ...
    def is_hermitian(self, *, atol: float = 1e-12) -> bool: ...
    def body_rank(self) -> int: ...
    def particle_number_changes(self) -> frozenset[int]: ...
    def spin_changes(self) -> frozenset[int]: ...

    def __add__(self, other: Operator | complex) -> Operator: ...
    def __sub__(self, other: Operator | complex) -> Operator: ...
    def __mul__(self, scalar: complex) -> Operator: ...
    def __rmul__(self, scalar: complex) -> Operator: ...
    def __matmul__(self, other: Operator) -> Operator: ...
```

#### Operator product decision

- `*` is reserved for scalar multiplication.
- `@` is the noncommutative operator product.

This avoids ambiguity and makes the symbolic layer cleaner than the current implementation.

### 4. `ElectronicIntegrals`

`ElectronicIntegrals` is the structured container for number-conserving one-/two-body Hamiltonians.

#### Scope

It is not a general symbolic operator replacement. It exists to support export and solver pipelines.

#### Representation decision

The container stores **literal normal-ordered spin-orbital coefficients**:

```text
constant
+ sum_{p,q} h[p,q] a_p^† a_q
+ sum_{p,q,r,s} g[p,q,r,s] a_p^† a_q^† a_r a_s
```

There are **no implicit 1/2 factors** in stored tensors. Backend exporters are responsible for
target-specific prefactors, symmetrization, and tensor order conventions.

#### API sketch

```python
class ElectronicIntegrals:
    def __init__(
        self,
        space: ElectronicSpace,
        *,
        constant: complex = 0.0,
        one_body: np.ndarray | None = None,
        two_body: np.ndarray | None = None,
    ) -> None: ...

    @classmethod
    def zeros(cls, space: ElectronicSpace) -> ElectronicIntegrals: ...

    @classmethod
    def from_raw(
        cls,
        space: ElectronicSpace,
        *,
        one_body: np.ndarray,
        two_body: np.ndarray | None = None,
        constant: complex = 0.0,
    ) -> ElectronicIntegrals: ...

    def to_operator(self) -> Operator: ...
```

#### Supported conversion

- `Operator -> ElectronicIntegrals` is supported only for:
  - constants
  - one-body number-conserving terms
  - two-body number-conserving terms
- It must raise `UnsupportedTransformError` on:
  - pairing terms
  - three-body or higher-body terms
  - mixed unsupported term sets

### 5. `models/`

The `models` layer is the main user-facing frontend for common Hamiltonians.

#### Required builders

```python
def hubbard(
    space: ElectronicSpace,
    *,
    bonds: str | Sequence[str] | Sequence[Bond] = "all",
    t: complex | Mapping[str, complex],
    U: complex,
    mu: complex | Mapping[int, complex] | None = None,
) -> Operator: ...


def slater_kanamori(
    space: ElectronicSpace,
    *,
    sites: str | int | Sequence[int] = "all",
    orbitals: str | int | Sequence[str | int] = "all",
    U: complex,
    Up: complex,
    J: complex = 0.0,
    Jp: complex = 0.0,
) -> Operator: ...


def extended_hubbard(
    space: ElectronicSpace,
    *,
    hopping: complex | Mapping[str, complex],
    onsite_U: complex,
    intersite_V: complex | Mapping[str, complex] | None = None,
    bonds: str | Sequence[str] | Sequence[Bond] = "all",
) -> Operator: ...


def chemical_potential(
    space: ElectronicSpace,
    *,
    mu: complex | Mapping[int, complex],
    sites: str | int | Sequence[int] = "all",
    orbitals: str | int | Sequence[str | int] = "all",
    spin: str = "both",
) -> Operator: ...


def density_density(... ) -> Operator: ...
def exchange(... ) -> Operator: ...
def pair_hopping(... ) -> Operator: ...
def soc(... ) -> Operator: ...
```

#### Builder rules

- `plus_hc=True` behavior must be implemented where the builder semantics imply it.
- Bond-kind mapping is preferred over raw edge-list loops.
- The v1 `extended_hubbard()` builder must be general-purpose; the current hard-coded donor/silicon
  parameter set is moved out of core into a preset or application-specific module.
- There is no hidden inference of rotationally invariant Slater-Kanamori parameters in v1:
  `U`, `Up`, `J`, and `Jp` are explicit inputs.

## Frontend Ergonomics

### Example 1: 1D Hubbard model

```python
from mbkit.operator import ElectronicSpace, LineLattice, models

lattice = LineLattice(4, boundary="open")
space = ElectronicSpace(lattice, orbitals=["a"])

H = models.hubbard(space, t=-1.0, U=4.0, bonds="all")
```

Requirements:

- no raw spin-orbital indices
- no raw QuSpin strings
- no manual Hermitian-conjugate duplication

### Example 2: Custom lattice/graph Hamiltonian

```python
from mbkit.operator import Bond, ElectronicSpace, GeneralLattice, models

lattice = GeneralLattice(
    num_sites=4,
    bonds=[
        Bond(0, 1, kind="nn"),
        Bond(1, 2, kind="nn"),
        Bond(2, 3, kind="nn"),
        Bond(0, 2, kind="diag"),
    ],
)
space = ElectronicSpace(lattice, orbitals=["a"])

H = (
    models.extended_hubbard(
        space,
        hopping={"nn": -1.0, "diag": -0.2},
        onsite_U=4.0,
        intersite_V={"diag": 0.5},
    )
    + models.chemical_potential(space, mu={0: 0.1, 3: -0.1})
)
```

### Example 3: Multi-orbital local interaction

```python
from mbkit.operator import ElectronicSpace, LineLattice, models

lattice = LineLattice(2, boundary="open")
space = ElectronicSpace(lattice, orbitals=["dxz", "dyz"])

H = models.slater_kanamori(
    space,
    U=4.0,
    Up=2.6,
    J=0.7,
    Jp=0.7,
)
```

## Backend And Export Strategy

### Compiler boundary

The new symbolic `Operator` has **no**:

- basis caching
- sparse-matrix construction
- direct diagonalization
- solver-specific tensor logic

All of that moves into `mbkit.operator.transforms.*` and the solver layer.

### Export targets

#### `transforms.quspin`

Input:

- new symbolic `Operator`

Output:

- temporary legacy `mbkit.operator.legacy.Operator`
- or directly the QuSpin static-list representation if/when the solver layer is updated

Support:

- all terms expressible by the current legacy lowering rules
- exact coefficient preservation on small-system tests

Failure:

- raise `UnsupportedTransformError` with the offending term and why it cannot be lowered

#### `transforms.integrals`

Input:

- new symbolic `Operator`

Output:

- `ElectronicIntegrals`

Support:

- number-conserving one-/two-body operators
- spin-conserving and spin-mixing one-/two-body terms are both allowed in spin-orbital form

Failure:

- any non-number-conserving or higher-body term

#### `transforms.block2`, `transforms.quantax`, `transforms.openfermion`

Input:

- new symbolic `Operator` directly or `ElectronicIntegrals`, depending on target

Rule:

- compiler implementations should reuse the symbolic or integral compiler boundary rather than
  re-encoding Hamiltonian semantics independently

#### `transforms.pyscf`

Do not add a dedicated `pyscf.py` transform in v1. PySCF export can be expressed through
`ElectronicIntegrals` plus solver-specific conversion helpers.

## Representation Rules

These rules are mandatory in the redesign.

- Constants are first-class.
- Normal ordering is mandatory in the symbolic layer.
- Fermionic sign handling happens in the symbolic layer.
- Backend exporters are not allowed to implement ad hoc algebra that should have been handled in
  `Operator`.
- `Operator` metadata tracks:
  - max body rank
  - particle-number change set
  - spin-change set
  - hermiticity query
- Symbolic labels are never raw backend op strings in the user-facing API.
- `ElectronicSpace` owns the only public mode-index convention.

## Migration Strategy

This redesign is a hard reset of the operator frontend.

### Namespace decisions

- The new API is the only documented API for new user code.
- The current public helpers move to `mbkit.operator.legacy`.
- `mbkit.operator.__init__` exports only the new API.
- `legacy` is importable but not prominently documented.

### Temporary compatibility boundary

Existing solver code continues working during migration by compiling:

```text
new Operator -> transforms.quspin -> legacy Operator -> existing solvers
```

This compiler boundary allows the symbolic/lattice frontend to be redesigned without forcing an
all-at-once solver rewrite.

### Mapping from current to new API

- `create_u(nsites, i)` -> `space.create(i, spin="up")`
- `create_d(nsites, i)` -> `space.create(i, spin="down")`
- `annihilate_u(nsites, i)` -> `space.destroy(i, spin="up")`
- `number_u(nsites, i)` -> `space.number(i, spin="up")`
- `Hubbard(neighbors, nsites, U, t)` -> `models.hubbard(space, t=t, U=U, bonds=...)`
- `Slater_Kanamori(nsites, t, ...)` -> explicit one-body operator construction plus
  `models.slater_kanamori(...)`
- specialized extended-Hubbard preset -> general `models.extended_hubbard(...)` plus optional
  preset/application wrappers

### Implementation phases

#### Phase 1: Design and API freeze

Deliverables:

- this design doc
- reviewed and approved API signatures
- reviewed migration rules

Acceptance:

- no unresolved interface decisions remain for the new operator frontend

#### Phase 2: Core public abstractions

Implement:

- `ElectronicSpace`
- `Lattice`, `GeneralLattice`, `LineLattice`, `SquareLattice`, `Bond`
- new backend-neutral `Operator`

Acceptance:

- users can construct basic one-body and local interaction operators without raw mode integers
- constants and normal ordering are implemented

#### Phase 3: Integral container and compilers

Implement:

- `ElectronicIntegrals`
- `transforms.integrals`
- `transforms.quspin`

Acceptance:

- new operators can compile to the current legacy QuSpin path with result equivalence on small
  systems
- supported operators can compile to structured one-/two-body integrals

#### Phase 4: Rebuild model constructors

Implement:

- `models.hubbard`
- `models.slater_kanamori`
- `models.extended_hubbard`
- `models.chemical_potential`
- core interaction builders

Acceptance:

- the canonical examples in this document become passing tests
- the current hard-coded extended-Hubbard preset is removed from core model logic

#### Phase 5: Solver migration

Implement:

- solver-facing code compiles from the new symbolic operator layer
- direct dependence on the legacy representation is retired module by module

Acceptance:

- solver modules no longer import the old public operator helpers
- the legacy representation becomes an internal compatibility shim only

## Testing And Acceptance Criteria

### Design-doc acceptance examples

These examples must become executable tests as soon as implementation starts:

- build a 1D Hubbard Hamiltonian without manually specifying spin-orbital indices
- build a custom lattice/graph Hamiltonian from typed bond groups
- build a multi-orbital local interaction Hamiltonian with onsite, exchange, and pair-hopping
  terms

### Symbolic layer tests

- constant-only operator behavior
- operator addition and scalar multiplication
- operator product using `@`
- fermionic sign handling
- normal ordering
- simplification and zero-term dropping
- adjoint and hermiticity
- body-rank detection
- particle-number change metadata
- spin-change metadata

### Space and lattice tests

- canonical mode ordering
- site/orbital/spin label lookup
- selector validation
- predefined lattice bond generation
- `GeneralLattice` bond-kind filtering

### Compiler tests

- exact equivalence between:
  - new symbolic operator -> QuSpin compiler
  - current legacy operator output
- `Operator -> ElectronicIntegrals` conversion for supported one-/two-body Hamiltonians
- explicit failure on unsupported non-number-conserving or higher-body terms
- tensor convention tests for one-body and two-body coefficients

### Model-builder tests

- Hubbard construction on `LineLattice` and `SquareLattice`
- Slater-Kanamori construction on a multi-orbital site
- general extended-Hubbard construction with bond-type selection
- chemical-potential and density-density helpers
- automatic Hermitian-conjugate behavior where promised

### Migration tests

- the new symbolic frontend can feed existing solver workflows through the legacy QuSpin compiler
- old solver results remain unchanged on small regression systems during the transition

## Assumptions

- The redesign only covers electronic fermionic operators.
- The first milestone is lattice-first, not chemistry-first.
- Numeric coefficients only in v1.
- The current public operator API is allowed to break.
- Existing solver internals may temporarily rely on a compiler into the legacy representation.

## Immediate Next Steps

1. Freeze this document.
2. Move the current implementation under `mbkit.operator.legacy`.
3. Implement `ElectronicSpace`, `Lattice`, and the new symbolic `Operator`.
4. Implement the QuSpin legacy compiler boundary before touching solver internals.
5. Rebuild the common model constructors on top of the new public API.
