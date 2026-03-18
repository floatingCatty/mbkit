# mbkit

`mbkit` is a many-body toolkit centered on Hamiltonian construction first and solver adapters second. The core v1 surface keeps `mbkit.operator.Operator` as the canonical intermediate representation, but the normal user path is now `space`-first: create an `ElectronicSpace`, build Hamiltonians directly from it, and pass the result to a solver.

## Install

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[dmrg]"
pip install -e ".[interop]"
pip install -e ".[pyscf]"
```

## Quickstart

```python
from mbkit import EDSolver, chain

space = chain(2, orbitals=["a"], n_electrons_per_spin=(1, 1))
hamiltonian = space.hubbard(t=1.0, U=4.0)

solver = EDSolver()
solver.solve(hamiltonian)

print("Ground-state energy:", solver.energy())
print("Double occupancy:", solver.docc())
print("Spin quantum number:", solver.s2())
print("One-body RDM shape:", solver.rdm1().shape)
```

For extended lattice models, the recommended workflow is shell-first:

```python
from mbkit import square

space = square(4, 4, orbitals=["a"], max_shell=3)
print(space.available_shells())
print(space.bond_summary())

hamiltonian = space.hubbard(t={1: 1.0, 2: 0.3, 3: 0.15}, U=4.0)
```

Here `shells=` selects neighbor order, while `bonds=` optionally refines that to
a geometric family such as `horizontal`, `vertical`, or `diagonal`.

The advanced path is still available when you want explicit control over the geometry:

```python
from mbkit import ElectronicSpace, SquareLattice

space = ElectronicSpace(SquareLattice(2, 2), orbitals=["a"])
hamiltonian = space.extended_hubbard(hopping=1.0, onsite_U=4.0, intersite_V={"horizontal": 1.0})
```

`mbkit` exposes direct `EDSolver`, `DMRGSolver`, and `PySCFSolver` implementations on the new operator frontend. Optional OpenFermion and Qiskit Nature bridges live under `mbkit.operator.interop`.
