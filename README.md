# mbkit

`mbkit` is a many-body toolkit centered on Hamiltonian construction first and solver adapters second. The core v1 surface keeps `mbkit.operator.Operator` as the canonical intermediate representation and provides a direct ED path for small problems.

## Install

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[dmrg]"
pip install -e ".[pyscf]"
```

## Quickstart

```python
from mbkit import EDSolver, ElectronicSpace, LineLattice, models

space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
hamiltonian = models.hubbard(space, t=1.0, U=4.0)

solver = EDSolver()
solver.solve(hamiltonian, n_particles=[(1, 1)])

print("Ground-state energy:", solver.energy())
print("Double occupancy:", solver.docc())
print("Spin quantum number:", solver.s2())
print("One-body RDM shape:", solver.rdm1().shape)
```

`mbkit` now exposes direct `EDSolver`, `DMRGSolver`, and `PySCFSolver` implementations on the new operator frontend. The old legacy operator and embedding-style solver stack has been retired.
