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
pip install -e ".[nqs]"
pip install -e ".[dmrg]"
pip install -e ".[pyscf]"
```

## Quickstart

```python
from mbkit import EDSolver, Hubbard

neighbors = [[(0, 1)]]
hamiltonian = Hubbard(neighbors=neighbors, nsites=2, U=4.0, t=1.0)

solver = EDSolver()
solver.solve(hamiltonian, nsites=2, n_particles=[(1, 1)])

print("Ground-state energy:", solver.energy())
print("Double occupancy:", solver.docc())
print("Spin quantum number:", solver.s2())
print("One-body RDM shape:", solver.rdm1().shape)
```

The legacy embedding-style solver classes are still available under `mbkit.solver`, but only `EDSolver` is treated as the stable public solver in the current milestone.
