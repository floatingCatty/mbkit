# Examples

This folder contains runnable tutorials that demonstrate the public `mbkit`
operator-construction workflow together with the solver façade API.

## Formal Tutorials

### `tutorial_ed_solver.py`

Exact-diagonalization tutorial covering:

- building a small Hubbard Hamiltonian term by term
- solving in a fixed particle sector with `EDSolver`
- computing energies, one-body densities, double occupancies, and `<S^2>`
- using the generic `expect()`, `expect_value()`, `available_properties()`,
  and `diagnostics()` helpers

Run it from the source tree with:

```bash
python examples/tutorial_ed_solver.py
```

### `tutorial_block2_dmrg_solver.py`

block2 / pyblock2 DMRG tutorial covering:

- building a one-orbital Hubbard chain explicitly from operator terms
- solving a half-filled problem with `DMRGSolver`
- computing energies, densities, double occupancies, and `<S^2>`
- inspecting the generic façade helpers shared with the ED solver

This tutorial requires the optional DMRG dependency set:

```bash
pip install -e ".[dmrg]"
python examples/tutorial_block2_dmrg_solver.py
```

The DMRG tutorial writes scratch data under `/tmp/mbkit_block2_tutorial`.
