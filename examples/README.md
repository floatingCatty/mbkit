# Examples

This folder contains runnable tutorials that demonstrate the public `mbkit`
operator-construction workflow together with the solver façade API.

## Formal Tutorials

### `tutorial_ed_solver.py`

Exact-diagonalization tutorial covering:

- building the same four-site one-orbital Hubbard chain used by the block2 and
  PySCF tutorials
- solving a fixed two-electron problem with `EDSolver`
- computing energies, one-body densities, double occupancies, and `<S^2>`
- using the generic `expect()`, `expect_value()`, `available_properties()`,
  and `diagnostics()` helpers

Run it from the source tree with:

```bash
python examples/tutorial_ed_solver.py
```

### `tutorial_quantax_nqs_solver.py`

Quantax NQS tutorial covering:

- building a two-site one-orbital hopping dimer explicitly from operator terms
- solving the fixed `(n_up, n_down) = (1, 1)` sector with `NQSSolver`
- comparing the NQS energy against a lightweight `EDSolver` reference on the same Hamiltonian
- computing energies, one-body densities, double occupancies, and `<S^2>`
- using the generic `expect()`, `expect_value()`, `available_properties()`, and `diagnostics()` helpers
- noting that `n_particles=(1, 1)` is intentional because an integer electron count selects the full total-particle sector in `NQSSolver`

This tutorial requires the optional NQS dependency set. Quantax depends on
JAX, so install the appropriate JAX build for your platform and accelerator if
needed. The exact-sampling workflow in this example also requires QuSpin,
which is included in `.[nqs]`.

```bash
pip install -e ".[nqs]"
python examples/tutorial_quantax_nqs_solver.py
```

### `tutorial_block2_dmrg_solver.py`

block2 / pyblock2 DMRG tutorial covering:

- building a one-orbital Hubbard chain explicitly from operator terms
- solving a fixed two-electron problem with `DMRGSolver`
- computing energies, densities, double occupancies, and `<S^2>`
- inspecting the generic façade helpers shared with the ED solver

This tutorial requires the optional DMRG dependency set:

```bash
pip install -e ".[dmrg]"
python examples/tutorial_block2_dmrg_solver.py
```

The DMRG tutorial writes scratch data under `/tmp/mbkit_block2_tutorial`.

### `tutorial_pyscf_mp2_solver.py`

PySCF MP2 tutorial covering:

- building the same four-site one-orbital Hubbard chain as the DMRG tutorial
- solving a fixed two-electron problem with `MP2Solver`
- inspecting the method-specific observable semantics of
  `rdm1(kind="reference")`, `rdm1(kind="unrelaxed")`, and
  `s2(kind="reference")`
- noting that UMP2 is non-variational on this benchmark, so its energy can
  fall below the matching ED value
- using `available_properties()` and `diagnostics()` to understand what the
  backend exposes

This tutorial requires the optional PySCF dependency set:

```bash
pip install -e ".[pyscf]"
python examples/tutorial_pyscf_mp2_solver.py
```

### `tutorial_pyscf_ccsd_solver.py`

PySCF CCSD tutorial covering:

- building the same four-site one-orbital Hubbard chain as the ED and MP2 tutorials
- solving a fixed two-electron problem with `CCSDSolver`
- inspecting the method-specific observable semantics of
  `rdm1(kind="reference")`, `rdm1(kind="unrelaxed")`, and
  `s2(kind="reference")`
- using `available_properties()` and `diagnostics()` to understand what the
  backend exposes

This tutorial requires the optional PySCF dependency set:

```bash
pip install -e ".[pyscf]"
python examples/tutorial_pyscf_ccsd_solver.py
```

### `tutorial_pyscf_ccsdt_solver.py`

PySCF CCSD(T) tutorial covering:

- building the same four-site one-orbital Hubbard chain as the ED, MP2, and CCSD tutorials
- solving a fixed two-electron problem with `CCSDTSolver`
- inspecting the reference-only observable semantics of
  `rdm1(kind="reference")` and `s2(kind="reference")`
- reading the perturbative triples correction from `diagnostics()`

This tutorial requires the optional PySCF dependency set:

```bash
pip install -e ".[pyscf]"
python examples/tutorial_pyscf_ccsdt_solver.py
```
