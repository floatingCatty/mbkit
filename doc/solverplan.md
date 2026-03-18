# Solver Integration Plan

Research summary prepared on March 18, 2026.

## Goal

`mbkit` now has a fairly general symbolic many-body Hamiltonian representation.
The next major step is to connect that frontend to the strongest solver
ecosystems available, across exact diagonalization, tensor networks, quantum
Monte Carlo, neural-network solvers, and impurity/DMFT workflows.

The core design principle should stay the same as the operator layer:

- `mbkit` owns the Hamiltonian construction frontend
- solver packages own the heavy numerical algorithms
- `mbkit` adds clean lowering / transform layers for each backend

## Recommended Shortlist

These are the strongest solver packages to prioritize for `mbkit`.

### Top Priority

- `QuSpin`
  - Best near-term exact diagonalization backend for lattice many-body models.
  - Strong fit with the current `mbkit` operator lowering path.
  - Official docs: https://quspin.github.io/QuSpin/

- `block2` / `pyblock2`
  - High-performance DMRG / MPS backend and one of the most important tensor-network targets.
  - Especially valuable for large fermionic lattice Hamiltonians and chemistry-style QC tensors.
  - Official docs: https://block2.readthedocs.io/
  - Package index: https://pypi.org/project/block2/

- `PySCF`
  - Core chemistry solver ecosystem for FCI, CASSCF, CC, SCF, and chemistry-style workflows.
  - Also an important umbrella for DMRG-based chemistry integrations.
  - Official docs: https://pyscf.org/
  - DMRGSCF repo: https://github.com/pyscf/dmrgscf

- `NetKet`
  - Best current target for neural quantum states and variational Monte Carlo.
  - Very relevant if `mbkit` wants to support modern machine-learning-based many-body solvers.
  - Official docs: https://netket.readthedocs.io/en/latest/api/api.html
  - Repo: https://github.com/netket/netket

- `TeNPy`
  - Python-native tensor-network package with flexible MPS / DMRG / TEBD workflows.
  - Excellent as a developer-friendly tensor-network backend even when `block2` remains the production-performance option.
  - Official docs: https://tenpy.readthedocs.io/
  - Repo: https://github.com/tenpy/tenpy

- `TRIQS` impurity solvers
  - Important route if `mbkit` wants strong impurity and DMFT support.
  - Most relevant packages are `triqs_cthyb`, `triqs_ctseg`, `triqs_hubbardI`, and `triqs_hartree_fock`.
  - TRIQS app overview: https://triqs.github.io/triqs/latest/applications.html
  - CTHYB docs: https://triqs.github.io/cthyb/latest/
  - CTSEG docs: https://triqs.github.io/ctseg/latest/_ref/triqs_ctseg.solver.Solver.html

### Second Wave

- `ITensors.jl` / `ITensorMPS.jl`
  - Very strong Julia tensor-network ecosystem.
  - Best treated as a serious later-stage backend if `mbkit` is willing to support a Julia bridge.
  - Repos: https://github.com/ITensor/ITensors.jl
  - https://github.com/ITensor/ITensorMPS.jl

- `YASTN`
  - Symmetry-aware tensor-network library, useful for advanced MPS / PEPS work.
  - Official docs: https://yastn.github.io/yastn/
  - Repo: https://github.com/yastn/yastn

- `XDiag`
  - High-performance exact diagonalization package.
  - More specialized than QuSpin, but very strong for serious ED workflows.
  - Docs: https://awietek.github.io/xdiag/
  - Repo: https://github.com/awietek/xdiag

- `ALF`
  - Major auxiliary-field lattice fermion QMC package.
  - Important if `mbkit` grows a strong determinantal QMC route.
  - Docs: https://gitpages.physik.uni-wuerzburg.de/ALF/ALF_Webpage/page/documentation

- `HANDE-QMC`
  - Strong stochastic many-body suite with FCIQMC / CCMC / DMQMC.
  - Official site: https://www.hande.org.uk/

- `Dice`
  - Advanced selected-CI / FCIQMC / AFQMC oriented solver package.
  - Repo: https://github.com/sanshar/Dice

- `SmoQyDQMC.jl`
  - Modern determinant-QMC Julia ecosystem for Hubbard-like lattice models.
  - Docs: https://smoqysuite.github.io/SmoQyDQMC.jl/stable/

- `solid_dmft`
  - DMFT workflow layer built around TRIQS solvers.
  - Best viewed as a workflow integration target rather than a raw solver backend.
  - Docs: https://triqs.github.io/solid_dmft/

- `DCore`
  - Another DMFT workflow ecosystem.
  - Docs: https://issp-center-dev.github.io/DCore/v3.0.0/about.html

- `QCMaquis`
  - Advanced chemistry DMRG backend.
  - Repo: https://github.com/qcscine/qcmaquis

- `CheMPS2`
  - Mature chemistry DMRG package.
  - Repo: https://github.com/SebWouters/CheMPS2

## Recommended Priority Order For `mbkit`

If `mbkit` wants a practical, high-impact integration roadmap, the recommended order is:

1. `QuSpin`
2. `block2`
3. `PySCF`
4. `NetKet`
5. `TeNPy`
6. `TRIQS` impurity solvers
7. `ALF`
8. `HANDE-QMC`
9. `ITensorMPS.jl`

This ordering reflects:

- immediate usefulness for the current electronic Hamiltonian frontend
- solver maturity and ecosystem importance
- realistic integration complexity
- breadth across condensed-matter and chemistry workflows

## Why These Packages

The shortlist gives `mbkit` strong coverage across solver classes:

- exact diagonalization:
  - `QuSpin`, `XDiag`
- tensor networks / DMRG:
  - `block2`, `TeNPy`, `ITensorMPS.jl`, `YASTN`, `QCMaquis`, `CheMPS2`
- chemistry solvers:
  - `PySCF`
- impurity / DMFT:
  - `TRIQS` solvers, `solid_dmft`, `DCore`
- Monte Carlo:
  - `ALF`, `HANDE-QMC`, `Dice`, `SmoQyDQMC.jl`
- neural quantum states / VMC:
  - `NetKet`

That solver mix is broad enough that `mbkit` can stay solver-agnostic while
still targeting state-of-the-art backends in the main many-body directions.

## Design Recommendation For `mbkit`

The solver layer should likely be organized around transform families, not one
off wrappers.

Recommended transform targets:

- symmetry / operator-string backend:
  - `QuSpin`
- QC tensor backend:
  - `block2`, `PySCF`, chemistry DMRG backends
- tensor-network lattice backend:
  - `TeNPy`, later `ITensorMPS.jl`, `YASTN`
- graph / variational backend:
  - `NetKet`
- impurity-action / Green's-function workflow backend:
  - `TRIQS`
- determinant / auxiliary-field QMC backend:
  - `ALF`, `SmoQyDQMC.jl`

This suggests a future solver architecture like:

- `mbkit.solver.ed`
- `mbkit.solver.dmrg`
- `mbkit.solver.qc`
- `mbkit.solver.nqs`
- `mbkit.solver.dmft`
- `mbkit.solver.qmc`

with backend-specific adapters under each family.

## Immediate Recommendation

For the next implementation stage, `mbkit` should focus on:

1. stabilizing `QuSpin`, `block2`, and `PySCF`
2. designing a clean backend interface for `NetKet`
3. evaluating whether `TeNPy` or `TRIQS` should be the next major integration

That path keeps the package focused while still leaving a clear expansion route
to truly state-of-the-art many-body solver ecosystems.
