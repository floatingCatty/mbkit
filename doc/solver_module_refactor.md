# Solver Module Refactor Plan

Prepared on March 18, 2026.

## Goal

Refactor `mbkit.solver` from a small set of backend-specific solver files into a
backend-family-driven architecture that can grow toward a state-of-the-art
many-body solver package.

The target is:

- keep the public user-facing API simple
- isolate backend-specific logic behind explicit adapters
- separate Hamiltonian compilation from numerical solving
- make it practical to add new solver backends such as quimb and TeNPy without
  rewriting the public solver surface each time

## Current Code Status

Current public solvers:

- `EDSolver`
- `DMRGSolver`
- `PySCFSolver`

Current implementation status:

- `EDSolver` is effectively a QuSpin backend implemented directly in
  `mbkit/solver/ed_solver.py`
- `DMRGSolver` is effectively a block2 QC-tensor backend implemented directly in
  `mbkit/solver/dmrg_solver.py`
- `PySCFSolver` is effectively a PySCF UHF-FCI backend implemented directly in
  `mbkit/solver/pyscf_solver.py`

Current structural problems:

- solver files mix frontend coercion, Hamiltonian compilation, backend dispatch,
  and observable logic
- there is no backend registry or capability model
- there is no shared compiled-problem layer
- the current DMRG path assumes QC tensors, which blocks natural TeNPy/quimb
  integration
- old experimental files still live under `mbkit/solver/`

## Research Summary

Official docs and current installed backends indicate three different solver
input contracts matter most right now:

- QuSpin wants basis + operator-string Hamiltonians
  - https://quspin.github.io/QuSpin/generated/quspin.operators.hamiltonian.html
- block2 and PySCF want structured one-/two-body tensor data
  - https://block2.readthedocs.io/en/latest/user/interfaces.html
  - https://pyscf.org/user/ci.html
- TeNPy and quimb want local lattice couplings / MPO-style Hamiltonian data
  - https://tenpy.readthedocs.io/en/stable/reference/tenpy.models.model.CouplingModel.html
  - https://quimb.readthedocs.io/en/latest/

This means the solver architecture should be organized by backend family, not
just by one file per currently supported backend.

## Target Architecture

Planned layout:

- `mbkit/solver/base.py`
  - backend protocol and user-facing façade helpers
- `mbkit/solver/capabilities.py`
  - backend capability metadata
- `mbkit/solver/registry.py`
  - lazy backend registry
- `mbkit/solver/result.py`
  - future shared result container layer
- `mbkit/solver/compile/`
  - backend-family compilation targets
- `mbkit/solver/backends/`
  - concrete backend adapters

Planned backend families:

- operator-string / basis
  - QuSpin
- QC tensors / integrals
  - block2
  - PySCF
- local lattice terms / MPO
  - quimb
  - TeNPy

## Public API Strategy

Keep the public solver names:

- `EDSolver`
- `DMRGSolver`
- `PySCFSolver`

But make them backend-selecting façades instead of the actual backend
implementations.

Examples:

- `EDSolver()` should default to the QuSpin backend
- `DMRGSolver()` should default to the block2 backend
- `PySCFSolver()` should default to the PySCF FCI backend

Internally, concrete classes should use precise names such as:

- `QuSpinEDBackend`
- `Block2DMRGBackend`
- `PySCFFCIBackend`
- later `QuimbDMRGBackend`
- later `TeNPYDMRGBackend`

## Implementation Phases

### Phase 1

- add solver backend registry
- add backend capability metadata
- split current QuSpin/block2/PySCF implementations into `solver/backends/`
- add façade classes that delegate to registered backends
- add `solver/compile/quspin.py`
- add `solver/compile/qc.py`

### Phase 2

- add a local lattice-term compiler for tensor-network backends
- refactor observables into shared backend-independent helpers
- add live backend-on tests for QuSpin, block2, and PySCF

### Phase 3

- add `quimb` backend
- add `TeNPy` backend
- add backend equivalence tests on small lattice models

### Phase 4

- add richer result objects
- support additional solver tasks such as excited states and time evolution
- clean or move old experimental solver files out of the main solver namespace

## Immediate Implementation Target

Start with Phase 1:

- build the registry and façade structure
- move existing backends into `solver/backends/`
- keep the public names stable
- preserve the current functionality while making the architecture ready for
  quimb/TeNPy integration next
