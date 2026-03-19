"""Compatibility umbrella for package-based PySCF solver selection."""

from __future__ import annotations

from .base import SolverFacade
from .backends.pyscf_fci import _PYSCF_IMPORT_ERROR, UHFFCIHamiltonian, _build_uhf_fci_hamiltonian
from .backends.pyscf_reference import PySCFReferenceProblem, _build_pyscf_reference_problem


class PySCFSolver(SolverFacade):
    """Compatibility PySCF solver façade.

    New code should prefer the method-based public facades such as
    `FCISolver`, `UHFSolver`, and `MP2Solver`. This class remains available as
    a package-level umbrella for compatibility and explicit backend access.

    Method routing:
    - `method="fci"` uses the exact sector backend on the general spin-conserving path
    - `method="uhf"` and `method="mp2"` use the UHF-based reference backend on the
      narrower UHF-compatible Hamiltonian class
    """

    solver_family = "qc"
    default_backend = "pyscf_fci"

    def __init__(self, *args, backend: str | None = None, method: str = "fci", **kwargs) -> None:
        if backend is None:
            backend = "pyscf_fci" if method == "fci" else "pyscf_reference"
        super().__init__(*args, backend=backend, method=method, **kwargs)


PYSCF_solver = PySCFSolver
