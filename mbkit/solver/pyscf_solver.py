"""Public PySCF solver façade."""

from __future__ import annotations

from .base import SolverFacade
from .backends.pyscf_fci import _PYSCF_IMPORT_ERROR, UHFFCIHamiltonian, _build_uhf_fci_hamiltonian
from .backends.pyscf_reference import PySCFReferenceProblem, _build_pyscf_reference_problem


class PySCFSolver(SolverFacade):
    """PySCF solver façade.

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
