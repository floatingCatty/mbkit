"""Public MP2 solver facade."""

from __future__ import annotations

from .base import SolverFacade


class MP2Solver(SolverFacade):
    """Method-based MP2 solver facade.

    The current implementation routes to the PySCF UMP2-capable reference
    backend.
    """

    solver_family = "qc"
    default_backend = "pyscf_reference"

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        super().__init__(*args, backend=backend or self.default_backend, method="mp2", **kwargs)


MP2_solver = MP2Solver
