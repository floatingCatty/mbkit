"""Public MP2 solver facade."""

from __future__ import annotations

from .base import SolverFacade


class MP2Solver(SolverFacade):
    """Method-based MP2 solver facade.

    The current implementation routes to the PySCF UMP2-capable reference
    backend.

    As with the other public solver facades, users call `solve(...)` first and
    then rely on the inherited helper surface from `SolverFacade`. For MP2 in
    particular, `available_properties()` and `diagnostics()` make it easier to
    discover the narrower observable semantics exposed by the backend.
    """

    solver_family = "qc"
    default_backend = "pyscf_reference"

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        super().__init__(*args, backend=backend or self.default_backend, method="mp2", **kwargs)


MP2_solver = MP2Solver
