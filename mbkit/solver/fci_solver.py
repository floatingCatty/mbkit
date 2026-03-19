"""Public FCI solver facade."""

from __future__ import annotations

from .base import SolverFacade


class FCISolver(SolverFacade):
    """Method-based exact FCI solver facade.

    The current implementation routes to the PySCF exact-sector backend. The
    public API is intentionally method-based so future FCI backends can be
    added without changing user code.

    This class inherits the shared `SolverFacade` runtime helpers, so the
    normal workflow is `solve(...)` followed by common helpers such as
    `expect()`, `expect_value()`, `diagnostics()`, and
    `available_properties()`.
    """

    solver_family = "qc"
    default_backend = "pyscf_fci"

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        super().__init__(*args, backend=backend or self.default_backend, method="fci", **kwargs)


FCI_solver = FCISolver
