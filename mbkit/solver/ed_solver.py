"""Public exact-diagonalization solver façade."""

from __future__ import annotations

from .base import SolverFacade


class EDSolver(SolverFacade):
    """Exact-diagonalization solver façade.

    The default backend is the QuSpin implementation registered under the
    `ed` solver family.

    Usage follows the shared `SolverFacade` pattern: instantiate the solver,
    call `solve(...)`, then inspect `available_properties()` or call inherited
    helpers such as `expect()`, `expect_value()`, and `diagnostics()` together
    with ED observables like `energy()`, `rdm1()`, and `s2()`.
    """

    solver_family = "ed"
    default_backend = "quspin"


ED_solver = EDSolver
