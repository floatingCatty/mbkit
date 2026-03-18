"""Public exact-diagonalization solver façade."""

from __future__ import annotations

from .base import SolverFacade


class EDSolver(SolverFacade):
    """Exact-diagonalization solver façade.

    The default backend is the QuSpin implementation registered under the
    `ed` solver family.
    """

    solver_family = "ed"
    default_backend = "quspin"


ED_solver = EDSolver
