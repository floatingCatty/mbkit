"""Public neural-quantum-state solver facade."""

from __future__ import annotations

from .base import SolverFacade
from .backends.quantax_nqs import _QUANTAX_IMPORT_ERROR


class NQSSolver(SolverFacade):
    """Quantax-backed neural-quantum-state solver facade."""

    solver_family = "nqs"
    default_backend = "quantax"

NQS_solver = NQSSolver

