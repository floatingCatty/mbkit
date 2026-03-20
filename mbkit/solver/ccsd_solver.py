"""Public CCSD solver facade."""

from __future__ import annotations

from .base import SolverFacade


class CCSDSolver(SolverFacade):
    """Method-based CCSD solver facade.

    The current implementation routes to the PySCF unrestricted CCSD-capable
    reference backend.

    As with the other public solver facades, users call `solve(...)` first and
    then rely on the inherited helper surface from `SolverFacade`. For CCSD,
    `available_properties()` and `diagnostics()` clarify that `rdm1()` uses
    explicit `kind=` selectors and that only the reference `<S^2>` is exposed.
    """

    solver_family = "qc"
    default_backend = "pyscf_reference"

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        super().__init__(*args, backend=backend or self.default_backend, method="ccsd", **kwargs)


CCSD_solver = CCSDSolver
