"""Public CCSD(T) solver facade."""

from __future__ import annotations

from .base import SolverFacade


class CCSDTSolver(SolverFacade):
    """Method-based perturbative CCSD(T) solver facade.

    The current implementation routes to the PySCF unrestricted CCSD backend
    and adds the perturbative triples correction to the total energy. Because
    the backend does not expose triples-corrected one-body observables,
    `rdm1()` and `s2()` remain reference-only through explicit `kind=`
    selectors.
    """

    solver_family = "qc"
    default_backend = "pyscf_reference"

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        super().__init__(*args, backend=backend or self.default_backend, method="ccsd(t)", **kwargs)


CCSDT_solver = CCSDTSolver
