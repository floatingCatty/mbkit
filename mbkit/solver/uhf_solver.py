"""Public UHF solver facade."""

from __future__ import annotations

from .base import SolverFacade


class UHFSolver(SolverFacade):
    """Method-based unrestricted Hartree-Fock solver facade.

    The current implementation routes to the PySCF UHF-compatible reference
    backend.

    The shared runtime surface comes from `SolverFacade`, so users interact
    with this solver the same way as the other public facades: call
    `solve(...)`, inspect `available_properties()`, then query supported
    helpers and observables.
    """

    solver_family = "qc"
    default_backend = "pyscf_reference"

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        super().__init__(*args, backend=backend or self.default_backend, method="uhf", **kwargs)


UHF_solver = UHFSolver
