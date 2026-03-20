"""Compatibility umbrella for package-based PySCF solver selection."""

from __future__ import annotations

from .base import SolverFacade
from .backends.pyscf_reference import (
    _PYSCF_IMPORT_ERROR,
    _SUPPORTED_REFERENCE_METHODS,
    _build_pyscf_reference_problem,
    normalize_reference_method,
)


class PySCFSolver(SolverFacade):
    """Compatibility PySCF solver façade.

    New code should prefer the method-based public facades such as
    `UHFSolver`, `MP2Solver`, `CCSDSolver`, and `CCSDTSolver`. This class
    remains available as a package-level umbrella for compatibility and explicit
    backend access.

    Runtime usage is still the shared `SolverFacade` pattern: instantiate the
    solver, call `solve(...)`, then use inherited helpers such as
    `diagnostics()` and `available_properties()` to inspect what the selected
    backend exposes.

    Method routing:
    - `method="uhf"` is the default and uses the UHF reference backend
    - `method="mp2"` uses the UHF-based MP2 backend on the narrower
      UHF-compatible Hamiltonian class
    - `method="ccsd"` uses the UHF-based CCSD backend on the same Hamiltonian class
    - `method="ccsd(t)"` adds the perturbative triples correction on top of CCSD

    Exact many-body solutions are handled by `EDSolver`, not PySCF.
    """

    solver_family = "qc"
    default_backend = "pyscf_reference"

    def __init__(self, *args, backend: str | None = None, method: str = "uhf", **kwargs) -> None:
        method = normalize_reference_method(method)
        if method not in _SUPPORTED_REFERENCE_METHODS:
            supported = ", ".join(f"`method={name!r}`" for name in _SUPPORTED_REFERENCE_METHODS)
            raise ValueError(
                f"PySCFSolver supports only {supported}. "
                "Use `EDSolver` for exact diagonalization."
            )
        super().__init__(*args, backend=backend or self.default_backend, method=method, **kwargs)


PYSCF_solver = PySCFSolver
