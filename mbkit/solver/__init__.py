"""Public solver exports for `mbkit`.

All exported solver classes are public façades built on `SolverFacade`, so
they share the same basic runtime workflow:

1. instantiate a solver class,
2. call `solve(...)`,
3. query shared helpers such as `expect()`, `expect_value()`,
   `diagnostics()`, and `available_properties()`, plus any solver-specific
   observable methods exposed by the selected backend.
"""

from .dmrg_solver import DMRGSolver, DMRG_solver
from .ed_solver import EDSolver, ED_solver
from .ccsd_solver import CCSDSolver, CCSD_solver
from .ccsdt_solver import CCSDTSolver, CCSDT_solver
from .mp2_solver import MP2Solver, MP2_solver
from .nqs_solver import NQSSolver, NQS_solver
from .pyscf_solver import PYSCF_solver, PySCFSolver
from .uhf_solver import UHFSolver, UHF_solver
from .registry import available_solver_backends, get_backend_spec, get_solver_backend_class

__all__ = [
    "EDSolver",
    "ED_solver",
    "DMRGSolver",
    "DMRG_solver",
    "UHFSolver",
    "UHF_solver",
    "CCSDSolver",
    "CCSD_solver",
    "CCSDTSolver",
    "CCSDT_solver",
    "MP2Solver",
    "MP2_solver",
    "NQSSolver",
    "NQS_solver",
    "PySCFSolver",
    "PYSCF_solver",
    "available_solver_backends",
    "get_backend_spec",
    "get_solver_backend_class",
]
