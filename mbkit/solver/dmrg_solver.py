"""Public DMRG solver façade."""

from __future__ import annotations

from .base import SolverFacade
from .backends.block2_dmrg import _PYBLOCK2_IMPORT_ERROR, _compile_direct_dmrg_hamiltonian


class DMRGSolver(SolverFacade):
    """DMRG solver façade.

    The default backend is the block2 implementation registered under the
    `dmrg` solver family.
    """

    solver_family = "dmrg"
    default_backend = "block2"


DMRG_solver = DMRGSolver
