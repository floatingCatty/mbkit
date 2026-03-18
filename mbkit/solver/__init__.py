"""Public solver exports for mbkit."""

from importlib import import_module

from .ed_solver import ED_solver, EDSolver

__all__ = [
    "EDSolver",
    "ED_solver",
    "DMRGSolver",
    "DMRG_solver",
    "NQSSolver",
    "NQS_solver",
    "PySCFSolver",
    "PYSCF_solver",
]

_OPTIONAL_EXPORTS = {
    "DMRGSolver": (".dmrg_solver", "DMRGSolver"),
    "DMRG_solver": (".dmrg_solver", "DMRG_solver"),
    "NQSSolver": (".nqs_solver", "NQSSolver"),
    "NQS_solver": (".nqs_solver", "NQS_solver"),
    "PySCFSolver": (".pyscf_solver", "PySCFSolver"),
    "PYSCF_solver": (".pyscf_solver", "PYSCF_solver"),
}


def __getattr__(name):
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
