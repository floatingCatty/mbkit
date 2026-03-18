import pytest

from mbkit import EDSolver
from mbkit import solver as solver_pkg
from mbkit.solver.dmrg_solver import _PYBLOCK2_IMPORT_ERROR
from mbkit.solver.pyscf_solver import _PYSCF_IMPORT_ERROR


def test_solver_package_exports_ed_solver():
    assert EDSolver is not None
    assert solver_pkg.EDSolver is EDSolver


def test_dmrg_backend_has_clear_optional_dependency_error():
    if _PYBLOCK2_IMPORT_ERROR is None:
        assert solver_pkg.DMRGSolver is not None
    else:
        with pytest.raises(ImportError, match=r"mbkit\[dmrg\]"):
            solver_pkg.DMRGSolver()


def test_pyscf_backend_has_clear_optional_dependency_error():
    if _PYSCF_IMPORT_ERROR is None:
        assert solver_pkg.PySCFSolver is not None
    else:
        with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
            solver_pkg.PySCFSolver()
