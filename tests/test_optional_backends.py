import pytest

from mbkit import EDSolver
from mbkit import solver as solver_pkg


def test_solver_package_exports_ed_solver():
    assert EDSolver is not None
    assert solver_pkg.EDSolver is EDSolver


def test_dmrg_backend_has_clear_optional_dependency_error():
    with pytest.raises(ImportError, match=r"mbkit\[dmrg\]"):
        solver_pkg.DMRGSolver()


def test_pyscf_backend_has_clear_optional_dependency_error():
    with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
        solver_pkg.PySCFSolver()
