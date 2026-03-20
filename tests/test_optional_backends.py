import pytest

from mbkit import CCSDSolver, CCSDTSolver, EDSolver, MP2Solver, NQSSolver, UHFSolver
from mbkit import solver as solver_pkg
from mbkit.solver.dmrg_solver import _PYBLOCK2_IMPORT_ERROR
from mbkit.solver.nqs_solver import _QUANTAX_IMPORT_ERROR
from mbkit.solver.pyscf_solver import _PYSCF_IMPORT_ERROR


def test_solver_package_exports_ed_solver():
    assert EDSolver is not None
    assert solver_pkg.EDSolver is EDSolver


def test_solver_package_exports_nqs_solver():
    assert NQSSolver is not None
    assert solver_pkg.NQSSolver is NQSSolver


def test_dmrg_backend_has_clear_optional_dependency_error():
    if _PYBLOCK2_IMPORT_ERROR is None:
        assert solver_pkg.DMRGSolver is not None
    else:
        with pytest.raises(ImportError, match=r"mbkit\[dmrg\]"):
            solver_pkg.DMRGSolver()


def test_nqs_backend_has_clear_optional_dependency_error():
    if _QUANTAX_IMPORT_ERROR is None:
        assert solver_pkg.NQSSolver is not None
    else:
        with pytest.raises(ImportError, match=r"mbkit\[nqs\]"):
            solver_pkg.NQSSolver()


def test_pyscf_backend_has_clear_optional_dependency_error():
    if _PYSCF_IMPORT_ERROR is None:
        assert solver_pkg.PySCFSolver is not None
        assert solver_pkg.UHFSolver is UHFSolver
        assert solver_pkg.MP2Solver is MP2Solver
        assert solver_pkg.CCSDSolver is CCSDSolver
        assert solver_pkg.CCSDTSolver is CCSDTSolver
    else:
        with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
            solver_pkg.PySCFSolver()
        with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
            solver_pkg.UHFSolver()
        with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
            solver_pkg.MP2Solver()
        with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
            solver_pkg.CCSDSolver()
        with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
            solver_pkg.CCSDTSolver()
