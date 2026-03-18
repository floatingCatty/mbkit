import pytest

from mbkit import EDSolver
from mbkit import solver as solver_pkg


def test_solver_package_exports_ed_solver():
    assert EDSolver is not None
    assert solver_pkg.EDSolver is EDSolver


def test_dmrg_backend_has_clear_optional_dependency_error():
    with pytest.raises(ImportError, match=r"mbkit\[dmrg\]"):
        solver_pkg.DMRGSolver(n_int=1, n_noint=0, n_elec=2, nspin=1)


def test_pyscf_backend_has_clear_optional_dependency_error():
    with pytest.raises(ImportError, match=r"mbkit\[pyscf\]"):
        solver_pkg.PySCFSolver(norb=1, naux=0, nspin=1)


def test_nqs_backend_is_available_when_extra_is_installed():
    try:
        nqs_cls = solver_pkg.NQSSolver
    except ImportError:
        pytest.skip("NQS extra is not installed in this environment.")
    assert nqs_cls is not None
