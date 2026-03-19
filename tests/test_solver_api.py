import numpy as np
import pytest

from mbkit import DMRGSolver, EDSolver, ElectronicSpace, FCISolver, LineLattice, MP2Solver, PySCFSolver, UHFSolver
from mbkit.solver.base import SolverFacade
from mbkit.solver.pyscf_solver import _PYSCF_IMPORT_ERROR


def test_ed_solver_exposes_generic_expect_and_diagnostics():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    solver = EDSolver().solve(hamiltonian, n_particles=[(1, 1)])
    n0_up = space.number(0, orbital="a", spin="up")

    props = solver.available_properties()
    assert "energy" in props
    assert "expect" in props
    assert "expect_value" in props
    assert "rdm1" in props
    assert "diagnostics" in props

    assert np.isclose(np.real(solver.expect_value(space.spin_squared_term())), np.real(solver.s2()))
    assert np.isclose(np.real(solver.expect_value(n0_up)), np.real(solver.rdm1()[0, 0]))

    stats = solver.expect(n0_up, stats=True)
    assert np.isclose(np.real(stats["mean"]), np.real(solver.expect_value(n0_up)))
    assert stats["stderr"] == 0.0
    assert stats["kind"] == "deterministic"

    diagnostics = solver.diagnostics()
    assert diagnostics["backend"] == "quspin"
    assert diagnostics["solve_mode"] == "quspin_eigsh"
    assert diagnostics["basis_size"] > 0


def test_solver_native_returns_backend_instance():
    solver = EDSolver()
    assert solver.native is solver.unwrap_backend()


@pytest.mark.parametrize(
    "solver_cls",
    [EDSolver, DMRGSolver, PySCFSolver, FCISolver, UHFSolver, MP2Solver],
)
def test_public_solver_facades_inherit_shared_runtime_helpers(solver_cls):
    assert issubclass(solver_cls, SolverFacade)
    assert solver_cls.expect is SolverFacade.expect
    assert solver_cls.expect_value is SolverFacade.expect_value
    assert solver_cls.diagnostics is SolverFacade.diagnostics
    assert solver_cls.available_properties is SolverFacade.available_properties


def test_expect_value_requires_solve_before_eval():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])

    with pytest.raises(RuntimeError, match="Call solve"):
        EDSolver().expect_value(space.spin_squared_term())


def test_pyscf_exact_solver_exposes_generic_expect_when_available():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    solver = PySCFSolver().solve(hamiltonian, n_particles=(1, 1))
    assert "expect" in solver.available_properties()
    assert np.isclose(np.real(solver.expect_value(space.spin_squared_term())), np.real(solver.s2()), atol=1e-8)


def test_method_based_qc_solvers_route_to_expected_backends():
    fci_solver = FCISolver()
    uhf_solver = UHFSolver()
    mp2_solver = MP2Solver()

    assert fci_solver.backend_name == "pyscf_fci"
    assert uhf_solver.backend_name == "pyscf_reference"
    assert mp2_solver.backend_name == "pyscf_reference"
    assert uhf_solver.unwrap_backend().method == "uhf"
    assert mp2_solver.unwrap_backend().method == "mp2"


def test_pyscf_reference_solver_reports_method_specific_diagnostics():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    solver = PySCFSolver(method="uhf").solve(hamiltonian, n_particles=(1, 1))
    assert "expect" not in solver.available_properties()

    diagnostics = solver.diagnostics()
    assert diagnostics["backend"] == "pyscf_reference"
    assert diagnostics["method"] == "uhf"
    assert diagnostics["reference_converged"] in (True, False)


def test_mp2_solver_reports_narrow_property_semantics():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    solver = MP2Solver().solve(hamiltonian, n_particles=(1, 1))

    assert "docc" not in solver.available_properties()
    diagnostics = solver.diagnostics()
    assert diagnostics["method"] == "mp2"
    assert diagnostics["rdm1_kinds"] == ("reference", "unrelaxed")
    assert diagnostics["s2_kinds"] == ("reference",)
