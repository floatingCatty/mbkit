import numpy as np
import pytest

from mbkit import CCSDSolver, CCSDTSolver, DMRGSolver, EDSolver, ElectronicSpace, LineLattice, MP2Solver, NQSSolver, PySCFSolver, UHFSolver
from mbkit.solver.base import SolverFacade
from mbkit.solver.nqs_solver import _QUANTAX_IMPORT_ERROR
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
    [EDSolver, DMRGSolver, NQSSolver, PySCFSolver, UHFSolver, MP2Solver, CCSDSolver, CCSDTSolver],
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


def test_nqs_solver_requires_quantax_dependencies_or_instantiates_cleanly():
    if _QUANTAX_IMPORT_ERROR is None:
        assert NQSSolver().backend_name == "quantax"
    else:
        with pytest.raises(ImportError, match=r"mbkit\[nqs\]"):
            NQSSolver()


def test_nqs_solver_requires_total_particle_sector_for_spin_mixing_hamiltonians():
    if _QUANTAX_IMPORT_ERROR is not None:
        pytest.skip("Quantax is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.spin_plus(0, orbital="a") + space.spin_minus(0, orbital="a")

    with pytest.raises(ValueError, match="total particle-count sector"):
        NQSSolver(model="general_det", sampler="exact", n_steps=0, nsamples=16).solve(
            hamiltonian,
            n_particles=(1, 0),
        )


def test_pyscf_solver_rejects_removed_fci_method():
    with pytest.raises(ValueError, match="EDSolver"):
        PySCFSolver(method="fci")


def test_method_based_qc_solvers_route_to_expected_backends():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    pyscf_solver = PySCFSolver()
    uhf_solver = UHFSolver()
    mp2_solver = MP2Solver()
    ccsd_solver = CCSDSolver()
    ccsdt_solver = CCSDTSolver()

    assert pyscf_solver.backend_name == "pyscf_reference"
    assert uhf_solver.backend_name == "pyscf_reference"
    assert mp2_solver.backend_name == "pyscf_reference"
    assert ccsd_solver.backend_name == "pyscf_reference"
    assert ccsdt_solver.backend_name == "pyscf_reference"
    assert pyscf_solver.unwrap_backend().method == "uhf"
    assert uhf_solver.unwrap_backend().method == "uhf"
    assert mp2_solver.unwrap_backend().method == "mp2"
    assert ccsd_solver.unwrap_backend().method == "ccsd"
    assert ccsdt_solver.unwrap_backend().method == "ccsd(t)"


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


def test_cc_solvers_report_method_specific_diagnostics():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    ccsd_solver = CCSDSolver().solve(hamiltonian, n_particles=(1, 1))
    ccsdt_solver = CCSDTSolver().solve(hamiltonian, n_particles=(1, 1))

    assert "docc" not in ccsd_solver.available_properties()
    assert "docc" not in ccsdt_solver.available_properties()

    ccsd_diagnostics = ccsd_solver.diagnostics()
    assert ccsd_diagnostics["method"] == "ccsd"
    assert ccsd_diagnostics["rdm1_kinds"] == ("reference", "unrelaxed")
    assert ccsd_diagnostics["s2_kinds"] == ("reference",)
    assert ccsd_diagnostics["triples_correction"] is None

    ccsdt_diagnostics = ccsdt_solver.diagnostics()
    assert ccsdt_diagnostics["method"] == "ccsd(t)"
    assert ccsdt_diagnostics["rdm1_kinds"] == ("reference",)
    assert ccsdt_diagnostics["s2_kinds"] == ("reference",)
    assert np.isclose(ccsdt_diagnostics["triples_correction"], 0.0, atol=1e-12)
