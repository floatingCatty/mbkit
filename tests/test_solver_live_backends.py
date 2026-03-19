import numpy as np
import pytest

from mbkit import DMRGSolver, EDSolver, ElectronicSpace, FCISolver, LineLattice, MP2Solver, PySCFSolver, UHFSolver
from mbkit.solver.dmrg_solver import _PYBLOCK2_IMPORT_ERROR
from mbkit.solver.pyscf_solver import _PYSCF_IMPORT_ERROR
from mbkit.solver.backends.quimb_dmrg import _QUIMB_IMPORT_ERROR
from mbkit.solver.backends.tenpy_dmrg import _TENPY_IMPORT_ERROR


def test_quspin_ed_solver_uses_native_quspin_eigensolve_and_expectations():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    solver = EDSolver().solve(hamiltonian, n_particles=[(1, 1)])
    backend = solver.unwrap_backend()

    assert backend.quspin_operator is not None
    assert backend.solve_mode == "quspin_eigsh"
    assert np.isclose(np.trace(solver.rdm1()), 2.0, atol=1e-8)
    assert np.isfinite(np.real(solver.s2()))


def test_pyscf_solver_runs_live_one_site_hubbard_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    solver = PySCFSolver()
    solver.solve(hamiltonian, n_particles=(1, 1))

    assert np.isclose(np.real(solver.energy()), 2.0)
    assert solver.rdm1().shape == (2, 2)


def test_method_based_fci_solver_runs_live_one_site_hubbard_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    solver = FCISolver().solve(hamiltonian, n_particles=(1, 1))

    assert np.isclose(np.real(solver.energy()), 2.0)
    assert solver.rdm1().shape == (2, 2)


def test_pyscf_solver_routes_uhf_and_mp2_methods_to_reference_backend():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    uhf_solver = PySCFSolver(method="uhf")
    mp2_solver = PySCFSolver(method="mp2")

    assert uhf_solver.backend_name == "pyscf_reference"
    assert mp2_solver.backend_name == "pyscf_reference"


def test_method_based_uhf_and_mp2_solvers_route_to_reference_backend():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    uhf_solver = UHFSolver()
    mp2_solver = MP2Solver()

    assert uhf_solver.backend_name == "pyscf_reference"
    assert mp2_solver.backend_name == "pyscf_reference"
    assert uhf_solver.unwrap_backend().method == "uhf"
    assert mp2_solver.unwrap_backend().method == "mp2"


def test_pyscf_solver_uhf_and_mp2_run_live_two_site_hubbard_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    uhf_solver = PySCFSolver(method="uhf").solve(hamiltonian, n_particles=(1, 1))
    mp2_solver = PySCFSolver(method="mp2").solve(hamiltonian, n_particles=(1, 1))

    assert np.isfinite(np.real(uhf_solver.energy()))
    assert np.isfinite(np.real(mp2_solver.energy()))
    assert mp2_solver.energy() <= uhf_solver.energy() + 1e-8
    assert np.isclose(np.trace(uhf_solver.rdm1()), 2.0, atol=1e-8)
    assert np.isclose(np.trace(mp2_solver.rdm1(kind="unrelaxed")), 2.0, atol=1e-8)
    assert uhf_solver.rdm1().shape == (space.num_spin_orbitals, space.num_spin_orbitals)
    assert mp2_solver.rdm1(kind="unrelaxed").shape == (space.num_spin_orbitals, space.num_spin_orbitals)


def test_method_based_uhf_and_mp2_solvers_run_live_two_site_hubbard_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    uhf_solver = UHFSolver().solve(hamiltonian, n_particles=(1, 1))
    mp2_solver = MP2Solver().solve(hamiltonian, n_particles=(1, 1))

    assert np.isfinite(np.real(uhf_solver.energy()))
    assert np.isfinite(np.real(mp2_solver.energy()))
    assert mp2_solver.energy() <= uhf_solver.energy() + 1e-8
    assert np.isclose(np.trace(uhf_solver.rdm1()), 2.0, atol=1e-8)
    assert np.isclose(np.trace(mp2_solver.rdm1(kind="unrelaxed")), 2.0, atol=1e-8)


def test_method_based_uhf_and_mp2_match_known_noninteracting_reference_values():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    uhf_solver = UHFSolver().solve(hamiltonian, n_particles=(1, 1))
    mp2_solver = MP2Solver().solve(hamiltonian, n_particles=(1, 1))

    expected_rdm1 = np.array(
        [
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
        ]
    )

    assert np.isclose(np.real(uhf_solver.energy()), -2.0, atol=1e-8)
    assert np.isclose(np.real(mp2_solver.energy()), -2.0, atol=1e-8)
    assert np.allclose(uhf_solver.rdm1(), expected_rdm1, atol=1e-8)
    assert np.allclose(mp2_solver.rdm1(kind="unrelaxed"), expected_rdm1, atol=1e-8)
    assert np.allclose(uhf_solver.docc(), [0.25, 0.25], atol=1e-8)
    assert np.isclose(np.real(uhf_solver.s2()), 0.0, atol=1e-8)
    assert np.isclose(np.real(mp2_solver.s2(kind="reference")), 0.0, atol=1e-8)


def test_method_based_mp2_requires_explicit_property_semantics():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    solver = MP2Solver().solve(hamiltonian, n_particles=(1, 1))

    with pytest.raises(ValueError, match="kind='unrelaxed'"):
        solver.rdm1()
    with pytest.raises(NotImplementedError, match="two-body"):
        solver.docc()
    with pytest.raises(ValueError, match="kind='reference'"):
        solver.s2()


def test_method_based_uhf_solver_raises_on_unconverged_reference():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    with pytest.raises(RuntimeError, match="did not converge"):
        UHFSolver(max_cycle=0).solve(hamiltonian, n_particles=(1, 1))


def test_pyscf_solver_matches_ed_on_one_site_two_orbital_tensor_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a", "b"])
    tensor = np.zeros((2, 2, 2, 2), dtype=float)
    tensor[0, 0, 0, 0] = 4.0
    tensor[1, 1, 1, 1] = 3.0
    tensor[0, 1, 1, 0] = 1.5
    tensor[1, 0, 0, 1] = 1.5
    tensor[0, 1, 0, 1] = 0.7
    tensor[1, 0, 1, 0] = 0.7

    hamiltonian = space.crystal_field_term(values={"a": -1.0, "b": 0.5}, spin="both")
    hamiltonian += space.local_interaction_tensor_term(tensor=tensor, orbitals=("a", "b"), strength=0.5)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(2, 0)])
    pyscf_solver = PySCFSolver().solve(hamiltonian, n_particles=(2, 0))

    assert np.isclose(np.real(pyscf_solver.energy()), np.real(ed_solver.energy()), atol=1e-8)
    assert np.allclose(pyscf_solver.rdm1(), ed_solver.rdm1(), atol=1e-8)


def test_method_based_fci_solver_matches_ed_on_one_site_two_orbital_tensor_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a", "b"])
    tensor = np.zeros((2, 2, 2, 2), dtype=float)
    tensor[0, 0, 0, 0] = 4.0
    tensor[1, 1, 1, 1] = 3.0
    tensor[0, 1, 1, 0] = 1.5
    tensor[1, 0, 0, 1] = 1.5
    tensor[0, 1, 0, 1] = 0.7
    tensor[1, 0, 1, 0] = 0.7

    hamiltonian = space.crystal_field_term(values={"a": -1.0, "b": 0.5}, spin="both")
    hamiltonian += space.local_interaction_tensor_term(tensor=tensor, orbitals=("a", "b"), strength=0.5)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(2, 0)])
    fci_solver = FCISolver().solve(hamiltonian, n_particles=(2, 0))

    assert np.isclose(np.real(fci_solver.energy()), np.real(ed_solver.energy()), atol=1e-8)
    assert np.allclose(fci_solver.rdm1(), ed_solver.rdm1(), atol=1e-8)


def test_block2_dmrg_solver_runs_live_one_site_hubbard_problem():
    if _PYBLOCK2_IMPORT_ERROR is not None:
        pytest.skip("block2 / pyblock2 is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    solver = DMRGSolver(
        bond_dim=16,
        bond_mul=1,
        n_sweep=2,
        nupdate=1,
        iprint=0,
        scratch_dir="/tmp/mbkit_block2_smoke",
    )
    solver.solve(hamiltonian, n_particles=(1, 1))

    assert np.isclose(np.real(solver.energy()), 2.0, atol=1e-6)


def test_quimb_dmrg_solver_runs_live_one_site_hubbard_problem():
    if _QUIMB_IMPORT_ERROR is not None:
        pytest.skip("quimb is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    solver = DMRGSolver(
        backend="quimb",
        bond_dims=[8, 16],
        cutoffs=1e-9,
        solve_tol=1e-8,
        max_sweeps=4,
        verbosity=0,
    )
    solver.solve(hamiltonian, n_particles=(1, 1))

    assert np.isclose(np.real(solver.energy()), 2.0, atol=1e-6)
    assert solver.rdm1().shape == (2, 2)


def test_quimb_dmrg_solver_uses_direct_mpo_path_without_sector_penalty():
    if _QUIMB_IMPORT_ERROR is not None:
        pytest.skip("quimb is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.crystal_field_term(values=1.0, spin="both")

    solver = DMRGSolver(
        backend="quimb",
        bond_dims=[8, 16],
        cutoffs=1e-9,
        solve_tol=1e-8,
        max_sweeps=4,
        verbosity=0,
    )
    solver.solve(hamiltonian)

    assert solver.unwrap_backend().mpo is not None
    assert solver.unwrap_backend().dmrg is not None
    assert solver.unwrap_backend().solve_mode == "local_terms"
    assert np.isclose(np.real(solver.energy()), 0.0, atol=1e-6)


def test_quimb_dmrg_solver_uses_penalty_mpo_and_native_state_with_particle_constraint():
    if _QUIMB_IMPORT_ERROR is not None:
        pytest.skip("quimb is not installed.")

    space = ElectronicSpace(LineLattice(3, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=-1.0, U=2.0)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(1, 1)])
    solver = DMRGSolver(
        backend="quimb",
        bond_dims=[16, 32],
        cutoffs=1e-9,
        solve_tol=1e-8,
        max_sweeps=6,
        verbosity=0,
    )
    solver.solve(hamiltonian, n_particles=(1, 1))

    assert solver.unwrap_backend().solve_mode == "local_terms_penalty_mpo"
    assert solver.unwrap_backend().state is not None
    assert solver.unwrap_backend().state_vector is None
    assert np.isclose(np.real(solver.energy()), np.real(ed_solver.energy()), atol=1e-6)
    assert solver.rdm1().shape == (space.num_spin_orbitals, space.num_spin_orbitals)


def test_tenpy_dmrg_solver_matches_ed_on_three_site_hubbard_problem():
    if _TENPY_IMPORT_ERROR is not None:
        pytest.skip("TeNPy is not installed.")

    space = ElectronicSpace(LineLattice(3, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=-1.0, U=2.0)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(1, 1)])
    dmrg_solver = DMRGSolver(
        backend="tenpy",
        chi_max=32,
        svd_min=1e-12,
        max_sweeps=6,
        max_E_err=1e-8,
    ).solve(hamiltonian, n_particles=(1, 1))

    assert dmrg_solver.unwrap_backend().psi is not None
    assert dmrg_solver.unwrap_backend().mpo is not None
    assert dmrg_solver.unwrap_backend().mpo_mode == "local_terms"
    assert np.isclose(np.real(dmrg_solver.energy()), np.real(ed_solver.energy()), atol=1e-6)
    assert dmrg_solver.rdm1().shape == (space.num_spin_orbitals, space.num_spin_orbitals)


def test_tenpy_dmrg_solver_uses_local_term_mpo_without_sector_penalty():
    if _TENPY_IMPORT_ERROR is not None:
        pytest.skip("TeNPy is not installed.")

    space = ElectronicSpace(LineLattice(3, boundary="open"), orbitals=["a"])
    hamiltonian = space.crystal_field_term(values=1.0, spin="both")

    solver = DMRGSolver(
        backend="tenpy",
        chi_max=16,
        svd_min=1e-12,
        max_sweeps=4,
        max_E_err=1e-8,
    )
    solver.solve(hamiltonian)

    assert solver.unwrap_backend().mpo is not None
    assert solver.unwrap_backend().psi is not None
    assert solver.unwrap_backend().mpo_mode == "local_terms"
    assert np.isclose(np.real(solver.energy()), 0.0, atol=1e-6)
