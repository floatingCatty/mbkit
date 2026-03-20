import numpy as np
import pytest

from mbkit import CCSDSolver, CCSDTSolver, DMRGSolver, EDSolver, ElectronicSpace, LineLattice, MP2Solver, NQSSolver, PySCFSolver, UHFSolver
from mbkit.solver.dmrg_solver import _PYBLOCK2_IMPORT_ERROR
from mbkit.solver.backends.quantax_nqs import _QUSPIN_IMPORT_ERROR
from mbkit.solver.nqs_solver import _QUANTAX_IMPORT_ERROR
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


def test_quantax_nqs_solver_matches_ed_on_one_site_hubbard_problem():
    if _QUANTAX_IMPORT_ERROR is not None or _QUSPIN_IMPORT_ERROR is not None:
        pytest.skip("Quantax exact-sampling stack is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=(1, 1))
    nqs_solver = NQSSolver(
        model="general_det",
        optimizer="sr",
        sampler="exact",
        nsamples=128,
        n_steps=2,
        step_size=0.1,
        seed=7,
    ).solve(hamiltonian, n_particles=(1, 1))

    number = space.number(0, orbital="a", spin="up")
    stats = nqs_solver.expect(number, stats=True)

    assert np.isclose(np.real(nqs_solver.energy()), np.real(ed_solver.energy()), atol=1e-8)
    assert np.allclose(nqs_solver.rdm1(), ed_solver.rdm1(), atol=1e-8)
    assert np.allclose(nqs_solver.docc(), ed_solver.docc(), atol=1e-8)
    assert np.isclose(np.real(nqs_solver.s2()), np.real(ed_solver.s2()), atol=1e-8)
    assert np.isclose(np.real(nqs_solver.expect_value(number)), np.real(ed_solver.expect_value(number)), atol=1e-8)
    assert stats["kind"] == "stochastic"
    assert stats["backend"] == "quantax"
    assert stats["n_samples"] == 128
    assert np.isfinite(stats["stderr"])


def test_quantax_nqs_solver_matches_ed_on_two_site_noninteracting_problem():
    if _QUANTAX_IMPORT_ERROR is not None or _QUSPIN_IMPORT_ERROR is not None:
        pytest.skip("Quantax exact-sampling stack is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=(1, 1))
    nqs_solver = NQSSolver(
        model="general_det",
        optimizer="sr",
        sampler="exact",
        nsamples=256,
        n_steps=25,
        step_size=0.2,
        seed=13,
    ).solve(hamiltonian, n_particles=(1, 1))

    number = space.number(0, orbital="a", spin="up")

    assert np.isclose(np.real(nqs_solver.energy()), np.real(ed_solver.energy()), atol=5e-3)
    assert np.allclose(nqs_solver.rdm1(), ed_solver.rdm1(), atol=4e-2)
    assert np.allclose(nqs_solver.docc(), ed_solver.docc(), atol=4e-2)
    assert np.isclose(np.real(nqs_solver.s2()), np.real(ed_solver.s2()), atol=4e-2)
    assert np.isclose(np.real(nqs_solver.expect_value(number)), np.real(ed_solver.expect_value(number)), atol=4e-2)


def test_quantax_nqs_solver_supports_total_particle_conservation_with_spin_mixing():
    if _QUANTAX_IMPORT_ERROR is not None or _QUSPIN_IMPORT_ERROR is not None:
        pytest.skip("Quantax exact-sampling stack is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.spin_plus(0, orbital="a") + space.spin_minus(0, orbital="a")

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(1, 0), (0, 1)])
    nqs_solver = NQSSolver(
        model="general_det",
        optimizer="sr",
        sampler="exact",
        nsamples=128,
        n_steps=12,
        step_size=0.2,
        seed=11,
    ).solve(hamiltonian, n_particles=1)

    assert nqs_solver.diagnostics()["particle_sector_kind"] == "total"
    assert np.isclose(np.real(nqs_solver.energy()), np.real(ed_solver.energy()), atol=5e-3)
    assert np.allclose(nqs_solver.rdm1(), ed_solver.rdm1(), atol=4e-2)
    assert np.isclose(
        np.real(nqs_solver.expect_value(space.number(0, orbital="a", spin="up"))),
        np.real(ed_solver.expect_value(space.number(0, orbital="a", spin="up"))),
        atol=4e-2,
    )


def test_quantax_nqs_solver_default_stochastic_path_reports_diagnostics():
    if _QUANTAX_IMPORT_ERROR is not None:
        pytest.skip("Quantax is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    solver = NQSSolver(
        nsamples=128,
        n_steps=4,
        step_size=1e-2,
        seed=3,
    ).solve(hamiltonian, n_particles=(1, 1))

    diagnostics = solver.diagnostics()
    stats = solver.expect(space.number(0, orbital="a", spin="up"), stats=True)

    assert diagnostics["backend"] == "quantax"
    assert diagnostics["model"] == "pf_backflow"
    assert diagnostics["optimizer"] == "march"
    assert diagnostics["linear_solver"] == "auto_pinv_eig"
    assert diagnostics["sampler"] in {"exact", "particle_hop"}
    assert diagnostics["quantax_source"] in {"installed", "extern"}
    assert len(diagnostics["training_history"]["energy"]) == 4
    assert len(diagnostics["training_history"]["variance"]) == 4
    assert np.isfinite(np.real(solver.energy()))
    assert np.isfinite(stats["stderr"])


def test_pyscf_solver_runs_live_one_site_hubbard_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=0.0, U=2.0)

    solver = PySCFSolver(method="uhf")
    solver.solve(hamiltonian, n_particles=(1, 1))

    assert np.isclose(np.real(solver.energy()), 2.0)
    assert solver.rdm1().shape == (2, 2)


def test_pyscf_solver_routes_uhf_mp2_and_cc_methods_to_reference_backend():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    default_solver = PySCFSolver()
    uhf_solver = PySCFSolver(method="uhf")
    mp2_solver = PySCFSolver(method="mp2")
    ccsd_solver = PySCFSolver(method="ccsd")
    ccsdt_solver = PySCFSolver(method="ccsd(t)")

    assert default_solver.backend_name == "pyscf_reference"
    assert uhf_solver.backend_name == "pyscf_reference"
    assert mp2_solver.backend_name == "pyscf_reference"
    assert ccsd_solver.backend_name == "pyscf_reference"
    assert ccsdt_solver.backend_name == "pyscf_reference"
    assert default_solver.unwrap_backend().method == "uhf"
    assert ccsd_solver.unwrap_backend().method == "ccsd"
    assert ccsdt_solver.unwrap_backend().method == "ccsd(t)"


def test_method_based_uhf_mp2_and_cc_solvers_route_to_reference_backend():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    uhf_solver = UHFSolver()
    mp2_solver = MP2Solver()
    ccsd_solver = CCSDSolver()
    ccsdt_solver = CCSDTSolver()

    assert uhf_solver.backend_name == "pyscf_reference"
    assert mp2_solver.backend_name == "pyscf_reference"
    assert ccsd_solver.backend_name == "pyscf_reference"
    assert ccsdt_solver.backend_name == "pyscf_reference"
    assert uhf_solver.unwrap_backend().method == "uhf"
    assert mp2_solver.unwrap_backend().method == "mp2"
    assert ccsd_solver.unwrap_backend().method == "ccsd"
    assert ccsdt_solver.unwrap_backend().method == "ccsd(t)"


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


def test_method_based_cc_solvers_match_known_noninteracting_reference_values():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    ccsd_solver = CCSDSolver().solve(hamiltonian, n_particles=(1, 1))
    ccsdt_solver = CCSDTSolver().solve(hamiltonian, n_particles=(1, 1))

    expected_rdm1 = np.array(
        [
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
        ]
    )

    assert np.isclose(np.real(ccsd_solver.energy()), -2.0, atol=1e-8)
    assert np.isclose(np.real(ccsdt_solver.energy()), -2.0, atol=1e-8)
    assert np.allclose(ccsd_solver.rdm1(kind="unrelaxed"), expected_rdm1, atol=1e-8)
    assert np.allclose(ccsdt_solver.rdm1(kind="reference"), expected_rdm1, atol=1e-8)
    assert np.isclose(np.real(ccsd_solver.s2(kind="reference")), 0.0, atol=1e-8)
    assert np.isclose(np.real(ccsdt_solver.s2(kind="reference")), 0.0, atol=1e-8)
    assert np.isclose(ccsdt_solver.diagnostics()["triples_correction"], 0.0, atol=1e-12)


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


def test_method_based_ccsd_requires_explicit_property_semantics():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    solver = CCSDSolver().solve(hamiltonian, n_particles=(1, 1))

    with pytest.raises(ValueError, match="kind='unrelaxed'"):
        solver.rdm1()
    with pytest.raises(NotImplementedError, match="two-body"):
        solver.docc()
    with pytest.raises(ValueError, match="kind='reference'"):
        solver.s2()


def test_method_based_ccsdt_requires_explicit_property_semantics():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=0.0)

    solver = CCSDTSolver().solve(hamiltonian, n_particles=(1, 1))

    with pytest.raises(ValueError, match="kind='reference'"):
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


def test_method_based_cc_solvers_match_ed_on_four_site_two_electron_hubbard_problem():
    if _PYSCF_IMPORT_ERROR is not None:
        pytest.skip("PySCF is not installed.")

    space = ElectronicSpace(LineLattice(4, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    ed_solver = EDSolver().solve(hamiltonian, n_electrons=2)
    ccsd_solver = CCSDSolver().solve(hamiltonian, n_electrons=2)
    ccsdt_solver = CCSDTSolver().solve(hamiltonian, n_electrons=2)

    assert np.isclose(np.real(ccsd_solver.energy()), np.real(ed_solver.energy()), atol=1e-7)
    assert np.isclose(np.real(ccsdt_solver.energy()), np.real(ed_solver.energy()), atol=1e-7)
    assert np.isclose(np.real(ccsdt_solver.energy()), np.real(ccsd_solver.energy()), atol=1e-10)
    assert np.isclose(np.trace(ccsd_solver.rdm1(kind="unrelaxed")), 2.0, atol=1e-8)
    assert np.isclose(np.trace(ccsdt_solver.rdm1(kind="reference")), 2.0, atol=1e-8)
    assert np.isclose(ccsdt_solver.diagnostics()["triples_correction"], 0.0, atol=1e-12)


def test_pyscf_solver_matches_uhf_on_one_site_two_orbital_tensor_problem():
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

    pyscf_solver = PySCFSolver().solve(hamiltonian, n_particles=(2, 0))
    uhf_solver = UHFSolver().solve(hamiltonian, n_particles=(2, 0))

    assert np.isclose(np.real(pyscf_solver.energy()), np.real(uhf_solver.energy()), atol=1e-8)
    assert np.allclose(pyscf_solver.rdm1(), uhf_solver.rdm1(), atol=1e-8)


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


@pytest.mark.parametrize(
    ("symmetry", "extra_kwargs", "resolved_symmetry"),
    [
        ("sgf", {}, "sgf"),
        ("sz", {"symmetry": "sz"}, "sz"),
        ("su2", {"symmetry": "su2", "target_spin": 0}, "su2"),
        ("auto", {"symmetry": "auto"}, "su2"),
    ],
)
def test_block2_dmrg_solver_matches_ed_on_four_site_hubbard_problem(symmetry, extra_kwargs, resolved_symmetry):
    if _PYBLOCK2_IMPORT_ERROR is not None:
        pytest.skip("block2 / pyblock2 is not installed.")

    space = ElectronicSpace(LineLattice(4, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(2, 2)])
    dmrg_solver = DMRGSolver(
        bond_dim=64,
        bond_mul=2,
        n_sweep=10,
        nupdate=2,
        eig_cutoff=1e-8,
        iprint=0,
        scratch_dir=f"/tmp/mbkit_block2_match_ed_{symmetry}",
        **extra_kwargs,
    ).solve(hamiltonian, n_particles=(2, 2))

    assert np.isclose(np.real(dmrg_solver.energy()), np.real(ed_solver.energy()), atol=1e-6)
    assert dmrg_solver.diagnostics()["resolved_symmetry"] == resolved_symmetry
    assert np.isclose(np.trace(dmrg_solver.rdm1()), 4.0, atol=1e-6)


def test_block2_dmrg_solver_su2_nonzero_spin_requires_spin_traced_rdm1():
    if _PYBLOCK2_IMPORT_ERROR is not None:
        pytest.skip("block2 / pyblock2 is not installed.")

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    ed_solver = EDSolver().solve(hamiltonian, n_particles=[(2, 0)])
    dmrg_solver = DMRGSolver(
        symmetry="su2",
        target_spin=2,
        bond_dim=32,
        bond_mul=1,
        n_sweep=6,
        nupdate=2,
        eig_cutoff=1e-8,
        iprint=0,
        scratch_dir="/tmp/mbkit_block2_su2_triplet",
    ).solve(hamiltonian, n_particles=(2, 0))

    assert np.isclose(np.real(dmrg_solver.energy()), np.real(ed_solver.energy()), atol=1e-6)
    assert dmrg_solver.diagnostics()["target_spin_twos"] == 2
    assert dmrg_solver.rdm1(kind="spin_traced").shape == (space.num_spatial_orbitals, space.num_spatial_orbitals)
    with pytest.raises(NotImplementedError, match="spin_traced"):
        dmrg_solver.rdm1()


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
