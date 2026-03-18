import numpy as np

from mbkit import (
    Bond,
    CubicLattice,
    ElectronicSpace,
    EDSolver,
    GeneralLattice,
    HoneycombLattice,
    KagomeLattice,
    LadderLattice,
    LineLattice,
    Operator,
    RectangularLattice,
    SquareLattice,
    TriangularLattice,
)
from mbkit.operator import to_electronic_integrals, to_qc_tensors, to_quspin_operator


def test_symbolic_operator_tracks_constants_and_metadata():
    space = ElectronicSpace(num_sites=2)
    hop = space.hopping(0, 1, spin="up", coeff=-1.0, plus_hc=True)
    shifted = hop + 2.5

    assert isinstance(shifted, Operator)
    assert np.isclose(shifted.constant, 2.5)
    assert hop.is_hermitian()
    assert hop.particle_number_changes() == frozenset({0})
    assert hop.spin_changes() == frozenset({0})


def test_line_lattice_hubbard_builder_compiles_to_backend_transforms():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    quspin_compiled = to_quspin_operator(hamiltonian)
    qc_compiled = to_qc_tensors(hamiltonian)

    assert hamiltonian.is_hermitian()
    assert quspin_compiled.constant_shift == 0.0
    assert quspin_compiled.static_list
    assert qc_compiled.h1e.shape == (4, 4)
    assert qc_compiled.g2e.shape == (4, 4, 4, 4)


def test_line_lattice_shell_selection_tracks_neighbor_order():
    space = ElectronicSpace(LineLattice(5, max_shell=3), orbitals=["a"])

    second_shell = space.select_bonds(shells=2)
    third_shell = space.select_bonds(shells=3)

    assert space.available_shells() == (1, 2, 3)
    assert len(second_shell) == 3
    assert len(third_shell) == 2
    assert {bond.shell for bond in second_shell} == {2}
    assert {bond.shell for bond in third_shell} == {3}


def test_slater_kanamori_builder_produces_quartic_terms():
    space = ElectronicSpace(LineLattice(1), orbitals=["dxz", "dyz"])
    hamiltonian = space.slater_kanamori(U=4.0, Up=2.6, J=0.7, Jp=0.7)

    factor_lengths = {len(term.factors) for term, _ in hamiltonian.iter_terms()}
    assert 4 in factor_lengths
    assert hamiltonian.particle_number_changes() == frozenset({0})


def test_general_lattice_extended_hubbard_accepts_bond_kinds():
    lattice = GeneralLattice(
        num_sites=4,
        bonds=[
            Bond(0, 1, kind="nn"),
            Bond(1, 2, kind="nn"),
            Bond(2, 3, kind="nn"),
            Bond(0, 2, kind="diag"),
        ],
    )
    space = ElectronicSpace(lattice, orbitals=["a"])
    hamiltonian = space.extended_hubbard(
        hopping={"nn": -1.0, "diag": -0.2},
        onsite_U=4.0,
        intersite_V={"diag": 0.5},
    )

    compiled = to_quspin_operator(hamiltonian)
    assert compiled.static_list


def test_square_lattice_hubbard_supports_kind_resolved_hoppings_and_mu():
    space = ElectronicSpace(SquareLattice(2, 2, boundary=("open", "open")), orbitals=["a"])
    hamiltonian = space.hubbard(
        t={"horizontal": 1.0, "vertical": 0.5},
        U=2.0,
        mu={0: 0.25, 3: -0.25},
    )

    compiled = to_quspin_operator(hamiltonian)
    assert compiled.static_list


def test_ladder_and_rectangular_lattices_expose_expected_bond_kinds():
    ladder_space = ElectronicSpace(LadderLattice(3, max_shell=2), orbitals=["a"])
    rectangular_space = ElectronicSpace(RectangularLattice(2, 3, max_shell=2), orbitals=["a"])

    ladder_hamiltonian = ladder_space.hubbard(t={"leg": 1.0, "rung": 0.3}, U=2.0)
    rectangular_hamiltonian = rectangular_space.hubbard(t={"horizontal": 1.0, "vertical": 0.5}, U=2.0)

    assert ladder_space.select_bonds("leg")
    assert ladder_space.select_bonds("rung")
    assert to_quspin_operator(ladder_hamiltonian).static_list
    assert to_quspin_operator(rectangular_hamiltonian).static_list


def test_triangular_and_cubic_lattices_support_directional_hoppings():
    triangular_space = ElectronicSpace(TriangularLattice(2, 2, max_shell=2), orbitals=["a"])
    cubic_space = ElectronicSpace(CubicLattice(2, 2, 2, max_shell=2), orbitals=["a"])

    triangular_hamiltonian = triangular_space.hubbard(
        t={"horizontal": 1.0, "up_right": 0.8, "down_right": 0.6},
        U=2.0,
    )
    cubic_hamiltonian = cubic_space.hubbard(t={"x": 1.0, "y": 0.5, "z": 0.2}, U=1.5)

    assert triangular_space.select_bonds("horizontal")
    assert triangular_space.select_bonds("up_right")
    assert triangular_space.select_bonds("down_right")
    assert to_quspin_operator(triangular_hamiltonian).static_list
    assert to_quspin_operator(cubic_hamiltonian).static_list


def test_square_shell_resolved_hubbard_uses_second_neighbor_bonds():
    space = ElectronicSpace(SquareLattice(2, 2, max_shell=2), orbitals=["a"])
    hamiltonian = space.hubbard(t={1: 0.0, 2: 0.25}, U=0.0)
    integrals = to_electronic_integrals(hamiltonian)
    summary = space.bond_summary()

    up_00 = space.mode_index(0, spin="up")
    up_11 = space.mode_index(3, spin="up")
    up_01 = space.mode_index(1, spin="up")

    assert summary[1] == {"horizontal": 2, "vertical": 2}
    assert summary[2] == {"diagonal": 2}
    assert space.select_bonds("diagonal", shells=2)
    assert np.isclose(integrals.one_body[up_00, up_11], -0.25)
    assert np.isclose(integrals.one_body[up_11, up_00], -0.25)
    assert np.isclose(integrals.one_body[up_00, up_01], 0.0)


def test_honeycomb_and_kagome_lattices_support_bond_resolved_models():
    honeycomb_space = ElectronicSpace(HoneycombLattice(2, 2, max_shell=2), orbitals=["a"])
    kagome_space = ElectronicSpace(KagomeLattice(2, 2, max_shell=2), orbitals=["a"])

    honeycomb_hamiltonian = honeycomb_space.extended_hubbard(
        hopping={"vertical": 1.0, "down_left": 0.7, "down_right": 0.7},
        onsite_U=3.0,
        intersite_V={"vertical": 0.2},
    )
    kagome_operator = kagome_space.hopping_term(coeff={"ab": -1.0, "ac": -0.5, "bc": -0.25}, plus_hc=True)
    kagome_integrals = to_electronic_integrals(kagome_operator)

    assert honeycomb_space.select_bonds("vertical")
    assert honeycomb_space.select_bonds("down_left")
    assert honeycomb_space.select_bonds("down_right")
    assert kagome_space.select_bonds("ab")
    assert kagome_space.select_bonds("ac")
    assert kagome_space.select_bonds("bc")
    assert to_quspin_operator(honeycomb_hamiltonian).static_list
    assert kagome_integrals.one_body.shape == (24, 24)


def test_general_lattice_can_filter_and_weight_by_explicit_shell_metadata():
    lattice = GeneralLattice(
        num_sites=3,
        bonds=[
            Bond(0, 1, kind="edge", shell=1),
            Bond(0, 2, kind="edge", shell=2),
        ],
    )
    space = ElectronicSpace(lattice, orbitals=["a"])
    operator = space.hopping_term(coeff={1: -1.0, 2: -0.4}, shells=2, plus_hc=True)
    integrals = to_electronic_integrals(operator)

    up_0 = space.mode_index(0, spin="up")
    up_1 = space.mode_index(1, spin="up")
    up_2 = space.mode_index(2, spin="up")

    assert len(space.select_bonds(shells=2)) == 1
    assert np.isclose(integrals.one_body[up_0, up_1], 0.0)
    assert np.isclose(integrals.one_body[up_0, up_2], -0.4)


def test_generic_hopping_builder_respects_bond_weight():
    lattice = GeneralLattice(num_sites=2, bonds=[Bond(0, 1, kind="nn", weight=2.0)])
    space = ElectronicSpace(lattice, orbitals=["a"])
    operator = space.hopping_term(coeff=-1.5, bonds="all", spin="both", plus_hc=True)
    integrals = to_electronic_integrals(operator)

    assert np.isclose(integrals.one_body[0, 2], -3.0)
    assert np.isclose(integrals.one_body[2, 0], -3.0)
    assert np.isclose(integrals.one_body[1, 3], -3.0)
    assert np.isclose(integrals.one_body[3, 1], -3.0)


def test_soc_builder_creates_spin_mixing_one_body_terms():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["dxz", "dyz"])
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 3] = 1.0j
    matrix[3, 0] = -1.0j
    operator = space.soc_term(strength=0.5, matrix=matrix)
    integrals = to_electronic_integrals(operator)

    assert np.isclose(integrals.one_body[0, 3], 0.5j)
    assert np.isclose(integrals.one_body[3, 0], -0.5j)


def test_local_matrix_term_matches_generic_one_body_basis_order():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["dxz", "dyz"])
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 3] = 1.0j
    matrix[3, 0] = -1.0j

    generic = space.local_matrix_term(matrix=matrix, strength=0.5)
    soc = space.soc_term(strength=0.5, matrix=matrix)
    integrals = to_electronic_integrals(generic)

    assert generic.equiv(soc)
    assert np.isclose(integrals.one_body[0, 3], 0.5j)
    assert np.isclose(integrals.one_body[3, 0], -0.5j)


def test_local_two_body_tensor_term_expands_spin_independent_onsite_tensor():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a", "b"])
    tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    tensor[0, 1, 1, 0] = 0.75

    operator = space.local_two_body_tensor_term(tensor=tensor)
    integrals = to_electronic_integrals(operator)
    manual = Operator.zero(space)
    for left_spin in space.spins:
        for right_spin in space.spins:
            manual += 0.75 * (
                space.create(0, orbital="a", spin=left_spin)
                @ space.create(0, orbital="b", spin=right_spin)
                @ space.destroy(0, orbital="b", spin=right_spin)
                @ space.destroy(0, orbital="a", spin=left_spin)
            )

    assert operator.particle_number_changes() == frozenset({0})
    assert operator.equiv(manual)
    assert np.count_nonzero(np.abs(integrals.two_body) > 1e-12) == 4


def test_observable_builders_match_simple_one_site_limits():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])

    spin_solver = EDSolver()
    spin_solver.solve(space.spin_squared_term(), nsites=space.num_spatial_orbitals, n_particles=[(1, 0)])
    assert np.isclose(np.real(spin_solver.energy()), 0.75)

    docc_solver = EDSolver()
    docc_solver.solve(space.double_occupancy_term(), nsites=space.num_spatial_orbitals, n_particles=[(1, 1)])
    assert np.isclose(np.real(docc_solver.energy()), 1.0)


def test_ed_solver_accepts_new_symbolic_operator():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0) + 0.25

    solver = EDSolver()
    solver.solve(hamiltonian, nsites=space.num_spatial_orbitals, n_particles=[(1, 1)])

    assert np.isfinite(np.real(solver.energy()))
