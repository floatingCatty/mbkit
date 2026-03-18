import numpy as np

from mbkit import Bond, ElectronicSpace, EDSolver, GeneralLattice, LineLattice, Operator, SquareLattice, models
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
    hamiltonian = models.hubbard(space, t=1.0, U=4.0)

    quspin_compiled = to_quspin_operator(hamiltonian)
    qc_compiled = to_qc_tensors(hamiltonian)

    assert hamiltonian.is_hermitian()
    assert quspin_compiled.constant_shift == 0.0
    assert quspin_compiled.static_list
    assert qc_compiled.h1e.shape == (4, 4)
    assert qc_compiled.g2e.shape == (4, 4, 4, 4)


def test_slater_kanamori_builder_produces_quartic_terms():
    space = ElectronicSpace(LineLattice(1), orbitals=["dxz", "dyz"])
    hamiltonian = models.slater_kanamori(space, U=4.0, Up=2.6, J=0.7, Jp=0.7)

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
    hamiltonian = models.extended_hubbard(
        space,
        hopping={"nn": -1.0, "diag": -0.2},
        onsite_U=4.0,
        intersite_V={"diag": 0.5},
    )

    compiled = to_quspin_operator(hamiltonian)
    assert compiled.static_list


def test_square_lattice_hubbard_supports_kind_resolved_hoppings_and_mu():
    space = ElectronicSpace(SquareLattice(2, 2, boundary=("open", "open")), orbitals=["a"])
    hamiltonian = models.hubbard(
        space,
        t={"horizontal": 1.0, "vertical": 0.5},
        U=2.0,
        mu={0: 0.25, 3: -0.25},
    )

    compiled = to_quspin_operator(hamiltonian)
    assert compiled.static_list


def test_generic_hopping_builder_respects_bond_weight():
    lattice = GeneralLattice(num_sites=2, bonds=[Bond(0, 1, kind="nn", weight=2.0)])
    space = ElectronicSpace(lattice, orbitals=["a"])
    operator = models.hopping(space, coeff=-1.5, bonds="all", spin="both", plus_hc=True)
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
    operator = models.soc(space, strength=0.5, matrix=matrix)
    integrals = to_electronic_integrals(operator)

    assert np.isclose(integrals.one_body[0, 3], 0.5j)
    assert np.isclose(integrals.one_body[3, 0], -0.5j)


def test_observable_builders_match_simple_one_site_limits():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])

    spin_solver = EDSolver()
    spin_solver.solve(models.spin_squared(space), nsites=space.num_spatial_orbitals, n_particles=[(1, 0)])
    assert np.isclose(np.real(spin_solver.energy()), 0.75)

    docc_solver = EDSolver()
    docc_solver.solve(models.double_occupancy(space), nsites=space.num_spatial_orbitals, n_particles=[(1, 1)])
    assert np.isclose(np.real(docc_solver.energy()), 1.0)


def test_ed_solver_accepts_new_symbolic_operator():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = models.hubbard(space, t=1.0, U=4.0) + 0.25

    solver = EDSolver()
    solver.solve(hamiltonian, nsites=space.num_spatial_orbitals, n_particles=[(1, 1)])

    assert np.isfinite(np.real(solver.energy()))
