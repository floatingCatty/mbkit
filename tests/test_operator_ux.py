import numpy as np

from mbkit import (
    EDSolver,
    ElectronicIntegrals,
    ElectronicSpace,
    Operator,
    QuadraticHamiltonian,
    chain,
    cubic,
    general,
    honeycomb,
    kagome,
    ladder,
    rectangular,
    square,
    triangular,
)


def test_space_helpers_create_ready_to_use_spaces():
    line = chain(2, orbitals=["a"], n_electrons_per_spin=(1, 1))
    shelled_line = chain(5, orbitals=["a"], max_shell=3)
    rung = ladder(3, orbitals=["a"])
    grid = square(2, 2, orbitals=["a"])
    rect = rectangular(2, 3, orbitals=["a"])
    tri = triangular(2, 2, orbitals=["a"])
    hexagon = honeycomb(2, 2, orbitals=["a"])
    corner = kagome(2, 1, orbitals=["a"])
    cube = cubic(2, 2, 2, orbitals=["a"])
    custom = general(3, bonds=[(0, 1), (1, 2, "nn", 0.5)])

    assert line.num_sites == 2
    assert shelled_line.available_shells() == (1, 2, 3)
    assert len(shelled_line.select_bonds(shells=2)) == 3
    assert shelled_line.bond_summary()[2]["shell_2"] == 3
    assert rung.num_sites == 6
    assert set(rung.lattice.bond_kinds()) == {"leg", "rung"}
    assert grid.lattice is not None
    assert rect.num_sites == 6
    assert tri.select_bonds("up_right")
    assert hexagon.num_sites == 8
    assert corner.num_sites == 6
    assert cube.select_bonds("x")
    assert cube.select_bonds("y")
    assert cube.select_bonds("z")
    assert custom.select_bonds("nn")


def test_space_first_hubbard_path_and_solver_default_particle_sector():
    space = chain(2, orbitals=["a"], n_electrons_per_spin=(1, 1))
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    solver = EDSolver()
    solver.solve(hamiltonian)

    assert solver.n_particles == [(1, 1)]
    assert np.isfinite(np.real(solver.energy()))


def test_space_methods_cover_common_model_builders_directly():
    space = chain(2, orbitals=["a"])
    multi_orbital = chain(1, orbitals=["dxz", "dyz"])

    assert space.hubbard(t=1.0, U=4.0).is_hermitian()
    assert space.extended_hubbard(hopping=1.0, onsite_U=4.0).is_hermitian()
    assert multi_orbital.slater_kanamori(U=4.0, Up=2.0, J=0.5, Jp=0.5).particle_number_changes() == frozenset({0})


def test_symbolic_operator_anticommutator_and_serialization():
    space = ElectronicSpace(num_sites=1)
    create = space.create(0, spin="up")
    destroy = space.destroy(0, spin="up")

    anticommutator = create.anticommutator(destroy).simplify()

    assert anticommutator.equiv(Operator.identity(space, 1.0))
    assert create.normal_ordered().equiv(create)
    assert anticommutator.serialize()["space"]["num_sites"] == 1


def test_integrals_helpers_cover_restricted_unrestricted_stack_and_transform():
    space = chain(2, orbitals=["a"])
    one_body = np.array([[1.0, 0.2], [0.2, 1.5]])
    two_body = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    two_body[0, 0, 0, 0] = 0.7

    restricted = ElectronicIntegrals.from_restricted(space, one_body=one_body, two_body=two_body, constant=0.3)
    unrestricted = ElectronicIntegrals.from_unrestricted(
        space,
        one_body_alpha=one_body,
        one_body_beta=2.0 * one_body,
        two_body_aa=two_body,
        two_body_ab=0.5 * two_body,
        two_body_bb=1.5 * two_body,
    )
    stacked = ElectronicIntegrals.stack([restricted, 2.0 * restricted])
    split = stacked.split(2)

    assert np.allclose(restricted.one_body_alpha, one_body)
    assert np.allclose(restricted.one_body_beta, one_body)
    assert np.allclose(unrestricted.one_body_beta, 2.0 * one_body)
    assert restricted.basis_transform(np.eye(space.num_spatial_orbitals)).equiv(restricted)
    assert split[0].equiv(restricted)
    assert split[1].equiv(2.0 * restricted)


def test_quadratic_hamiltonian_round_trip():
    space = ElectronicSpace(num_sites=1)
    hermitian = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    pairing = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)

    quadratic = QuadraticHamiltonian(space, constant=0.5, hermitian_part=hermitian, pairing_part=pairing)
    rebuilt = QuadraticHamiltonian.from_operator(quadratic.to_operator())

    assert not quadratic.conserves_particle_number()
    assert rebuilt.equiv(quadratic)
