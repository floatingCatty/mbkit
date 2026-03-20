import pytest

from mbkit import ElectronicSpace, LineLattice
from mbkit.operator.transforms import UnsupportedTransformError
from mbkit.solver.compile import compile_quantax_hamiltonian


def test_quantax_compiler_builds_expected_onsite_density_terms():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    operator = space.crystal_field_term(values={"a": 2.0}, spin="both")

    compiled = compile_quantax_hamiltonian(operator)

    assert compiled.constant_shift == 0.0
    assert compiled.num_spatial_orbitals == 1
    assert compiled.hopping_graph == ()
    assert compiled.op_list == [["+-", [[2.0 + 0.0j, 0, 0], [2.0 + 0.0j, 1, 1]]]]


def test_quantax_compiler_builds_expected_hopping_terms_and_graph():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    operator = space.hopping_term(coeff=-1.0, spin="up", plus_hc=True)

    compiled = compile_quantax_hamiltonian(operator)

    assert compiled.constant_shift == 0.0
    assert compiled.hopping_graph == ((0, 1),)
    assert compiled.op_list == [["+-", [[-1.0 + 0.0j, 0, 1], [-1.0 + 0.0j, 1, 0]]]]


def test_quantax_compiler_handles_inter_orbital_terms_with_up_down_blocks():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a", "b"])
    operator = space.create(0, orbital="a", spin="down") @ space.destroy(0, orbital="b", spin="down")

    compiled = compile_quantax_hamiltonian(operator)

    assert compiled.num_spatial_orbitals == 2
    assert compiled.op_list == [["+-", [[1.0 + 0.0j, 2, 3]]]]


def test_quantax_compiler_preserves_constant_shifts():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    operator = 1.25 + space.hopping_term(coeff=-1.0, spin="down", plus_hc=True)

    compiled = compile_quantax_hamiltonian(operator)

    assert compiled.constant_shift == 1.25 + 0.0j
    assert compiled.op_list == [["+-", [[-1.0 + 0.0j, 2, 3], [-1.0 + 0.0j, 3, 2]]]]


def test_quantax_compiler_matches_expected_mode_index_remapping():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a", "b"])
    operator = space.create(1, orbital="b", spin="down") @ space.destroy(0, orbital="a", spin="down")

    compiled = compile_quantax_hamiltonian(operator)

    assert compiled.num_spatial_orbitals == 4
    assert compiled.op_list == [["+-", [[1.0 + 0.0j, 7, 4]]]]


def test_quantax_compiler_rejects_particle_nonconserving_terms():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    operator = space.create(0, orbital="a", spin="up") @ space.create(0, orbital="a", spin="down")

    with pytest.raises(UnsupportedTransformError, match="particle-number-conserving"):
        compile_quantax_hamiltonian(operator)


def test_quantax_compiler_accepts_spin_changing_particle_conserving_terms():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    operator = space.spin_plus(0, orbital="a")

    compiled = compile_quantax_hamiltonian(operator)

    assert compiled.constant_shift == 0.0
    assert compiled.num_spatial_orbitals == 1
    assert compiled.hopping_graph == ()
    assert compiled.op_list == [["+-", [[1.0 + 0.0j, 0, 1]]]]
