import numpy as np

from mbkit import ElectronicSpace, LineLattice
from mbkit.solver.compile import compile_local_terms


def test_local_term_compiler_builds_expected_single_site_density_matrix():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    operator = space.crystal_field_term(values={"a": 2.0}, spin="both")

    compiled = compile_local_terms(operator)

    assert compiled.constant_shift == 0.0
    assert compiled.local_basis.local_dim == 4
    assert len(compiled.terms) == 1
    assert compiled.terms[0].sites == (0,)
    assert compiled.terms[0].matrix.shape == (4, 4)
    assert np.allclose(np.diag(compiled.terms[0].matrix), [0.0, 2.0, 2.0, 4.0])


def test_local_term_compiler_builds_expected_two_site_hopping_block():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    operator = space.hopping_term(coeff=-1.0, spin="up", plus_hc=True)

    compiled = compile_local_terms(operator)

    assert len(compiled.terms) == 1
    term = compiled.terms[0]
    assert term.sites == (0, 1)
    assert term.matrix.shape == (16, 16)

    # Local basis per site is:
    # 0 -> |0>, 1 -> |up>, 2 -> |down>, 3 -> |up,down>
    initial = 0 * 4 + 1  # |0, up>
    final = 1 * 4 + 0    # |up, 0>
    assert np.isclose(term.matrix[final, initial], -1.0)
    assert np.isclose(term.matrix[initial, final], -1.0)


def test_local_term_compiler_expands_second_neighbor_hopping_to_three_site_block():
    space = ElectronicSpace(LineLattice(3, boundary="open", max_shell=2), orbitals=["a"])
    operator = space.hopping_term(coeff=-1.0, shells=2, spin="up", plus_hc=True)

    compiled = compile_local_terms(operator)

    assert len(compiled.terms) == 1
    assert compiled.terms[0].sites == (0, 1, 2)
    assert compiled.terms[0].matrix.shape == (64, 64)
