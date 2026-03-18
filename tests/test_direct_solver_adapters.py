import numpy as np
import pytest

from mbkit import ElectronicSpace, LineLattice
from mbkit.solver.dmrg_solver import _compile_direct_dmrg_hamiltonian
from mbkit.solver.pyscf_solver import _build_pyscf_reference_problem, _build_uhf_fci_hamiltonian
from mbkit.operator import UnsupportedTransformError


def test_direct_dmrg_compiler_accepts_symbolic_hamiltonian():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0) + 0.25

    compiled = _compile_direct_dmrg_hamiltonian(hamiltonian)

    assert compiled.space.num_spatial_orbitals == 2
    assert np.isclose(compiled.constant_shift, 0.25)
    assert compiled.h1e.shape == (4, 4)
    assert compiled.g2e.shape == (4, 4, 4, 4)


def test_direct_pyscf_builder_extracts_spin_sector_problem():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0) + 0.25

    problem = _build_uhf_fci_hamiltonian(hamiltonian, n_particles=(1, 1))

    assert problem.norb == 2
    assert problem.nelec == (1, 1)
    assert np.isclose(problem.constant, 0.25)
    assert len(problem.alpha_strings) == 2
    assert len(problem.beta_strings) == 2
    assert len(problem.basis_states) == 4
    assert len(problem.one_body_terms) == 4
    assert len(problem.two_body_terms) == 2

    one_body = {(p, q): complex(coeff) for p, q, coeff in problem.one_body_terms}
    assert np.isclose(one_body[(0, 1)], -1.0)
    assert np.isclose(one_body[(1, 0)], -1.0)
    assert np.isclose(one_body[(2, 3)], -1.0)
    assert np.isclose(one_body[(3, 2)], -1.0)

    two_body = {(p, q, r, s): complex(coeff) for p, q, r, s, coeff in problem.two_body_terms}
    assert np.isclose(two_body[(2, 0, 2, 0)], -4.0)
    assert np.isclose(two_body[(3, 1, 3, 1)], -4.0)


def test_direct_pyscf_builder_rejects_spin_mixing_hamiltonians():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = space.soc_term(matrix=matrix)

    with pytest.raises(UnsupportedTransformError, match="preserve n_up and n_down"):
        _build_uhf_fci_hamiltonian(hamiltonian, n_particles=(1, 0))


def test_direct_pyscf_reference_builder_extracts_hubbard_style_spatial_integrals():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0) + 0.25

    problem = _build_pyscf_reference_problem(hamiltonian, n_particles=(1, 1))

    expected_h1 = np.array([[0.0, -1.0], [-1.0, 0.0]])
    expected_eri = np.zeros((2, 2, 2, 2), dtype=float)
    expected_eri[0, 0, 0, 0] = 4.0
    expected_eri[1, 1, 1, 1] = 4.0

    assert problem.norb == 2
    assert problem.nelec == (1, 1)
    assert np.isclose(problem.ecore, 0.25)
    assert np.allclose(problem.h1e[0], expected_h1)
    assert np.allclose(problem.h1e[1], expected_h1)
    assert np.allclose(problem.eri, expected_eri)
