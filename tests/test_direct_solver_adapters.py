import numpy as np
import pytest

from mbkit import ElectronicSpace, LineLattice
from mbkit.solver.dmrg_solver import _compile_direct_dmrg_hamiltonian
from mbkit.solver.pyscf_solver import _build_pyscf_reference_problem
from mbkit.operator import UnsupportedTransformError


def test_direct_dmrg_compiler_accepts_symbolic_hamiltonian():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0) + 0.25

    compiled = _compile_direct_dmrg_hamiltonian(hamiltonian)

    assert compiled.space.num_spatial_orbitals == 2
    assert np.isclose(compiled.constant_shift, 0.25)
    assert compiled.h1e.shape == (4, 4)
    assert compiled.g2e.shape == (4, 4, 4, 4)


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


def test_direct_pyscf_reference_builder_extracts_four_site_tutorial_hubbard_integrals():
    space = ElectronicSpace(LineLattice(4, boundary="open"), orbitals=["a"])
    hopping = space.hopping_term(coeff=-1.0, shells=1, spin="both", plus_hc=True)
    interaction = 4.0 * space.double_occupancy_term()
    hamiltonian = (hopping + interaction).simplify()

    problem = _build_pyscf_reference_problem(hamiltonian, n_electrons=2)

    expected_h1 = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    expected_eri = np.zeros((4, 4, 4, 4), dtype=float)
    for orbital in range(4):
        expected_eri[orbital, orbital, orbital, orbital] = 4.0

    assert problem.norb == 4
    assert problem.nelec == (1, 1)
    assert np.isclose(problem.ecore, 0.0)
    assert np.allclose(problem.h1e[0], expected_h1)
    assert np.allclose(problem.h1e[1], expected_h1)
    assert np.allclose(problem.eri, expected_eri)
