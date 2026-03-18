import numpy as np
import pytest

from mbkit import ElectronicSpace, LineLattice, models
from mbkit.solver.dmrg_solver import _compile_direct_dmrg_hamiltonian
from mbkit.solver.pyscf_solver import _build_uhf_fci_hamiltonian
from mbkit.operator import UnsupportedTransformError


def test_direct_dmrg_compiler_accepts_symbolic_hamiltonian():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = models.hubbard(space, t=1.0, U=4.0) + 0.25

    compiled = _compile_direct_dmrg_hamiltonian(hamiltonian)

    assert compiled.space.num_spatial_orbitals == 2
    assert np.isclose(compiled.constant_shift, 0.25)
    assert compiled.h1e.shape == (4, 4)
    assert compiled.g2e.shape == (4, 4, 4, 4)


def test_direct_pyscf_builder_extracts_uhf_fci_tensors():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = models.hubbard(space, t=1.0, U=4.0) + 0.25

    problem = _build_uhf_fci_hamiltonian(hamiltonian, n_particles=(1, 1))

    expected_h1 = np.array([[0.0, -1.0], [-1.0, 0.0]])
    assert problem.norb == 2
    assert problem.nelec == (1, 1)
    assert np.isclose(problem.ecore, 0.25)
    assert np.allclose(problem.h1e[0], expected_h1)
    assert np.allclose(problem.h1e[1], expected_h1)
    assert np.isclose(problem.eri[1][0, 0, 0, 0], 4.0)
    assert np.isclose(problem.eri[1][1, 1, 1, 1], 4.0)
    assert np.allclose(problem.eri[0], 0.0)
    assert np.allclose(problem.eri[2], 0.0)


def test_direct_pyscf_builder_rejects_spin_mixing_hamiltonians():
    space = ElectronicSpace(LineLattice(1, boundary="open"), orbitals=["a"])
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = models.soc(space, matrix=matrix)

    with pytest.raises(UnsupportedTransformError, match="spin-mixing one-body terms"):
        _build_uhf_fci_hamiltonian(hamiltonian, n_particles=(1, 0))
