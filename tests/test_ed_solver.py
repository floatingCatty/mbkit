import numpy as np

from mbkit import EDSolver, ElectronicSpace, LineLattice, models


def test_ed_solver_runs_end_to_end():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = models.hubbard(space, t=1.0, U=4.0)

    solver = EDSolver()
    solver.solve(hamiltonian, nsites=space.num_spatial_orbitals, n_particles=[(1, 1)])

    energy = solver.energy()
    rdm1 = solver.rdm1()
    docc = solver.docc()
    s2 = solver.s2()

    assert np.isfinite(np.real(energy))
    assert rdm1.shape == (4, 4)
    assert np.allclose(rdm1, rdm1.conj().T)
    assert docc.shape == (2,)
    assert np.all(docc >= -1e-10)
    assert np.all(docc <= 1.0 + 1e-10)
    assert np.isfinite(np.real(s2))
