import numpy as np

from mbkit import Hubbard, Operator, Slater_Kanamori, create_u, annihilate_u, multi_orbital_extended_hubbard


def test_operator_round_trip_qc_tensors():
    op = create_u(2, 0) * annihilate_u(2, 1) + create_u(2, 1) * annihilate_u(2, 0)
    h1e, g2e = op.get_Hqc(2)
    recovered = Operator.from_Hqc(h1e, g2e)
    h1e_new, g2e_new = recovered.get_Hqc(2)
    assert np.allclose(h1e_new, h1e)
    assert np.allclose(g2e_new, g2e)


def test_hubbard_builder_produces_terms():
    hamiltonian = Hubbard(neighbors=[[(0, 1)]], nsites=2, U=4.0, t=1.0)
    assert len(hamiltonian.op_list) > 0
    h1e, g2e = hamiltonian.get_Hqc(2)
    assert h1e.shape == (4, 4)
    assert g2e.shape == (4, 4, 4, 4)


def test_slater_kanamori_builder_produces_operator():
    t = np.zeros((4, 4))
    t[0, 2] = t[2, 0] = -1.0
    hamiltonian = Slater_Kanamori(nsites=2, t=t, U=2.0)
    assert len(hamiltonian.op_list) > 0


def test_multi_orbital_extended_hubbard_builder_shapes():
    hamiltonian = multi_orbital_extended_hubbard(Nx=2, Ny=1, chem_1=0.0, chem_2=0.0)
    h1e, g2e = hamiltonian.get_Hqc(4)
    assert h1e.shape == (8, 8)
    assert g2e.shape == (8, 8, 8, 8)
