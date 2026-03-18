import numpy as np

from mbkit.solver._solver import Solver
import mbkit.solver._solver as solver_module


class DummySolver(Solver):
    def solve(self, T, intparam):
        raise NotImplementedError


def _spin_degenerate_matrix(orbital_matrix):
    nsite = orbital_matrix.shape[0]
    out = np.zeros((nsite * 2, nsite * 2), dtype=float)
    for i in range(nsite):
        for j in range(nsite):
            out[2 * i, 2 * j] = orbital_matrix[i, j]
            out[2 * i + 1, 2 * j + 1] = orbital_matrix[i, j]
    return out


def test_decoupled_bath_round_trip():
    solver = DummySolver(n_int=1, n_noint=1, n_elec=2, nspin=1)
    t = _spin_degenerate_matrix(np.array([[0.0, -1.0], [-1.0, 0.5]]))

    transformed = solver.cal_decoupled_bath(t)
    recovered = solver.recover_decoupled_bath(transformed)

    assert np.allclose(recovered, t)


def test_natural_orbital_path_uses_local_hartree_fock(monkeypatch):
    solver = DummySolver(n_int=1, n_noint=1, n_elec=2, nspin=1)
    solver.kBT = 1e-5
    solver.mutol = 1e-4
    t = _spin_degenerate_matrix(np.array([[0.0, -1.0], [-1.0, 0.5]]))
    calls = {"hf": 0}

    def fake_hartree_fock_sk(**kwargs):
        calls["hf"] += 1
        return kwargs["h_mat"], np.eye(kwargs["h_mat"].shape[0]) * 0.5, 0.0

    def fake_nao_two_chain(h_mat, D, n_imp, n_bath, nspin):
        return h_mat, D, np.eye(h_mat.shape[0])

    monkeypatch.setattr(solver_module, "hartree_fock_sk", fake_hartree_fock_sk)
    monkeypatch.setattr(solver_module, "nao_two_chain", fake_nao_two_chain)

    new_t = solver.to_natrual_orbital(t, {"U": 1.0, "Up": 0.0, "J": 0.0, "Jp": 0.0})

    assert calls["hf"] == 1
    assert np.allclose(new_t, t)
