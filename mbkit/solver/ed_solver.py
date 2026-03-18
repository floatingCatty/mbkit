from __future__ import annotations

import numpy as np
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian as quspin_hamiltonian

from ..operator import Ladder, Operator as SymbolicOperator, Term, models, to_quspin_operator
from ._frontend import coerce_symbolic_operator


def _operator_is_complex(operator: SymbolicOperator, *, atol: float = 1e-12) -> bool:
    if abs(complex(operator.constant).imag) > atol:
        return True
    return any(abs(complex(coeff).imag) > atol for _, coeff in operator.iter_terms())


class EDSolver:
    """Direct exact-diagonalization solver for symbolic operators and structured integrals."""

    def __init__(self, *, iscomplex: bool = False) -> None:
        self.iscomplex = iscomplex
        self.operator = None
        self.space = None
        self.nsites = None
        self.n_particles = None
        self.basis = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.ground_state = None
        self._dtype = None

    def solve(self, hamiltonian, *, nsites: int | None = None, n_particles=None):
        operator = coerce_symbolic_operator(hamiltonian)
        inferred_nsites = operator.space.num_spatial_orbitals
        if nsites is None:
            nsites = inferred_nsites
        elif int(nsites) != inferred_nsites:
            raise ValueError(
                "EDSolver.solve() received an `nsites` value that does not match "
                "the Hamiltonian ElectronicSpace."
            )

        self.operator = operator
        self.space = operator.space
        self.nsites = int(nsites)
        self.n_particles = n_particles
        self._dtype = np.complex128 if (self.iscomplex or _operator_is_complex(operator)) else np.float64
        self.basis = spinful_fermion_basis_general(
            self.nsites,
            self.n_particles,
            simple_symm=False,
            make_basis=True,
        )

        compilation = to_quspin_operator(operator)
        if compilation.static_list:
            quspin_op = quspin_hamiltonian(
                static_list=compilation.static_list,
                dynamic_list=[],
                basis=self.basis,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
                dtype=self._dtype,
            )
            dense = np.asarray(quspin_op.tocsr().toarray(), dtype=self._dtype)
        else:
            dense = np.zeros((self.basis.Ns, self.basis.Ns), dtype=self._dtype)

        if dense.shape[0] == 0:
            raise RuntimeError("EDSolver constructed an empty basis; check the requested particle sector.")

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(dense)
        self.eigenvalues = np.asarray(self.eigenvalues) + compilation.constant_shift
        self.ground_state = np.asarray(self.eigenvectors)[:, 0]
        return self

    def _require_solution(self) -> None:
        if self.ground_state is None or self.operator is None or self.basis is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def _expectation(self, operator: SymbolicOperator):
        compilation = to_quspin_operator(operator)
        value = compilation.constant_shift
        if compilation.static_list:
            quspin_op = quspin_hamiltonian(
                static_list=compilation.static_list,
                dynamic_list=[],
                basis=self.basis,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
                dtype=self._dtype,
            )
            value += quspin_op.expt_value(np.asarray(self.ground_state))
        return value

    def energy(self):
        self._require_solution()
        return np.asarray(self.eigenvalues)[0]

    def rdm1(self):
        self._require_solution()
        nmode = self.space.num_spin_orbitals
        rdm = np.zeros((nmode, nmode), dtype=np.complex128)
        for p in range(nmode):
            for q in range(nmode):
                operator = SymbolicOperator(
                    self.space,
                    terms={Term((Ladder(p, "create"), Ladder(q, "destroy"))): 1.0},
                )
                rdm[p, q] = self._expectation(operator)
        return np.real_if_close(rdm)

    def docc(self):
        self._require_solution()
        values = []
        for site in range(self.space.num_sites):
            for orbital in self.space.orbitals:
                operator = (
                    self.space.number(site, orbital=orbital, spin="up")
                    @ self.space.number(site, orbital=orbital, spin="down")
                )
                values.append(np.real_if_close(self._expectation(operator)).item())
        return np.asarray(values, dtype=float)

    def s2(self):
        self._require_solution()
        return self._expectation(models.spin_squared(self.space))

    def E(self):
        return self.energy()

    def RDM(self):
        return self.rdm1()

    def S2(self):
        return self.s2()


ED_solver = EDSolver
