from __future__ import annotations

import numpy as np
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian as quspin_hamiltonian

from ...operator import Ladder, Operator as SymbolicOperator, Term
from ...utils.solver import coerce_symbolic_operator, normalize_electron_count
from ..base import SolverBackend
from ..capabilities import BackendCapabilities
from ..compile import compile_quspin_hamiltonian


def _operator_is_complex(operator: SymbolicOperator, *, atol: float = 1e-12) -> bool:
    if abs(complex(operator.constant).imag) > atol:
        return True
    return any(abs(complex(coeff).imag) > atol for _, coeff in operator.iter_terms())

class QuSpinEDBackend(SolverBackend):
    """Concrete exact-diagonalization backend implemented with QuSpin."""

    solver_family = "ed"
    backend_name = "quspin"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_excited_states=False,
        supports_rdm1=True,
        supports_native_expectation=True,
        supports_complex=True,
        preferred_problem_type="quspin",
    )

    def __init__(self, *, iscomplex: bool = False) -> None:
        self.iscomplex = iscomplex
        self.operator = None
        self.space = None
        self.nsites = None
        self.n_particles = None
        self.basis = None
        self.quspin_operator = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.ground_state = None
        self.solve_mode = None
        self.constant_shift = None
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
        if n_particles is None and operator.space.electron_count() is not None:
            pair, _ = normalize_electron_count(space=operator.space)
            n_particles = [pair]
        self.n_particles = n_particles
        self._dtype = np.complex128 if (self.iscomplex or _operator_is_complex(operator)) else np.float64
        self.basis = spinful_fermion_basis_general(
            self.nsites,
            self.n_particles,
            simple_symm=False,
            make_basis=True,
        )

        compilation = compile_quspin_hamiltonian(operator)
        self.constant_shift = compilation.constant_shift
        self.quspin_operator = None
        self.solve_mode = None

        if self.basis.Ns == 0:
            raise RuntimeError("EDSolver constructed an empty basis; check the requested particle sector.")

        if compilation.static_list:
            self.quspin_operator = quspin_hamiltonian(
                static_list=compilation.static_list,
                dynamic_list=[],
                basis=self.basis,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
                dtype=self._dtype,
            )
            if self.basis.Ns == 1:
                self.eigenvalues = np.asarray([self.quspin_operator.eigh()[0][0] + self.constant_shift])
                self.eigenvectors = np.asarray([[1.0]], dtype=self._dtype)
                self.ground_state = np.asarray(self.eigenvectors)[:, 0]
                self.solve_mode = "quspin_eigh_small"
                return self

            eigenvalues, eigenvectors = self.quspin_operator.eigsh(k=1, which="SA")
            order = np.argsort(np.real(np.asarray(eigenvalues)))
            self.eigenvalues = np.asarray(eigenvalues)[order] + self.constant_shift
            self.eigenvectors = np.asarray(eigenvectors)[:, order]
            self.ground_state = np.asarray(self.eigenvectors)[:, 0]
            self.solve_mode = "quspin_eigsh"
            return self

        # Pure constant Hamiltonians have no operator terms to pass to QuSpin.
        # The exact spectrum is still trivial within the requested basis.
        self.eigenvalues = np.full(self.basis.Ns, self.constant_shift, dtype=self._dtype)
        self.eigenvectors = np.eye(self.basis.Ns, dtype=self._dtype)
        self.ground_state = np.asarray(self.eigenvectors)[:, 0]
        self.solve_mode = "analytic_constant"
        return self

    def _require_solution(self) -> None:
        if self.ground_state is None or self.operator is None or self.basis is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def _expectation(self, operator: SymbolicOperator):
        compilation = compile_quspin_hamiltonian(operator)
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
        return self._expectation(self.space.spin_squared_term())

    def diagnostics(self) -> dict[str, object]:
        data = super().diagnostics()
        data.update(
            {
                "basis_size": None if self.basis is None else int(self.basis.Ns),
                "solve_mode": self.solve_mode,
                "n_particles": self.n_particles,
                "dtype": None if self._dtype is None else np.dtype(self._dtype).name,
                "constant_shift": self.constant_shift,
            }
        )
        return data

    def E(self):
        return self.energy()

    def RDM(self):
        return self.rdm1()

    def S2(self):
        return self.s2()
