import numpy as np
from ..operator import S_m, S_p, S_z, Operator, annihilate_d, annihilate_u, create_d, create_u, number_d, number_u
from ._solver import Solver


class EDSolver:
    """Direct exact-diagonalization solver for ``Operator`` objects."""

    def __init__(self, *, iscomplex: bool = False) -> None:
        self.iscomplex = iscomplex
        self.operator = None
        self.nsites = None
        self.n_particles = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.ground_state = None

    def solve(self, operator: Operator, *, nsites: int, n_particles):
        if not isinstance(operator, Operator):
            raise TypeError("EDSolver.solve() expects an mbkit.operator.Operator instance.")

        self.operator = operator
        self.nsites = int(nsites)
        self.n_particles = n_particles
        self.eigenvalues, self.eigenvectors = operator.diagonalize(
            nsites=self.nsites,
            Nparticle=self.n_particles,
            iscomplex=self.iscomplex,
        )
        self.ground_state = np.asarray(self.eigenvectors)[:, 0]
        return self

    def _require_solution(self) -> None:
        if self.ground_state is None or self.operator is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def energy(self):
        self._require_solution()
        return np.asarray(self.eigenvalues)[0]

    def rdm1(self):
        self._require_solution()
        vec = np.asarray(self.ground_state)
        rdm = np.zeros((self.nsites, 2, self.nsites, 2), dtype=np.complex128 if self.iscomplex else np.float64)
        for a in range(self.nsites):
            for s in range(2):
                for b in range(a, self.nsites):
                    start = s if a == b else 0
                    for s_ in range(start, 2):
                        if s == 0 and s_ == 0:
                            op = create_u(self.nsites, a) * annihilate_u(self.nsites, b)
                        elif s == 1 and s_ == 1:
                            op = create_d(self.nsites, a) * annihilate_d(self.nsites, b)
                        elif s == 0 and s_ == 1:
                            op = create_u(self.nsites, a) * annihilate_d(self.nsites, b)
                        else:
                            op = create_d(self.nsites, a) * annihilate_u(self.nsites, b)

                        value = op.get_quspin_op(self.nsites, self.n_particles, iscomplex=self.iscomplex).expt_value(vec)
                        rdm[a, s, b, s_] = value
                        rdm[b, s_, a, s] = value.conj()
        return rdm.reshape(2 * self.nsites, 2 * self.nsites)

    def docc(self):
        self._require_solution()
        vec = np.asarray(self.ground_state)
        docc = np.zeros(self.nsites)
        for i in range(self.nsites):
            op = number_u(self.nsites, i) * number_d(self.nsites, i)
            docc[i] = op.get_quspin_op(self.nsites, self.n_particles, iscomplex=self.iscomplex).expt_value(vec).real
        return docc

    def s2(self):
        self._require_solution()
        vec = np.asarray(self.ground_state)
        s_minus = sum(S_m(self.nsites, i) for i in range(self.nsites))
        s_plus = sum(S_p(self.nsites, i) for i in range(self.nsites))
        s_z = sum(S_z(self.nsites, i) for i in range(self.nsites))
        s2 = s_minus * s_plus + s_z * s_z + s_z
        return s2.get_quspin_op(self.nsites, self.n_particles, iscomplex=self.iscomplex).expt_value(vec)

    def E(self):
        return self.energy()

    def RDM(self):
        return self.rdm1()

    def S2(self):
        return self.s2()

class ED_solver(Solver):
    def __init__(
            self, 
            n_int, 
            n_noint,
            n_particle: list, 
            nspin, 
            kBT: float=0.025,
            mutol: float=1e-4,
            decouple_bath: bool=False, 
            natural_orbital: bool=False, 
            iscomplex=False
            ) -> None:
        super(ED_solver, self).__init__(
            n_int, 
            n_noint,
            sum(n_particle[0]),
            nspin,
            decouple_bath, 
            natural_orbital,
            iscomplex
        )

        self.mutol = mutol
        self.kBT = kBT
        if self.nspin == 1:
            assert len(n_particle) == 1 and sum(n_particle[0]) < 2 * self.n_int + self.n_noint, "The spin degenerate case cannot have more electron occupation than the number of orbitals."
            assert n_particle[0][0] == n_particle[0][1]
            self.Nparticle = n_particle
        else:
            # self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
            for pair in n_particle:
                assert sum(pair) < 2 * (self.n_int + self.n_noint)
            self.Nparticle = n_particle

        self._t = 0.
        self._intparam = {}
        
    def solve(self, T, intparam):
        Hop = self.get_Hop(T=T, intparam=intparam)
        nsites = self.n_int+self.n_noint

        val, vec = Hop.diagonalize(
            nsites=nsites, 
            Nparticle=self.Nparticle,
            iscomplex=self.iscomplex,
        )

        self.vec = vec
        return True
    
    def cal_RDM(self, vec):
        vec = np.asarray(vec)
        nsites = self.n_int+self.n_noint

        # # compute RDM
        RDM = np.zeros((nsites, 2, nsites, 2))
        if self.iscomplex:
            RDM = RDM + 0j
        for a in range(nsites):
            for s in range(2):
                for b in range(a, nsites):
                    if a == b:
                        start = s
                    else:
                        start = 0

                    for s_ in range(start,2):
                        if s == 0 and s_ == 0:
                            op = create_u(nsites, a) * annihilate_u(nsites, b)
                        elif s == 1 and s_ == 1:
                            op = create_d(nsites, a) * annihilate_d(nsites, b)
                        elif s == 0 and s_ == 1:
                            op = create_u(nsites, a) * annihilate_d(nsites, b)
                        else:
                            op = create_d(nsites, a) * annihilate_u(nsites, b)
                        
                        v = op.get_quspin_op(nsites, self.Nparticle).expt_value(vec)
                        RDM[a, s, b, s_] = v
                        RDM[b, s_, a, s] = v.conj()

        RDM = RDM.reshape(2*nsites, 2*nsites)
        
        if self.decouple_bath:
            RDM = self.recover_decoupled_bath(RDM)

        if self.natural_orbital:
            RDM = self.transmat.conj().T @ RDM @ self.transmat
        return RDM
    
    def cal_E(self, vec, intonly=False):
        Eop = self.get_Eop(intonly=intonly)

        E = Eop.get_quspin_op(self.n_int+self.n_noint, self.Nparticle).expt_value(vec)

        return E
    
    def cal_docc(self, vec, intonly=False):
        vec = np.asarray(vec)
        nsites = self.n_int+self.n_noint

        if intonly:
            DOCC = np.zeros(self.n_int)
            dorb = self.n_int
        else:
            DOCC = np.zeros(nsites)
            dorb = nsites
        
        for i in range(dorb):
            op = number_u(nsites, i) * number_d(nsites, i)

            v = op.get_quspin_op(nsites, self.Nparticle).expt_value(vec)
            DOCC[i] = v.real
        
        return DOCC

    def cal_S2(self, vec, intonly=False):
        vec = np.asarray(vec)
        nsites = self.n_int+self.n_noint

        S2 = self.get_S2op(intonly=intonly)
        v = S2.get_quspin_op(nsites, self.Nparticle).expt_value(vec)
        
        return v
    
    def E(self, intonly=False):
        return self.cal_E(self.vec, intonly=intonly)

    def S2(self, intonly=False):
        return self.cal_S2(self.vec, intonly=intonly)
    
    def RDM(self):
        return self.cal_RDM(self.vec)
    
    def docc(self, intonly=False):
        return self.cal_docc(self.vec, intonly=intonly)
