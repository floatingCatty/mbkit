from mbkit import EDSolver, ElectronicSpace, LineLattice, available_solver_backends, get_backend_spec, get_solver_backend_class


def test_solver_registry_exposes_registered_backend_families():
    backends = available_solver_backends()

    assert backends["ed"] == ("quspin",)
    assert backends["dmrg"] == ("block2", "quimb", "tenpy")
    assert backends["qc"] == ("pyscf_fci", "pyscf_reference")


def test_solver_registry_resolves_default_backend_specs():
    assert get_backend_spec("ed").name == "quspin"
    assert get_backend_spec("dmrg").name == "block2"
    assert get_backend_spec("qc").name == "pyscf_fci"
    assert get_solver_backend_class("ed").__name__ == "QuSpinEDBackend"
    assert get_solver_backend_class("dmrg", "quimb").__name__ == "QuimbDMRGBackend"
    assert get_solver_backend_class("dmrg", "tenpy").__name__ == "TeNPyDMRGBackend"
    assert get_solver_backend_class("qc", "pyscf_reference").__name__ == "PySCFReferenceBackend"


def test_ed_solver_facade_delegates_to_registered_backend():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hamiltonian = space.hubbard(t=1.0, U=4.0)

    solver = EDSolver()
    returned = solver.solve(hamiltonian, nsites=space.num_spatial_orbitals, n_particles=[(1, 1)])

    assert returned is solver
    assert solver.backend_name == "quspin"
    assert solver.unwrap_backend().__class__.__name__ == "QuSpinEDBackend"
    assert solver.available_backends() == ("quspin",)
