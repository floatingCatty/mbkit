"""Formal tutorial for operator construction and block2 DMRG.

This example mirrors the public ED workflow but uses the stable block2-backed
`DMRGSolver` implementation:

1. define an `ElectronicSpace`,
2. construct the Hamiltonian explicitly from operator terms,
3. solve a fixed-particle problem with `DMRGSolver`,
4. compute observables through the shared solver façade helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

import numpy as np

# Make the example runnable directly from the source tree without requiring an
# editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mbkit import DMRGSolver, ElectronicSpace, LineLattice
from mbkit.solver.dmrg_solver import _PYBLOCK2_IMPORT_ERROR


def build_four_site_hubbard():
    """Build a four-site one-orbital Hubbard chain explicitly from operators."""

    space = ElectronicSpace(LineLattice(4, boundary="open"), orbitals=["a"])
    hopping = space.hopping_term(coeff=-1.0, shells=1, spin="both", plus_hc=True)
    interaction = 4.0 * space.double_occupancy_term()
    hamiltonian = (hopping + interaction).simplify()
    return space, hamiltonian


def main() -> None:
    if _PYBLOCK2_IMPORT_ERROR is not None:
        raise SystemExit(
            "block2 / pyblock2 is not installed. Install the optional dependency "
            'set with `pip install -e ".[dmrg]"` before running this tutorial.'
        )

    space, hamiltonian = build_four_site_hubbard()

    print("block2 DMRG tutorial: four-site Hubbard chain")
    print("=" * 72)
    print("Space:", space)
    print("Hamiltonian body rank:", hamiltonian.body_rank())
    print("Hamiltonian term count:", hamiltonian.term_count())
    print("Hamiltonian is Hermitian:", hamiltonian.is_hermitian())
    print()

    solver = DMRGSolver(
        bond_dim=64,
        bond_mul=2,
        n_sweep=10,
        nupdate=2,
        eig_cutoff=1e-8,
        iprint=0,
        scratch_dir="/tmp/mbkit_block2_tutorial",
    ).solve(hamiltonian, n_particles=(2, 2))

    print("Available solver properties:", solver.available_properties())
    print()

    site1_density = space.number(1, orbital="a", spin="both")
    total_density = space.number_term()

    energy = np.real_if_close(solver.energy()).item()
    rdm1 = np.asarray(solver.rdm1())
    docc = np.asarray(solver.docc())
    s2 = np.real_if_close(solver.s2()).item()
    n1 = np.real_if_close(solver.expect_value(site1_density)).item()
    total_n = np.real_if_close(solver.expect_value(total_density)).item()

    print("Ground-state energy:", energy)
    print("trace(rdm1):", np.real_if_close(np.trace(rdm1)).item())
    print("rdm1 diagonal:", np.real_if_close(np.diag(rdm1)))
    print("Double occupancy by site:", docc)
    print("<S^2>:", s2)
    print("<n_1> from expect_value:", n1)
    print("<N> from expect_value:", total_n)
    print()

    print("Backend diagnostics:")
    pprint(solver.diagnostics())


if __name__ == "__main__":
    main()
