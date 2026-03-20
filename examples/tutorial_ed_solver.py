"""Formal tutorial for operator construction and exact diagonalization.

This example demonstrates the full public workflow for the stable ED solver on
the same explicit four-site Hubbard chain used by the block2 DMRG tutorial:

1. define an `ElectronicSpace`,
2. construct a Hamiltonian term by term from operator builders,
3. solve the model with `EDSolver`,
4. compute observables from both solver-specific helpers and the generic
   facade helpers shared across solver families.
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

from mbkit import EDSolver, ElectronicSpace, LineLattice


def build_four_site_hubbard():
    """Build a four-site one-orbital Hubbard chain explicitly from operators."""

    space = ElectronicSpace(LineLattice(4, boundary="open"), orbitals=["a"])
    hopping = space.hopping_term(coeff=-1.0, shells=1, spin="both", plus_hc=True)
    interaction = 4.0 * space.double_occupancy_term()
    hamiltonian = (hopping + interaction).simplify()
    return space, hamiltonian


def main() -> None:
    space, hamiltonian = build_four_site_hubbard()

    print("ED tutorial: four-site Hubbard chain")
    print("=" * 72)
    print("Space:", space)
    print("Hamiltonian body rank:", hamiltonian.body_rank())
    print("Hamiltonian term count:", hamiltonian.term_count())
    print("Hamiltonian is Hermitian:", hamiltonian.is_hermitian())
    print()

    solver = EDSolver().solve(hamiltonian, n_electrons=2)
    print("Available solver properties:", solver.available_properties())
    print()

    site1_up = space.number(1, orbital="a", spin="up")
    site1_total = space.number(1, orbital="a", spin="both")
    total_density = space.number_term()

    energy = np.real_if_close(solver.energy()).item()
    rdm1 = np.asarray(solver.rdm1())
    docc = np.asarray(solver.docc())
    s2 = np.real_if_close(solver.s2()).item()
    n1_up = np.real_if_close(solver.expect_value(site1_up)).item()
    n1_total = np.real_if_close(solver.expect_value(site1_total)).item()
    total_n = np.real_if_close(solver.expect_value(total_density)).item()
    n1_up_stats = solver.expect(site1_up, stats=True)

    print("Ground-state energy:", energy)
    print("trace(rdm1):", np.real_if_close(np.trace(rdm1)).item())
    print("rdm1 diagonal:", np.real_if_close(np.diag(rdm1)))
    print("Double occupancy by site:", docc)
    print("<S^2>:", s2)
    print("<n_1,up> from expect_value:", n1_up)
    print("<n_1> from expect_value:", n1_total)
    print("<N> from expect_value:", total_n)
    print("Deterministic statistics payload for <n_1,up>:")
    pprint(n1_up_stats)
    print()

    print("Backend diagnostics:")
    pprint(solver.diagnostics())


if __name__ == "__main__":
    main()
