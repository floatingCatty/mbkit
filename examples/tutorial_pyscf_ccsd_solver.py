"""Formal tutorial for operator construction and PySCF-backed CCSD.

This example uses the same explicit four-site Hubbard chain as the DMRG and
MP2 tutorials, but routes it through the method-based `CCSDSolver` facade. The
example solves the two-electron `n_up = n_down = 1` sector, not the half-filled
four-electron chain. The stable CCSD API exposes both reference and unrelaxed
one-body observables, so the example demonstrates the required `kind=`
selectors explicitly:

1. define an `ElectronicSpace`,
2. construct the Hamiltonian explicitly from operator terms,
3. solve a fixed-particle problem with `CCSDSolver`,
4. inspect the method-specific observable semantics exposed by the backend.
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

from mbkit import CCSDSolver, ElectronicSpace, LineLattice
from mbkit.solver.pyscf_solver import _PYSCF_IMPORT_ERROR


def build_four_site_hubbard():
    """Build a four-site one-orbital Hubbard chain explicitly from operators."""

    space = ElectronicSpace(LineLattice(4, boundary="open"), orbitals=["a"])
    hopping = space.hopping_term(coeff=-1.0, shells=1, spin="both", plus_hc=True)
    interaction = 4.0 * space.double_occupancy_term()
    hamiltonian = (hopping + interaction).simplify()
    return space, hamiltonian


def main() -> None:
    if _PYSCF_IMPORT_ERROR is not None:
        raise SystemExit(
            "PySCF is not installed. Install the optional dependency set with "
            '`pip install -e ".[pyscf]"` before running this tutorial.'
        )

    space, hamiltonian = build_four_site_hubbard()

    print("PySCF CCSD tutorial: four-site Hubbard chain")
    print("=" * 72)
    print("Space:", space)
    print("Hamiltonian body rank:", hamiltonian.body_rank())
    print("Hamiltonian term count:", hamiltonian.term_count())
    print("Hamiltonian is Hermitian:", hamiltonian.is_hermitian())
    print()

    solver = CCSDSolver().solve(hamiltonian, n_electrons=2)

    print("Available solver properties:", solver.available_properties())
    print("Requested sector: n_up = 1, n_down = 1 (two electrons total; not half filling).")
    print("CCSD exposes one-body observables with explicit kinds; `docc()` is not available.")
    print()

    reference_rdm1 = np.asarray(solver.rdm1(kind="reference"))
    unrelaxed_rdm1 = np.asarray(solver.rdm1(kind="unrelaxed"))
    energy = np.real_if_close(solver.energy()).item()
    s2_reference = np.real_if_close(solver.s2(kind="reference")).item()

    print("Total energy:", energy)
    print("trace(reference rdm1):", np.real_if_close(np.trace(reference_rdm1)).item())
    print("reference rdm1 diagonal:", np.real_if_close(np.diag(reference_rdm1)))
    print("trace(unrelaxed rdm1):", np.real_if_close(np.trace(unrelaxed_rdm1)).item())
    print("unrelaxed rdm1 diagonal:", np.real_if_close(np.diag(unrelaxed_rdm1)))
    print("Reference <S^2>:", s2_reference)
    print()

    print("Backend diagnostics:")
    pprint(solver.diagnostics())


if __name__ == "__main__":
    main()
