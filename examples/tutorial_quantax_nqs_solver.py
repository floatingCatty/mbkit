"""Formal tutorial for operator construction and Quantax-backed NQS solving.

This example demonstrates the public neural-quantum-state workflow on a small,
exactly sampled two-site one-orbital Hubbard dimer:

1. define an `ElectronicSpace`,
2. construct the hopping Hamiltonian explicitly from operator builders,
3. solve the fixed `(n_up, n_down) = (1, 1)` sector with `NQSSolver`,
4. compare the NQS energy against a lightweight ED reference, and
5. inspect observables through the shared solver facade helpers.
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

from mbkit import EDSolver, ElectronicSpace, LineLattice, NQSSolver
from mbkit.solver.backends.quantax_nqs import _QUSPIN_IMPORT_ERROR
from mbkit.solver.nqs_solver import _QUANTAX_IMPORT_ERROR


def build_two_site_dimer():
    """Build a two-site one-orbital hopping dimer explicitly from operators."""

    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    hopping = space.hopping_term(coeff=-1.0, shells=1, spin="both", plus_hc=True)
    hamiltonian = hopping.simplify()
    return space, hamiltonian


def main() -> None:
    if _QUANTAX_IMPORT_ERROR is not None:
        raise SystemExit(
            "Quantax is not installed. Install the optional dependency set with "
            '`pip install -e ".[nqs]"` before running this tutorial. Quantax also '
            "depends on JAX, so follow the JAX installation instructions for your "
            "platform and accelerator if needed."
        )
    if _QUSPIN_IMPORT_ERROR is not None:
        raise SystemExit(
            "QuSpin is not installed. This tutorial requests `sampler=\"exact\"`, "
            "which requires QuSpin in addition to the Quantax NQS stack. Install "
            'the optional dependency set with `pip install -e ".[nqs]"` and ensure '
            "JAX, Quantax, and QuSpin are available."
        )

    space, hamiltonian = build_two_site_dimer()

    print("Quantax NQS tutorial: two-site one-orbital hopping dimer")
    print("=" * 72)
    print("Space:", space)
    print("Hamiltonian body rank:", hamiltonian.body_rank())
    print("Hamiltonian term count:", hamiltonian.term_count())
    print("Hamiltonian is Hermitian:", hamiltonian.is_hermitian())
    print()

    ed_solver = EDSolver().solve(hamiltonian, n_particles=(1, 1))
    solver = NQSSolver(
        model="general_det",
        optimizer="sr",
        sampler="exact",
        nsamples=256,
        n_steps=25,
        step_size=0.2,
        seed=13,
    ).solve(hamiltonian, n_particles=(1, 1))

    print("Available solver properties:", solver.available_properties())
    print("Requested sector: n_up = 1, n_down = 1 via `n_particles=(1, 1)`.")
    print(
        "For `NQSSolver`, passing `n_electrons=2` would instead select the full "
        "two-particle sector, so this tutorial keeps the spin-resolved sector "
        "explicit to match the ED reference."
    )
    print()

    site0_up = space.number(0, orbital="a", spin="up")
    site0_total = space.number(0, orbital="a", spin="both")
    total_density = space.number_term()

    ed_energy = np.real_if_close(ed_solver.energy()).item()
    energy = np.real_if_close(solver.energy()).item()
    rdm1 = np.asarray(solver.rdm1())
    docc = np.asarray(solver.docc())
    s2 = np.real_if_close(solver.s2()).item()
    n0_up = np.real_if_close(solver.expect_value(site0_up)).item()
    n0_total = np.real_if_close(solver.expect_value(site0_total)).item()
    total_n = np.real_if_close(solver.expect_value(total_density)).item()
    n0_up_stats = solver.expect(site0_up, stats=True)

    print("ED reference energy:", ed_energy)
    print("NQS energy:", energy)
    print("Energy difference (NQS - ED):", energy - ed_energy)
    print("trace(rdm1):", np.real_if_close(np.trace(rdm1)).item())
    print("rdm1 diagonal:", np.real_if_close(np.diag(rdm1)))
    print("Double occupancy by site:", docc)
    print("<S^2>:", s2)
    print("<n_0,up> from expect_value:", n0_up)
    print("<n_0> from expect_value:", n0_total)
    print("<N> from expect_value:", total_n)
    print("Statistics payload for <n_0,up>:")
    pprint(n0_up_stats)
    print()

    print("Backend diagnostics:")
    pprint(solver.diagnostics())


if __name__ == "__main__":
    main()
