"""Two-orbital extended Hubbard example translated from `extern/extended_hubbard`.

This example shows that the current `mbkit` operator stack can express the
reference two-orbital square-lattice model without any legacy helpers:

- shell-resolved hopping on a square lattice
- orbital-resolved onsite energies
- a full onsite four-index Coulomb tensor
- intersite density-density couplings on first and second neighbor shells

The original reference function in `extern/extended_hubbard/extended_hubbard.py`
accepts `chem_1` and `chem_2` parameters but does not use them in the actual
Hamiltonian assembly. This example follows the implemented Hamiltonian terms and
shows how the generic local matrix/tensor helpers remove the need for manual
quartic expansion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the example runnable directly from the source tree without requiring an
# editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mbkit import Operator, square


REDUCED_COULOMB_TENSOR_MEV = np.array(
    [
        [
            [[56.4984977343, -11.9644010036], [-11.9644010036, 6.4936098316]],
            [[-11.9644010036, 43.5781887179], [6.4936098316, -5.4931427368]],
        ],
        [
            [[-11.9644010036, 6.4936098316], [43.5781887179, -5.4931427368]],
            [[6.4936098316, -5.4931427368], [-5.4931427368, 37.5052585228]],
        ],
    ],
    dtype=np.float64,
)

# The external reference stores the active-space onsite interaction as U[p, q, s, r].
# `mbkit` uses the direct symbolic convention U[p, q, r, s] * c_p^dagger c_q^dagger c_r c_s.
REDUCED_COULOMB_TENSOR_PQRS_MEV = np.transpose(REDUCED_COULOMB_TENSOR_MEV, (0, 1, 3, 2))


def build_reference_two_orbital_extended_hubbard(nx: int = 2, ny: int = 2):
    """Build the reference two-orbital square-lattice extended Hubbard model."""

    orbitals = ("1s-A1", "2s-A1")
    space = square(nx, ny, orbitals=orbitals, boundary=("open", "open"), max_shell=2)

    # Reference parameters from the external model.
    t_nn = 1.0
    t_nnn = 0.5
    t_nn_2s = 1.5
    t_nnn_2s = 0.75

    epsilon_1s = -74.0
    epsilon_2s = epsilon_1s + (43.59 - 14.79)

    w1 = 18.0
    w2 = 12.0

    hamiltonian = Operator.zero(space)

    # Shell-first hopping directly matches the external `nn` and `nnn` lattice.
    hamiltonian += space.hopping_term(
        coeff={1: -t_nn, 2: -t_nnn},
        shells=(1, 2),
        orbital_pairs=[("1s-A1", "1s-A1")],
        spin="both",
        plus_hc=True,
    )
    hamiltonian += space.hopping_term(
        coeff={1: -t_nn_2s, 2: -t_nnn_2s},
        shells=(1, 2),
        orbital_pairs=[("2s-A1", "2s-A1")],
        spin="both",
        plus_hc=True,
    )

    # Onsite orbital energies.
    hamiltonian += space.crystal_field_term(
        values={"1s-A1": epsilon_1s, "2s-A1": epsilon_2s},
        spin="both",
    )

    # Onsite four-index Coulomb tensor.
    hamiltonian += space.local_interaction_tensor_term(
        tensor=REDUCED_COULOMB_TENSOR_PQRS_MEV,
        strength=0.5,
        orbitals=orbitals,
    )

    # Intersite density-density interactions across both orbitals and spins.
    hamiltonian += space.density_density_term(
        coeff={1: w1, 2: w2},
        shells=(1, 2),
        spin="both",
        orbital_pairs=[
            ("1s-A1", "1s-A1"),
            ("1s-A1", "2s-A1"),
            ("2s-A1", "1s-A1"),
            ("2s-A1", "2s-A1"),
        ],
    )

    return space, hamiltonian.simplify()


def main() -> None:
    space, hamiltonian = build_reference_two_orbital_extended_hubbard(nx=12, ny=1)

    print("Space:", space)
    print("Available shells:", space.available_shells())
    print("Bond summary:", space.bond_summary())
    print("Body rank:", hamiltonian.body_rank())
    print("Term count:", hamiltonian.term_count())
    print("Hermitian:", hamiltonian.is_hermitian())

    from mbkit.solver import EDSolver

    solver = EDSolver()
    print(space.num_sites // 2, "electrons")
    solver.solve(hamiltonian, n_electrons=space.num_sites // 2)
    # print(solver.rdm1())
    import matplotlib.pyplot as plt

    rdm = solver.rdm1()
    nsites = space.num_sites
    rdm = rdm.reshape(nsites,2,2,nsites,2,2)
    rdm = rdm.transpose(1,0,2,4,3,5).reshape(nsites*4,-1)

    plt.matshow(rdm, cmap="bwr", vmin=-1, vmax=1)
    plt.savefig("two_orbital_extended_hubbard_rdm1.png", dpi=300)

if __name__ == "__main__":
    main()
