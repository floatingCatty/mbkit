"""Shared helpers for direct solver adapters."""

from __future__ import annotations

import numpy as np

from ..operator.integrals import ElectronicIntegrals
from ..operator.operator import Operator as SymbolicOperator
from ..operator.transforms import to_electronic_integrals


def coerce_symbolic_operator(hamiltonian) -> SymbolicOperator:
    if isinstance(hamiltonian, SymbolicOperator):
        return hamiltonian
    if isinstance(hamiltonian, ElectronicIntegrals):
        return hamiltonian.to_operator()
    raise TypeError(
        "Expected a symbolic mbkit.operator.Operator or an ElectronicIntegrals instance."
    )


def coerce_electronic_integrals(hamiltonian) -> ElectronicIntegrals:
    if isinstance(hamiltonian, ElectronicIntegrals):
        return hamiltonian
    if isinstance(hamiltonian, SymbolicOperator):
        return to_electronic_integrals(hamiltonian)
    raise TypeError(
        "Expected a symbolic mbkit.operator.Operator or an ElectronicIntegrals instance."
    )


def normalize_electron_count(*, space=None, n_electrons=None, n_particles=None) -> tuple[tuple[int, int], int]:
    if n_electrons is not None and n_particles is not None:
        raise ValueError("Provide only one of `n_electrons` or `n_particles`.")

    spec = n_particles if n_particles is not None else n_electrons
    if spec is None and space is not None:
        if getattr(space, "n_electrons_per_spin", None) is not None:
            spec = tuple(space.n_electrons_per_spin)
        elif getattr(space, "n_electrons", None) is not None:
            spec = int(space.n_electrons)
    if spec is None:
        raise ValueError("An electron count must be supplied via `n_electrons` or `n_particles`.")

    if isinstance(spec, (int, np.integer)):
        total = int(spec)
        pair = (total - total // 2, total // 2)
        return pair, total

    if isinstance(spec, tuple) and len(spec) == 2:
        pair = (int(spec[0]), int(spec[1]))
        return pair, sum(pair)

    if isinstance(spec, list):
        if len(spec) == 1 and isinstance(spec[0], tuple) and len(spec[0]) == 2:
            pair = (int(spec[0][0]), int(spec[0][1]))
            return pair, sum(pair)
        if len(spec) == 2 and all(isinstance(value, (int, np.integer)) for value in spec):
            pair = (int(spec[0]), int(spec[1]))
            return pair, sum(pair)

    raise TypeError(
        "Electron counts must be an int, a `(n_alpha, n_beta)` tuple, "
        "or a single-item list containing such a tuple."
    )

