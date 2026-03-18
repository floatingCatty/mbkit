"""Compilation helpers for QuSpin-based solver backends."""

from __future__ import annotations

from ...operator import to_quspin_operator
from ...utils.solver import coerce_symbolic_operator


def compile_quspin_hamiltonian(hamiltonian):
    """Compile a Hamiltonian into the QuSpin static-list representation."""
    operator = coerce_symbolic_operator(hamiltonian)
    return to_quspin_operator(operator)

