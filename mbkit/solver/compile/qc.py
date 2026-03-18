"""Compilation helpers for QC-tensor-based solver backends."""

from __future__ import annotations

from ...operator import to_qc_tensors


def compile_qc_hamiltonian(hamiltonian):
    """Compile a Hamiltonian into QC one-/two-body tensors."""
    return to_qc_tensors(hamiltonian)

