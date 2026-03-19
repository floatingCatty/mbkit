"""Capability metadata for solver backends.

This module is intentionally lightweight. The first refactor step is not to
fully standardize every solver task, but to give each backend a place to record
what it is good at so the public solver façade can grow without hard-coding
backend assumptions everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Describe which solver tasks and data paths a backend supports."""

    supports_ground_state: bool = True
    supports_excited_states: bool = False
    supports_time_evolution: bool = False
    supports_rdm1: bool = False
    supports_rdm2: bool = False
    supports_correlation: bool = False
    supports_entanglement: bool = False
    supports_greens_function: bool = False
    supports_statistics: bool = False
    supports_diagnostics: bool = True
    supports_native_expectation: bool = False
    supports_complex: bool = True
    preferred_problem_type: str = "symbolic"
