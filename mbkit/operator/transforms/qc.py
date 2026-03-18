from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..integrals import ElectronicIntegrals
from ..operator import Operator
from .integrals import to_electronic_integrals


@dataclass(frozen=True)
class QCTensorCompilation:
    space: object
    constant_shift: complex
    h1e: np.ndarray
    g2e: np.ndarray


def to_qc_tensors(hamiltonian, *, atol: float = 1e-12) -> QCTensorCompilation:
    if isinstance(hamiltonian, ElectronicIntegrals):
        integrals = hamiltonian
    elif isinstance(hamiltonian, Operator):
        integrals = to_electronic_integrals(hamiltonian, atol=atol)
    else:
        raise TypeError(
            "Expected a symbolic mbkit.operator.Operator or an ElectronicIntegrals instance."
        )

    h1e = np.asarray(integrals.one_body, dtype=np.complex128).copy()
    g2e = np.zeros_like(integrals.two_body, dtype=np.complex128)

    for p, q, r, s in np.argwhere(np.abs(integrals.two_body) > atol):
        g2e[p, s, q, r] += 2.0 * integrals.two_body[p, q, r, s]

    return QCTensorCompilation(
        space=integrals.space,
        constant_shift=integrals.constant,
        h1e=h1e,
        g2e=g2e,
    )
