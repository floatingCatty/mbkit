from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operator import Ladder, Operator, Term


def _validate_square_matrix(name: str, array: np.ndarray, nmode: int) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.complex128)
    if matrix.shape != (nmode, nmode):
        raise ValueError(f"{name} must have shape ({nmode}, {nmode}), got {matrix.shape}.")
    return matrix


def _validate_rank4_tensor(name: str, array: np.ndarray, nmode: int) -> np.ndarray:
    tensor = np.asarray(array, dtype=np.complex128)
    if tensor.shape != (nmode, nmode, nmode, nmode):
        raise ValueError(
            f"{name} must have shape ({nmode}, {nmode}, {nmode}, {nmode}), got {tensor.shape}."
        )
    return tensor


@dataclass(frozen=True)
class ElectronicIntegrals:
    """Structured constant, one-body, and two-body spin-orbital coefficients."""

    space: object
    constant: complex = 0.0
    one_body: np.ndarray | None = None
    two_body: np.ndarray | None = None

    def __post_init__(self) -> None:
        nmode = self.space.num_spin_orbitals
        one_body = (
            np.zeros((nmode, nmode), dtype=np.complex128)
            if self.one_body is None
            else _validate_square_matrix("one_body", self.one_body, nmode)
        )
        two_body = (
            np.zeros((nmode, nmode, nmode, nmode), dtype=np.complex128)
            if self.two_body is None
            else _validate_rank4_tensor("two_body", self.two_body, nmode)
        )

        object.__setattr__(self, "constant", complex(self.constant))
        object.__setattr__(self, "one_body", one_body.copy())
        object.__setattr__(self, "two_body", two_body.copy())

    @classmethod
    def zeros(cls, space) -> "ElectronicIntegrals":
        return cls(space)

    @classmethod
    def from_raw(
        cls,
        space,
        *,
        one_body: np.ndarray,
        two_body: np.ndarray | None = None,
        constant: complex = 0.0,
    ) -> "ElectronicIntegrals":
        return cls(space, constant=constant, one_body=one_body, two_body=two_body)

    def to_operator(self, *, atol: float = 1e-12) -> Operator:
        terms: dict[Term, complex] = {}
        nmode = self.space.num_spin_orbitals

        for p in range(nmode):
            for q in range(nmode):
                coeff = self.one_body[p, q]
                if abs(coeff) <= atol:
                    continue
                term = Term((Ladder(p, "create"), Ladder(q, "destroy")))
                terms[term] = terms.get(term, 0.0) + coeff

        for p in range(nmode):
            for q in range(nmode):
                for r in range(nmode):
                    for s in range(nmode):
                        coeff = self.two_body[p, q, r, s]
                        if abs(coeff) <= atol:
                            continue
                        term = Term(
                            (
                                Ladder(p, "create"),
                                Ladder(q, "create"),
                                Ladder(r, "destroy"),
                                Ladder(s, "destroy"),
                            )
                        )
                        terms[term] = terms.get(term, 0.0) + coeff

        return Operator(self.space, constant=self.constant, terms=terms).simplify(atol=atol)
