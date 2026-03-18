"""Structured quadratic Hamiltonian container.

This module captures the common special case of a constant plus one-body and
pairing terms. It is intentionally separate from the general symbolic operator
layer so quadratic-specific assumptions remain explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operator import Ladder, Operator, Term


def _validate_square(name: str, array: np.ndarray, nmode: int) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.complex128)
    if matrix.shape != (nmode, nmode):
        raise ValueError(f"{name} must have shape ({nmode}, {nmode}), got {matrix.shape}.")
    return matrix


@dataclass(frozen=True)
class QuadraticHamiltonian:
    """Constant + Hermitian one-body + antisymmetric pairing data."""

    space: object
    constant: complex = 0.0
    hermitian_part: np.ndarray | None = None
    pairing_part: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the quadratic blocks against the space dimension."""
        nmode = self.space.num_spin_orbitals
        hermitian = (
            np.zeros((nmode, nmode), dtype=np.complex128)
            if self.hermitian_part is None
            else _validate_square("hermitian_part", self.hermitian_part, nmode)
        )
        pairing = (
            np.zeros((nmode, nmode), dtype=np.complex128)
            if self.pairing_part is None
            else _validate_square("pairing_part", self.pairing_part, nmode)
        )
        object.__setattr__(self, "constant", complex(self.constant))
        object.__setattr__(self, "hermitian_part", hermitian.copy())
        object.__setattr__(self, "pairing_part", pairing.copy())

    @property
    def num_modes(self) -> int:
        return self.space.num_spin_orbitals

    def conserves_particle_number(self, *, atol: float = 1e-12) -> bool:
        """Return `True` when no pairing block is present."""
        return np.allclose(self.pairing_part, 0.0, atol=atol)

    def equiv(self, other, *, atol: float = 1e-12) -> bool:
        same_space = (
            isinstance(other, QuadraticHamiltonian)
            and (
                self.space is other.space
                or (
                    self.space.num_sites == other.space.num_sites
                    and tuple(self.space.orbitals) == tuple(other.space.orbitals)
                    and tuple(self.space.spins) == tuple(other.space.spins)
                )
            )
        )
        return (
            same_space
            and abs(self.constant - other.constant) <= atol
            and np.allclose(self.hermitian_part, other.hermitian_part, atol=atol)
            and np.allclose(self.pairing_part, other.pairing_part, atol=atol)
        )

    def to_operator(self, *, atol: float = 1e-12) -> Operator:
        """Convert the structured quadratic data into a symbolic `Operator`."""
        terms: dict[Term, complex] = {}
        for p in range(self.num_modes):
            for q in range(self.num_modes):
                coeff = self.hermitian_part[p, q]
                if abs(coeff) > atol:
                    terms[Term((Ladder(p, "create"), Ladder(q, "destroy")))] = (
                        terms.get(Term((Ladder(p, "create"), Ladder(q, "destroy"))), 0.0) + coeff
                    )

                pairing_coeff = self.pairing_part[p, q]
                if abs(pairing_coeff) > atol:
                    terms[Term((Ladder(p, "create"), Ladder(q, "create")))] = (
                        terms.get(Term((Ladder(p, "create"), Ladder(q, "create"))), 0.0) + 0.5 * pairing_coeff
                    )
                    terms[Term((Ladder(q, "destroy"), Ladder(p, "destroy")))] = (
                        terms.get(Term((Ladder(q, "destroy"), Ladder(p, "destroy"))), 0.0)
                        - 0.5 * np.conjugate(pairing_coeff)
                    )
        return Operator(self.space, constant=self.constant, terms=terms).simplify(atol=atol)

    @classmethod
    def from_operator(cls, operator: Operator, *, atol: float = 1e-12) -> "QuadraticHamiltonian":
        """Extract quadratic data from a symbolic operator.

        This is strict by design: any term outside the quadratic sector raises,
        which helps keep solver-side assumptions explicit.
        """
        hermitian = np.zeros((operator.space.num_spin_orbitals, operator.space.num_spin_orbitals), dtype=np.complex128)
        pairing = np.zeros_like(hermitian)

        for term, coeff in operator.iter_terms():
            actions = tuple(factor.action for factor in term.factors)
            if len(term.factors) == 2 and actions == ("create", "destroy"):
                hermitian[term.factors[0].mode, term.factors[1].mode] += coeff
                continue
            if len(term.factors) == 2 and actions == ("create", "create"):
                left = term.factors[0].mode
                right = term.factors[1].mode
                pairing[left, right] += coeff
                pairing[right, left] -= coeff
                continue
            if len(term.factors) == 2 and actions == ("destroy", "destroy"):
                continue
            if abs(coeff) > atol:
                raise ValueError("QuadraticHamiltonian.from_operator() only supports quadratic terms.")

        return cls(operator.space, constant=operator.constant, hermitian_part=hermitian, pairing_part=pairing)
