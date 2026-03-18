"""Backend-neutral symbolic fermionic algebra for `mbkit`.

This module deliberately stays at the second-quantized algebra layer. It knows
about fermionic anticommutation rules and canonical normal ordering, but it does
not silently assume a specific physical Hamiltonian class such as "quadratic",
"two-body electronic", or "density-density". Those more structured views live in
separate containers like `ElectronicIntegrals` and `QuadraticHamiltonian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Number
from typing import Iterator, Literal, Mapping

import numpy as np


@dataclass(frozen=True)
class Ladder:
    """A single fermionic ladder operator acting on one spin-orbital mode."""

    mode: int
    action: Literal["create", "destroy"]


@dataclass(frozen=True)
class Term:
    """An ordered product of ladder operators."""

    factors: tuple[Ladder, ...]


def _sort_key(ladder: Ladder) -> tuple[int, int]:
    action_key = 0 if ladder.action == "create" else 1
    return (action_key, -ladder.mode)


def _is_out_of_order(left: Ladder, right: Ladder) -> bool:
    if left.action == right.action:
        if left.mode == right.mode:
            return False
        return left.mode < right.mode
    return left.action == "destroy" and right.action == "create"


def _swap_sign(left: Ladder, right: Ladder) -> complex:
    if left.action == "destroy" and right.action == "create":
        return -1.0
    if left.action == right.action:
        return -1.0
    return 1.0


@lru_cache(maxsize=None)
def _normal_order_word(word: tuple[Ladder, ...]) -> tuple[tuple[tuple[Ladder, ...], complex], ...]:
    """Return the fermionic normal-ordered expansion of one operator word.

    This is the heart of the symbolic layer: it applies CAR exactly, including
    contractions like `a a^dagger = 1 - a^dagger a`, and returns all resulting
    terms plus their signs.
    """
    for idx in range(len(word) - 1):
        left = word[idx]
        right = word[idx + 1]

        if left.mode == right.mode and left.action == right.action:
            return tuple()

        if _is_out_of_order(left, right):
            swapped = word[:idx] + (right, left) + word[idx + 2 :]
            pieces = {tuple(term): coeff * _swap_sign(left, right) for term, coeff in _normal_order_word(swapped)}

            if left.action == "destroy" and right.action == "create" and left.mode == right.mode:
                contracted = word[:idx] + word[idx + 2 :]
                for term, coeff in _normal_order_word(contracted):
                    pieces[tuple(term)] = pieces.get(tuple(term), 0.0) + coeff

            return tuple((term, coeff) for term, coeff in pieces.items() if abs(coeff) > 1e-15)

    return ((word, 1.0),)


class Operator:
    """Backend-neutral symbolic fermionic operator.

    Design choice:
    the constructor eagerly normal-orders and merges equivalent terms. That gives
    `mbkit` a canonical symbolic representation without forcing users to call a
    separate normalization pass after every algebraic operation.
    """

    def __init__(
        self,
        space,
        *,
        constant: complex = 0.0,
        terms: Mapping[Term, complex] | None = None,
    ) -> None:
        self._space = space
        self._constant = complex(constant)
        reduced: dict[Term, complex] = {}

        if terms is not None:
            for term, coeff in terms.items():
                for ordered_factors, ordered_coeff in _normal_order_word(term.factors):
                    total_coeff = complex(coeff) * complex(ordered_coeff)
                    if not ordered_factors:
                        self._constant += total_coeff
                        continue
                    ordered_term = Term(tuple(ordered_factors))
                    reduced[ordered_term] = reduced.get(ordered_term, 0.0) + total_coeff

        self._terms = {term: coeff for term, coeff in reduced.items() if abs(coeff) > 1e-15}

    @property
    def space(self):
        return self._space

    @property
    def constant(self) -> complex:
        return self._constant

    @property
    def terms(self) -> Mapping[Term, complex]:
        return self._terms

    @classmethod
    def zero(cls, space) -> "Operator":
        return cls(space)

    @classmethod
    def identity(cls, space, coeff: complex = 1.0) -> "Operator":
        return cls(space, constant=coeff)

    def iter_terms(self) -> Iterator[tuple[Term, complex]]:
        return iter(self._terms.items())

    def _sorted_terms(self) -> tuple[tuple[Term, complex], ...]:
        return tuple(
            sorted(
                self._terms.items(),
                key=lambda item: (
                    len(item[0].factors),
                    tuple(_sort_key(factor) for factor in item[0].factors),
                ),
            )
        )

    def simplify(self, *, atol: float = 1e-12) -> "Operator":
        """Drop numerically tiny coefficients and return a cleaned copy."""
        return Operator(
            self.space,
            constant=0.0 if abs(self.constant) <= atol else self.constant,
            terms={term: coeff for term, coeff in self._terms.items() if abs(coeff) > atol},
        )

    def compress(self, *, atol: float = 1e-12) -> "Operator":
        """Alias for :meth:`simplify` for users coming from other packages."""
        return self.simplify(atol=atol)

    def adjoint(self) -> "Operator":
        """Return the Hermitian adjoint of the operator."""
        terms = {}
        for term, coeff in self.iter_terms():
            reversed_factors = []
            for ladder in reversed(term.factors):
                action = "destroy" if ladder.action == "create" else "create"
                reversed_factors.append(Ladder(ladder.mode, action))
            terms[Term(tuple(reversed_factors))] = np.conjugate(coeff)
        return Operator(self.space, constant=np.conjugate(self.constant), terms=terms)

    def is_hermitian(self, *, atol: float = 1e-12) -> bool:
        """Check Hermiticity in the canonical symbolic representation."""
        diff = (self - self.adjoint()).simplify(atol=atol)
        return abs(diff.constant) <= atol and not diff.terms

    def normal_ordered(self) -> "Operator":
        """Return a canonically normal-ordered copy.

        Since `Operator` is already normalized on construction, this mainly
        serves as an explicit readability hook in calling code.
        """
        return Operator(self.space, constant=self.constant, terms=self._terms)

    def serialize(self) -> dict[str, object]:
        """Return a stable, JSON-friendly description for debugging and tests."""
        def _serialize_coeff(value: complex) -> tuple[float, float]:
            scalar = complex(value)
            return (float(scalar.real), float(scalar.imag))

        return {
            "space": {
                "num_sites": self.space.num_sites,
                "orbitals": tuple(self.space.orbitals),
                "spins": tuple(self.space.spins),
            },
            "constant": _serialize_coeff(self.constant),
            "terms": tuple(
                {
                    "coefficient": _serialize_coeff(coeff),
                    "factors": tuple((factor.mode, factor.action) for factor in term.factors),
                }
                for term, coeff in self._sorted_terms()
            ),
        }

    def term_count(self) -> int:
        return len(self._terms)

    def body_rank(self) -> int:
        """Return the highest operator body-rank present in the symbolic sum."""
        if not self._terms:
            return 0
        return max(len(term.factors) // 2 if len(term.factors) % 2 == 0 else len(term.factors) for term in self._terms)

    def particle_number_changes(self) -> frozenset[int]:
        """Return all particle-number changes represented in the operator."""
        changes = set()
        if abs(self.constant) > 1e-15:
            changes.add(0)
        for term in self._terms:
            delta = 0
            for ladder in term.factors:
                delta += 1 if ladder.action == "create" else -1
            changes.add(delta)
        return frozenset(changes)

    def spin_changes(self) -> frozenset[int]:
        """Return all spin-projection changes represented in the operator."""
        changes = set()
        if abs(self.constant) > 1e-15:
            changes.add(0)
        for term in self._terms:
            delta = 0
            for ladder in term.factors:
                mode = self.space.unpack_mode(ladder.mode)
                spin_sign = 1 if mode.spin == "up" else -1
                delta += spin_sign if ladder.action == "create" else -spin_sign
            changes.add(delta)
        return frozenset(changes)

    def equiv(self, other, *, atol: float = 1e-12) -> bool:
        """Check symbolic equivalence up to numerical tolerance."""
        if not isinstance(other, Operator):
            return False
        same_space = (
            self.space is other.space
            or (
                self.space.num_sites == other.space.num_sites
                and tuple(self.space.orbitals) == tuple(other.space.orbitals)
                and tuple(self.space.spins) == tuple(other.space.spins)
            )
        )
        if not same_space:
            return False
        diff = (self - other).simplify(atol=atol)
        return abs(diff.constant) <= atol and not diff.terms

    def commutator(self, other) -> "Operator":
        """Return the commutator `[self, other]`."""
        if not isinstance(other, Operator):
            return NotImplemented
        return (self @ other) - (other @ self)

    def anticommutator(self, other) -> "Operator":
        """Return the anticommutator `{self, other}`."""
        if not isinstance(other, Operator):
            return NotImplemented
        return (self @ other) + (other @ self)

    def __add__(self, other) -> "Operator":
        if isinstance(other, Number):
            return Operator(self.space, constant=self.constant + other, terms=self._terms)
        if isinstance(other, Operator):
            if self.space is not other.space:
                raise ValueError("Cannot add operators from different ElectronicSpace objects.")
            terms = dict(self._terms)
            for term, coeff in other.iter_terms():
                terms[term] = terms.get(term, 0.0) + coeff
            return Operator(self.space, constant=self.constant + other.constant, terms=terms)
        return NotImplemented

    def __radd__(self, other) -> "Operator":
        if isinstance(other, Number):
            return self + other
        return NotImplemented

    def __sub__(self, other) -> "Operator":
        if isinstance(other, Number):
            return self + (-other)
        if isinstance(other, Operator):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other) -> "Operator":
        if isinstance(other, Number):
            return (-self) + other
        return NotImplemented

    def __neg__(self) -> "Operator":
        return (-1) * self

    def __mul__(self, scalar) -> "Operator":
        if not isinstance(scalar, Number):
            return NotImplemented
        return Operator(
            self.space,
            constant=self.constant * scalar,
            terms={term: coeff * scalar for term, coeff in self.iter_terms()},
        )

    def __rmul__(self, scalar) -> "Operator":
        return self * scalar

    def __truediv__(self, scalar) -> "Operator":
        if not isinstance(scalar, Number):
            return NotImplemented
        return self * (1 / scalar)

    def __matmul__(self, other) -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        if self.space is not other.space:
            raise ValueError("Cannot multiply operators from different ElectronicSpace objects.")

        result_constant = self.constant * other.constant
        result_terms: dict[Term, complex] = {}

        if abs(self.constant) > 1e-15:
            for term, coeff in other.iter_terms():
                result_terms[term] = result_terms.get(term, 0.0) + self.constant * coeff

        if abs(other.constant) > 1e-15:
            for term, coeff in self.iter_terms():
                result_terms[term] = result_terms.get(term, 0.0) + other.constant * coeff

        for left_term, left_coeff in self.iter_terms():
            for right_term, right_coeff in other.iter_terms():
                word = left_term.factors + right_term.factors
                for ordered_factors, ordered_coeff in _normal_order_word(word):
                    coeff = left_coeff * right_coeff * ordered_coeff
                    if not ordered_factors:
                        result_constant += coeff
                        continue
                    term = Term(tuple(ordered_factors))
                    result_terms[term] = result_terms.get(term, 0.0) + coeff

        return Operator(self.space, constant=result_constant, terms=result_terms)

    def __repr__(self) -> str:
        pieces = []
        if abs(self.constant) > 1e-15:
            pieces.append(f"{self.constant:+}")
        for term, coeff in self._sorted_terms():
            labels = []
            for ladder in term.factors:
                mode = self.space.unpack_mode(ladder.mode)
                action = "†" if ladder.action == "create" else ""
                labels.append(f"a{action}[{mode.site},{mode.orbital},{mode.spin}]")
            pieces.append(f"{coeff:+} {' '.join(labels)}")
        return " ".join(pieces) if pieces else "0"
