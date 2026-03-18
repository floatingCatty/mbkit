from __future__ import annotations

import numpy as np

from ..integrals import ElectronicIntegrals
from ..operator import Ladder, Operator, Term
from ..transforms import to_electronic_integrals

try:
    from openfermion import FermionOperator, InteractionOperator

    _OPENFERMION_IMPORT_ERROR = None
except ImportError as exc:
    FermionOperator = None
    InteractionOperator = None
    _OPENFERMION_IMPORT_ERROR = exc


def _require_openfermion() -> None:
    if FermionOperator is None or InteractionOperator is None:
        raise ImportError(
            "OpenFermion interop requires the optional `interop` dependencies. "
            "Install them with `pip install -e \".[interop]\"`."
        ) from _OPENFERMION_IMPORT_ERROR


def to_openfermion(operator: Operator, *, atol: float = 1e-12):
    _require_openfermion()

    result = FermionOperator()
    if abs(operator.constant) > atol:
        result += FermionOperator((), operator.constant)
    for term, coeff in operator.iter_terms():
        if abs(coeff) <= atol:
            continue
        word = tuple((factor.mode, 1 if factor.action == "create" else 0) for factor in term.factors)
        result += FermionOperator(word, coeff)
    return result


def from_openfermion(space, fermion_operator, *, atol: float = 1e-12) -> Operator:
    _require_openfermion()

    terms: dict[Term, complex] = {}
    constant = 0.0 + 0.0j
    for word, coeff in fermion_operator.terms.items():
        if abs(coeff) <= atol:
            continue
        if not word:
            constant += coeff
            continue
        term = Term(
            tuple(
                Ladder(int(index), "create" if int(action) == 1 else "destroy")
                for index, action in word
            )
        )
        terms[term] = terms.get(term, 0.0) + coeff
    return Operator(space, constant=constant, terms=terms).simplify(atol=atol)


def to_openfermion_interaction_operator(hamiltonian, *, atol: float = 1e-12):
    _require_openfermion()

    integrals = hamiltonian if isinstance(hamiltonian, ElectronicIntegrals) else to_electronic_integrals(hamiltonian, atol=atol)
    if integrals.batch_shape:
        raise ValueError("OpenFermion InteractionOperator export requires an unbatched ElectronicIntegrals object.")
    return InteractionOperator(
        complex(integrals.constant),
        np.asarray(integrals.one_body, dtype=np.complex128),
        np.asarray(integrals.two_body, dtype=np.complex128),
    )


def from_openfermion_interaction_operator(space, interaction_operator) -> ElectronicIntegrals:
    _require_openfermion()
    return ElectronicIntegrals.from_raw(
        space,
        constant=complex(interaction_operator.constant),
        one_body=np.asarray(interaction_operator.one_body_tensor, dtype=np.complex128),
        two_body=np.asarray(interaction_operator.two_body_tensor, dtype=np.complex128),
    )
