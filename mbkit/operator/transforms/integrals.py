from __future__ import annotations

import numpy as np

from ..integrals import ElectronicIntegrals
from .quspin import UnsupportedTransformError


def to_electronic_integrals(operator, *, atol: float = 1e-12) -> ElectronicIntegrals:
    nmode = operator.space.num_spin_orbitals
    one_body = np.zeros((nmode, nmode), dtype=np.complex128)
    two_body = np.zeros((nmode, nmode, nmode, nmode), dtype=np.complex128)

    for term, coeff in operator.iter_terms():
        if abs(coeff) <= atol:
            continue

        factors = term.factors
        actions = tuple(ladder.action for ladder in factors)

        if len(factors) == 2 and actions == ("create", "destroy"):
            one_body[factors[0].mode, factors[1].mode] += coeff
            continue

        if len(factors) == 4 and actions == ("create", "create", "destroy", "destroy"):
            two_body[
                factors[0].mode,
                factors[1].mode,
                factors[2].mode,
                factors[3].mode,
            ] += coeff
            continue

        raise UnsupportedTransformError(
            "ElectronicIntegrals only supports constants plus normal-ordered "
            "number-conserving one-body and two-body terms; "
            f"encountered term with actions {actions!r}."
        )

    return ElectronicIntegrals(
        operator.space,
        constant=operator.constant,
        one_body=one_body,
        two_body=two_body,
    )


def to_integrals(operator, *, atol: float = 1e-12) -> ElectronicIntegrals:
    return to_electronic_integrals(operator, atol=atol)
