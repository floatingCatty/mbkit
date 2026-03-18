from __future__ import annotations

import numpy as np

from ..integrals import ElectronicIntegrals
from ..operator import Ladder, Operator, Term
from ..quadratic import QuadraticHamiltonian

try:
    from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian as QiskitQuadraticHamiltonian
    from qiskit_nature.second_q.operators import (
        ElectronicIntegrals as QiskitElectronicIntegrals,
        FermionicOp,
        PolynomialTensor,
    )

    _QISKIT_NATURE_IMPORT_ERROR = None
except ImportError as exc:
    QiskitQuadraticHamiltonian = None
    QiskitElectronicIntegrals = None
    FermionicOp = None
    PolynomialTensor = None
    _QISKIT_NATURE_IMPORT_ERROR = exc


def _require_qiskit_nature() -> None:
    if (
        FermionicOp is None
        or QiskitElectronicIntegrals is None
        or PolynomialTensor is None
        or QiskitQuadraticHamiltonian is None
    ):
        raise ImportError(
            "Qiskit Nature interop requires the optional `interop` dependencies. "
            "Install them with `pip install -e \".[interop]\"`."
        ) from _QISKIT_NATURE_IMPORT_ERROR


def to_qiskit_fermionic_op(operator: Operator, *, atol: float = 1e-12):
    _require_qiskit_nature()

    data: dict[str, complex] = {}
    if abs(operator.constant) > atol:
        data[""] = operator.constant
    for term, coeff in operator.iter_terms():
        if abs(coeff) <= atol:
            continue
        label = " ".join(
            f"{'+_' if factor.action == 'create' else '-_'}{factor.mode}"
            for factor in term.factors
        )
        data[label] = data.get(label, 0.0) + coeff
    return FermionicOp(data, num_spin_orbitals=operator.space.num_spin_orbitals)


def from_qiskit_fermionic_op(space, fermionic_op, *, atol: float = 1e-12) -> Operator:
    _require_qiskit_nature()

    terms: dict[Term, complex] = {}
    constant = 0.0 + 0.0j
    for label, coeff in fermionic_op.items():
        if abs(coeff) <= atol:
            continue
        if label == "":
            constant += coeff
            continue
        factors = []
        for piece in label.split():
            action = "create" if piece.startswith("+_") else "destroy"
            factors.append(Ladder(int(piece[2:]), action))
        term = Term(tuple(factors))
        terms[term] = terms.get(term, 0.0) + coeff
    return Operator(space, constant=constant, terms=terms).simplify(atol=atol)


def to_qiskit_electronic_integrals(integrals: ElectronicIntegrals, *, atol: float = 1e-12):
    _require_qiskit_nature()
    if integrals.batch_shape:
        raise ValueError("Qiskit Nature ElectronicIntegrals export requires an unbatched ElectronicIntegrals object.")

    alpha_data: dict[str, object] = {}
    if abs(complex(integrals.constant)) > atol:
        alpha_data[""] = integrals.constant
    if np.any(np.abs(integrals.one_body_alpha) > atol):
        alpha_data["+-"] = np.asarray(integrals.one_body_alpha, dtype=np.complex128)
    if np.any(np.abs(integrals.two_body_aa) > atol):
        alpha_data["++--"] = np.asarray(integrals.two_body_aa, dtype=np.complex128)

    restricted_like = (
        np.allclose(integrals.one_body_beta, integrals.one_body_alpha, atol=atol)
        and np.allclose(integrals.two_body_bb, integrals.two_body_aa, atol=atol)
        and np.allclose(integrals.two_body_ab, integrals.two_body_aa, atol=atol)
    )
    if restricted_like:
        return QiskitElectronicIntegrals(
            PolynomialTensor(alpha_data, validate=False),
            validate=False,
        )

    beta_data: dict[str, object] = {}
    if np.any(np.abs(integrals.one_body_beta) > atol):
        beta_data["+-"] = np.asarray(integrals.one_body_beta, dtype=np.complex128)
    if np.any(np.abs(integrals.two_body_bb) > atol):
        beta_data["++--"] = np.asarray(integrals.two_body_bb, dtype=np.complex128)

    beta_alpha = None
    if np.any(np.abs(integrals.two_body_ab) > atol):
        beta_alpha = PolynomialTensor(
            {"++--": np.transpose(integrals.two_body_ab, (2, 3, 0, 1))},
            validate=False,
        )

    return QiskitElectronicIntegrals(
        PolynomialTensor(alpha_data, validate=False),
        PolynomialTensor(beta_data, validate=False) if beta_data else None,
        beta_alpha,
        validate=False,
    )


def from_qiskit_electronic_integrals(space, electronic_integrals) -> ElectronicIntegrals:
    _require_qiskit_nature()

    nspatial = space.num_spatial_orbitals
    zeros_1 = np.zeros((nspatial, nspatial), dtype=np.complex128)
    zeros_2 = np.zeros((nspatial, nspatial, nspatial, nspatial), dtype=np.complex128)

    alpha = electronic_integrals.alpha
    beta = electronic_integrals.beta
    beta_alpha = electronic_integrals.beta_alpha

    constant = complex(np.asarray(alpha.get("", 0.0), dtype=np.complex128).item())
    one_a = np.asarray(alpha.get("+-", zeros_1), dtype=np.complex128)
    two_aa = np.asarray(alpha.get("++--", zeros_2), dtype=np.complex128)

    if beta.is_empty() and beta_alpha.is_empty():
        return ElectronicIntegrals.from_restricted(
            space,
            constant=constant,
            one_body=one_a,
            two_body=two_aa,
        )

    one_b = np.asarray((alpha if beta.is_empty() else beta).get("+-", zeros_1), dtype=np.complex128)
    two_bb = np.asarray((alpha if beta.is_empty() else beta).get("++--", zeros_2), dtype=np.complex128)
    alpha_beta = electronic_integrals.alpha_beta
    two_ab = np.asarray(alpha_beta.get("++--", zeros_2), dtype=np.complex128)

    return ElectronicIntegrals.from_unrestricted(
        space,
        constant=constant,
        one_body_alpha=one_a,
        one_body_beta=one_b,
        two_body_aa=two_aa,
        two_body_ab=two_ab,
        two_body_bb=two_bb,
    )


def to_qiskit_quadratic_hamiltonian(hamiltonian, *, atol: float = 1e-12):
    _require_qiskit_nature()

    if isinstance(hamiltonian, QuadraticHamiltonian):
        quadratic = hamiltonian
    elif isinstance(hamiltonian, Operator):
        quadratic = QuadraticHamiltonian.from_operator(hamiltonian, atol=atol)
    elif isinstance(hamiltonian, ElectronicIntegrals):
        quadratic = QuadraticHamiltonian.from_operator(hamiltonian.to_operator(atol=atol), atol=atol)
    else:
        raise TypeError(
            "Expected a QuadraticHamiltonian, symbolic Operator, or ElectronicIntegrals instance."
        )

    return QiskitQuadraticHamiltonian(
        hermitian_part=np.asarray(quadratic.hermitian_part, dtype=np.complex128),
        antisymmetric_part=np.asarray(quadratic.pairing_part, dtype=np.complex128),
        constant=quadratic.constant,
        validate=False,
    )


def from_qiskit_quadratic_hamiltonian(space, quadratic_hamiltonian) -> QuadraticHamiltonian:
    _require_qiskit_nature()
    if int(quadratic_hamiltonian.register_length) != space.num_spin_orbitals:
        raise ValueError(
            "QuadraticHamiltonian register length does not match the supplied ElectronicSpace."
        )
    return QuadraticHamiltonian(
        space,
        constant=complex(quadratic_hamiltonian.constant),
        hermitian_part=np.asarray(quadratic_hamiltonian.hermitian_part, dtype=np.complex128),
        pairing_part=np.asarray(quadratic_hamiltonian.antisymmetric_part, dtype=np.complex128),
    )
