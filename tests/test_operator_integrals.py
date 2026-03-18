import numpy as np
import pytest

from mbkit import ElectronicIntegrals, ElectronicSpace, LineLattice, to_electronic_integrals
from mbkit.operator import UnsupportedTransformError, to_qc_tensors


def test_electronic_integrals_round_trip_matches_symbolic_operator():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    operator = space.hubbard(t=1.0, U=4.0) + 0.5

    integrals = to_electronic_integrals(operator)
    rebuilt = integrals.to_operator()

    original_compiled = to_qc_tensors(operator)
    rebuilt_compiled = to_qc_tensors(rebuilt)

    assert np.isclose(integrals.constant, 0.5)
    assert np.allclose(rebuilt_compiled.h1e, original_compiled.h1e)
    assert np.allclose(rebuilt_compiled.g2e, original_compiled.g2e)
    assert np.isclose(rebuilt_compiled.constant_shift, original_compiled.constant_shift)


def test_electronic_integrals_from_raw_validates_shapes():
    space = ElectronicSpace(num_sites=2)
    with pytest.raises(ValueError, match="one_body"):
        ElectronicIntegrals.from_raw(space, one_body=np.zeros((3, 3)))


def test_to_electronic_integrals_rejects_pairing_terms():
    space = ElectronicSpace(LineLattice(2, boundary="open"), orbitals=["a"])
    pairing = space.create(0, spin="up") @ space.create(1, spin="down")

    with pytest.raises(UnsupportedTransformError, match="number-conserving one-body and two-body terms"):
        to_electronic_integrals(pairing)
