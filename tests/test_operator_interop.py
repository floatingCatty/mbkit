import importlib

import pytest

from mbkit import ElectronicIntegrals, ElectronicSpace, chain
from mbkit.operator import interop


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def test_openfermion_interop_smoke():
    space = chain(1, orbitals=["a"])
    operator = space.number(0, spin="up")

    if _module_available("openfermion"):
        converted = interop.to_openfermion(operator)
        rebuilt = interop.from_openfermion(space, converted)
        assert rebuilt.equiv(operator)
    else:
        with pytest.raises(ImportError, match="interop"):
            interop.to_openfermion(operator)


def test_qiskit_nature_interop_smoke():
    space = chain(1, orbitals=["a"])
    integrals = ElectronicIntegrals.from_restricted(
        space,
        constant=0.25,
        one_body=[[1.0]],
        two_body=[[[[0.5]]]],
    )

    if _module_available("qiskit_nature"):
        converted = interop.to_qiskit_electronic_integrals(integrals)
        rebuilt = interop.from_qiskit_electronic_integrals(space, converted)
        assert rebuilt.equiv(integrals)
    else:
        with pytest.raises(ImportError, match="interop"):
            interop.to_qiskit_electronic_integrals(integrals)
