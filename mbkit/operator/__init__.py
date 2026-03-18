"""Public operator API for `mbkit`.

For normal usage, start with :func:`chain`, :func:`ladder`, :func:`square`,
:func:`honeycomb`, or another lattice helper, build Hamiltonians directly on
the resulting :class:`ElectronicSpace`, and use the transform or solver layers
only when needed.
"""

from . import interop, transforms
from .integrals import ElectronicIntegrals
from .lattice import (
    Bond,
    CubicLattice,
    GeneralLattice,
    HoneycombLattice,
    KagomeLattice,
    LadderLattice,
    Lattice,
    LineLattice,
    RectangularLattice,
    SquareLattice,
    TriangularLattice,
)
from .operator import Ladder, Operator, Term
from .quadratic import QuadraticHamiltonian
from .space import (
    ElectronicSpace,
    Mode,
    chain,
    cubic,
    general,
    honeycomb,
    kagome,
    ladder,
    rectangular,
    square,
    triangular,
)
from .transforms import (
    QCTensorCompilation,
    QuSpinCompilation,
    UnsupportedTransformError,
    to_electronic_integrals,
    to_integrals,
    to_qc_tensors,
    to_quspin_operator,
)

__all__ = [
    "Bond",
    "CubicLattice",
    "ElectronicIntegrals",
    "ElectronicSpace",
    "GeneralLattice",
    "HoneycombLattice",
    "KagomeLattice",
    "Ladder",
    "LadderLattice",
    "Lattice",
    "LineLattice",
    "Mode",
    "Operator",
    "QCTensorCompilation",
    "QuadraticHamiltonian",
    "QuSpinCompilation",
    "RectangularLattice",
    "SquareLattice",
    "Term",
    "TriangularLattice",
    "UnsupportedTransformError",
    "chain",
    "cubic",
    "general",
    "honeycomb",
    "interop",
    "kagome",
    "ladder",
    "rectangular",
    "square",
    "to_electronic_integrals",
    "to_integrals",
    "to_qc_tensors",
    "to_quspin_operator",
    "transforms",
    "triangular",
]
