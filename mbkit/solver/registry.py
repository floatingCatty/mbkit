"""Lazy registry for solver backends.

The public solver names in `mbkit` should remain short and stable. Internally,
however, different packages want different Hamiltonian contracts and solve
different tasks. This registry gives us a clean way to map a user-facing solver
family such as `ed` or `dmrg` to one concrete backend implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class BackendSpec:
    """Registration record for one concrete solver backend."""

    family: str
    name: str
    module: str
    attribute: str
    description: str
    default: bool = False


_REGISTRY: dict[str, dict[str, BackendSpec]] = {
    "ed": {
        "quspin": BackendSpec(
            family="ed",
            name="quspin",
            module=".backends.quspin_ed",
            attribute="QuSpinEDBackend",
            description="Exact diagonalization backend built on QuSpin.",
            default=True,
        ),
    },
    "dmrg": {
        "block2": BackendSpec(
            family="dmrg",
            name="block2",
            module=".backends.block2_dmrg",
            attribute="Block2DMRGBackend",
            description="DMRG backend built on block2 / pyblock2.",
            default=True,
        ),
        "quimb": BackendSpec(
            family="dmrg",
            name="quimb",
            module=".backends.quimb_dmrg",
            attribute="QuimbDMRGBackend",
            description="Tensor-network DMRG backend built on quimb.",
            default=False,
        ),
        "tenpy": BackendSpec(
            family="dmrg",
            name="tenpy",
            module=".backends.tenpy_dmrg",
            attribute="TeNPyDMRGBackend",
            description="Tensor-network DMRG backend built on TeNPy.",
            default=False,
        ),
    },
    "qc": {
        "pyscf_fci": BackendSpec(
            family="qc",
            name="pyscf_fci",
            module=".backends.pyscf_fci",
            attribute="PySCFFCIBackend",
            description="PySCF backend for exact sector FCI on general spin-conserving Hamiltonians.",
            default=True,
        ),
        "pyscf_reference": BackendSpec(
            family="qc",
            name="pyscf_reference",
            module=".backends.pyscf_reference",
            attribute="PySCFReferenceBackend",
            description="PySCF backend for UHF-based approximate methods on UHF-compatible Hamiltonians.",
            default=False,
        ),
    },
}


def register_backend(spec: BackendSpec) -> None:
    """Register a new concrete backend implementation."""
    family_entries = _REGISTRY.setdefault(spec.family, {})
    family_entries[spec.name] = spec


def available_solver_backends(family: str | None = None):
    """Return the registered backend names.

    If `family` is omitted, a `{family: (backend_names...)}` mapping is returned.
    """
    if family is None:
        return {name: tuple(sorted(entries)) for name, entries in sorted(_REGISTRY.items())}
    try:
        return tuple(sorted(_REGISTRY[family]))
    except KeyError as exc:
        raise KeyError(f"Unknown solver family {family!r}.") from exc


def get_backend_spec(family: str, name: str | None = None) -> BackendSpec:
    """Return the registry spec for one backend.

    If `name` is omitted, the default backend for the family is returned.
    """
    try:
        family_entries = _REGISTRY[family]
    except KeyError as exc:
        raise KeyError(f"Unknown solver family {family!r}.") from exc

    if name is None:
        for spec in family_entries.values():
            if spec.default:
                return spec
        raise KeyError(f"Solver family {family!r} has no default backend.")

    try:
        return family_entries[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown backend {name!r} for solver family {family!r}. "
            f"Available backends: {sorted(family_entries)!r}."
        ) from exc


def get_solver_backend_class(family: str, name: str | None = None):
    """Import and return one concrete backend class."""
    spec = get_backend_spec(family, name=name)
    module = import_module(spec.module, __package__)
    return getattr(module, spec.attribute)
