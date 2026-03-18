"""Shared base classes for solver backends and public façades."""

from __future__ import annotations

from .capabilities import BackendCapabilities
from .registry import available_solver_backends, get_backend_spec, get_solver_backend_class


class SolverBackend:
    """Metadata-only base class for concrete solver backends."""

    solver_family = ""
    backend_name = ""
    capabilities = BackendCapabilities()


class SolverFacade:
    """Thin public wrapper that delegates to a registered backend.

    The façade keeps the user-facing solver names stable while allowing the
    implementation underneath to be swapped or expanded later.
    """

    solver_family = ""
    default_backend: str | None = None

    def __init__(self, *args, backend: str | None = None, **kwargs) -> None:
        backend_cls = get_solver_backend_class(self.solver_family, backend or self.default_backend)
        self._backend = backend_cls(*args, **kwargs)
        self._backend_spec = get_backend_spec(self.solver_family, backend or self.default_backend)

    @property
    def backend(self):
        """Return the concrete backend instance."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Return the concrete backend name selected from the registry."""
        return self._backend_spec.name

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the selected backend's capability metadata."""
        return getattr(self._backend, "capabilities", BackendCapabilities())

    @classmethod
    def available_backends(cls):
        """Return the available backend names for this solver family."""
        return available_solver_backends(cls.solver_family)

    def unwrap_backend(self):
        """Return the concrete backend instance.

        This is useful when advanced users need backend-specific methods beyond
        the stable `mbkit` façade surface.
        """
        return self._backend

    def solve(self, *args, **kwargs):
        """Delegate `solve()` and return the façade for fluent usage."""
        self._backend.solve(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(backend={self.backend_name!r}, "
            f"family={self.solver_family!r})"
        )

