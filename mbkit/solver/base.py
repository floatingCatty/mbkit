"""Shared base classes for solver backends and public façades."""

from __future__ import annotations

from dataclasses import asdict

from .capabilities import BackendCapabilities
from .registry import available_solver_backends, get_backend_spec, get_solver_backend_class
from ..utils.solver import coerce_symbolic_operator


class SolverBackend:
    """Shared base class for concrete solver backends.

    Design choice:
    the public runtime object in `mbkit` remains the solver itself, not a
    separate task or result wrapper. To keep that solver-centric API flexible,
    backends can expose lightweight generic property hooks such as `expect()`
    and `diagnostics()` in addition to shorthand methods like `energy()` or
    `rdm1()`.

    Developer contract:
    subclasses implement the actual numerical work while this base class
    supplies the common helper surface exposed through `SolverFacade`.
    Backend authors should implement `solve()` plus whichever property helpers
    they support. Implement `_expectation()` to opt into the generic
    `expect()` / `expect_value()` API, and `_require_solution()` to provide a
    consistent "call solve() first" guard across all observables.
    """

    solver_family = ""
    backend_name = ""
    capabilities = BackendCapabilities()

    @property
    def native(self):
        """Return the backend-native runtime object.

        For the current direct backends this is the backend instance itself.
        More specialized backends can override this if they want to surface a
        wrapped engine object instead.
        """
        return self

    def expect(self, operator, *, stats: bool = False):
        """Evaluate the expectation value of a symbolic operator.

        Deterministic backends return a scalar by default. Passing
        `stats=True` returns a small dictionary-shaped statistics payload that
        future stochastic backends can enrich with uncertainties.

        Backends opt into this generic helper by implementing
        `_expectation(operator)`. If `_require_solution()` exists, it is called
        before the operator is evaluated so this method follows the same
        lifecycle contract as backend-specific helpers such as `energy()`.
        """
        if not hasattr(self, "_expectation"):
            raise NotImplementedError(
                f"{type(self).__name__} does not expose generic operator expectation values."
            )
        require_solution = getattr(self, "_require_solution", None)
        if callable(require_solution):
            require_solution()
        value = getattr(self, "_expectation")(coerce_symbolic_operator(operator))
        if not stats:
            return value
        return {
            "mean": value,
            "stderr": 0.0,
            "variance": 0.0,
            "n_samples": None,
            "kind": "deterministic",
            "backend": self.backend_name,
        }

    def expect_value(self, operator):
        """Return the scalar expectation value of a symbolic operator."""
        return self.expect(operator, stats=False)

    def diagnostics(self) -> dict[str, object]:
        """Return structured backend diagnostics for the last run.

        Backends can extend this dictionary with method-specific fields while
        preserving these common keys for tooling and debugging.
        """
        return {
            "backend": self.backend_name,
            "family": self.solver_family,
            "preferred_problem_type": self.capabilities.preferred_problem_type,
            "capabilities": asdict(self.capabilities),
        }

    def available_properties(self) -> tuple[str, ...]:
        """List the property helpers exposed by this backend.

        The returned names describe the stable runtime surface that users can
        call on the public solver façade after `solve()`.
        """
        properties: list[str] = ["energy", "diagnostics", "available_properties"]
        if hasattr(self, "_expectation"):
            properties.extend(["expect", "expect_value"])
        for name in (
            "rdm1",
            "rdm2",
            "docc",
            "s2",
            "correlation",
            "entanglement_entropy",
            "greens_function",
        ):
            if callable(getattr(self, name, None)):
                properties.append(name)
        return tuple(dict.fromkeys(properties))


class SolverFacade:
    """Thin public wrapper that delegates to a registered backend.

    The façade keeps the user-facing solver names stable while allowing the
    implementation underneath to be swapped or expanded later.

    User model:
    instantiate a public solver class such as `EDSolver` or `FCISolver`, call
    `solve(...)`, then use the shared helper methods defined here
    (`expect()`, `expect_value()`, `diagnostics()`,
    `available_properties()`) together with any backend-specific observable
    helpers delegated via `__getattr__`.

    Developer model:
    concrete public solver classes usually only set `solver_family` and a
    default backend. They inherit the common runtime API from this class so the
    user-facing surface stays aligned across solver families.
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
    def native(self):
        """Return the backend-native runtime object."""
        return getattr(self._backend, "native", self._backend)

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
        """Return the available backend names for this solver family.

        These are the strings accepted by the façade constructor's `backend=`
        argument.
        """
        return available_solver_backends(cls.solver_family)

    def unwrap_backend(self):
        """Return the concrete backend instance.

        This is useful when advanced users need backend-specific methods beyond
        the stable `mbkit` façade surface.
        """
        return self._backend

    def solve(self, *args, **kwargs):
        """Delegate `solve()` and return the façade for fluent usage.

        This keeps the public runtime object stable: `solver.solve(...)`
        returns the same façade instance that exposes shared helpers and any
        backend-specific observable methods.
        """
        self._backend.solve(*args, **kwargs)
        return self

    def expect(self, operator, *, stats: bool = False):
        """Evaluate a generic operator expectation directly from the solver.

        This is the façade-level entrypoint for backends that implement the
        generic expectation contract.
        """
        return self._backend.expect(operator, stats=stats)

    def expect_value(self, operator):
        """Return the scalar expectation value of a symbolic operator."""
        return self._backend.expect_value(operator)

    def diagnostics(self):
        """Return structured backend diagnostics for the current solve.

        The common keys come from `SolverBackend.diagnostics()`, and backends
        may add method- or engine-specific fields.
        """
        return self._backend.diagnostics()

    def available_properties(self):
        """Return the property helpers supported by the selected backend.

        This is useful when code wants to discover whether helpers such as
        `expect()`, `rdm1()`, or `docc()` are available before calling them.
        """
        return self._backend.available_properties()

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(backend={self.backend_name!r}, "
            f"family={self.solver_family!r})"
        )
