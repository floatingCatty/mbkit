"""Helpers for optional solver backends."""


def require_dependency(backend: str, extra: str, exc: Exception | None) -> None:
    """Raise a clear ImportError when an optional backend is unavailable."""
    if exc is None:
        return

    raise ImportError(
        f"{backend} requires the optional '{extra}' dependencies. "
        f"Install them with `pip install \"mbkit[{extra}]\"`."
    ) from exc
