"""Helpers for optional dependencies."""


def require_dependency(backend: str, extra: str, exc: Exception | None) -> None:
    if exc is None:
        return
    raise ImportError(
        f"{backend} requires the optional '{extra}' dependencies. "
        f"Install them with `pip install \"mbkit[{extra}]\"`."
    ) from exc
