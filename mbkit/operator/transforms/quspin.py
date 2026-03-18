from __future__ import annotations

from dataclasses import dataclass

class UnsupportedTransformError(RuntimeError):
    """Raised when a symbolic operator cannot be lowered to a target backend."""


@dataclass(frozen=True)
class QuSpinCompilation:
    static_list: list[list[object]]
    constant_shift: complex = 0.0
    num_spatial_orbitals: int = 0


def _to_quspin_mode(space, mode_index: int) -> int:
    mode = space.unpack_mode(mode_index)
    spatial_index = mode.site * space.num_orbitals_per_site + mode.orbital_index
    if mode.spin == "up":
        return spatial_index
    if mode.spin == "down":
        return spatial_index + space.num_spatial_orbitals
    raise UnsupportedTransformError(f"Unsupported spin label {mode.spin!r}.")


def to_quspin_operator(operator) -> QuSpinCompilation:
    if operator.space.num_spins != 2 or tuple(operator.space.spins) != ("up", "down"):
        raise UnsupportedTransformError("QuSpin lowering currently requires a two-spin ElectronicSpace.")

    grouped_terms: dict[str, list[list[complex | int]]] = {}
    for term, coeff in operator.iter_terms():
        opstr = "".join("+" if ladder.action == "create" else "-" for ladder in term.factors)
        indices = [_to_quspin_mode(operator.space, ladder.mode) for ladder in term.factors]
        grouped_terms.setdefault(opstr, []).append([coeff, *indices])

    static_list = [[opstr, interactions] for opstr, interactions in grouped_terms.items()]
    return QuSpinCompilation(
        static_list=static_list,
        constant_shift=operator.constant,
        num_spatial_orbitals=operator.space.num_spatial_orbitals,
    )
