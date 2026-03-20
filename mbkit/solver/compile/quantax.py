"""Compilation helpers for Quantax-based solver backends."""

from __future__ import annotations

from dataclasses import dataclass

from ...operator.transforms import UnsupportedTransformError
from ...utils.solver import coerce_symbolic_operator


@dataclass(frozen=True)
class QuantaxCompilation:
    """Lowered operator data for Quantax/QuSpin-style backends."""

    op_list: list[list[object]]
    constant_shift: complex = 0.0
    num_spatial_orbitals: int = 0
    hopping_graph: tuple[tuple[int, int], ...] = ()


def _to_quantax_mode(space, mode_index: int) -> int:
    mode = space.unpack_mode(mode_index)
    spatial_index = mode.site * space.num_orbitals_per_site + mode.orbital_index
    if mode.spin == "up":
        return spatial_index
    if mode.spin == "down":
        return spatial_index + space.num_spatial_orbitals
    raise UnsupportedTransformError(f"Unsupported spin label {mode.spin!r}.")


def _to_spatial_orbital(space, mode_index: int) -> int:
    mode = space.unpack_mode(mode_index)
    return mode.site * space.num_orbitals_per_site + mode.orbital_index


def _term_particle_change(term) -> int:
    delta = 0
    for ladder in term.factors:
        delta += 1 if ladder.action == "create" else -1
    return delta


def _validate_quantax_operator(operator) -> None:
    space = operator.space
    if space.num_spins != 2 or tuple(space.spins) != ("up", "down"):
        raise UnsupportedTransformError(
            "Quantax NQS lowering currently requires a two-spin ElectronicSpace."
        )

    for term, _ in operator.iter_terms():
        if len(term.factors) % 2 != 0:
            raise UnsupportedTransformError(
                "Quantax NQS v1 only supports even-parity operator terms."
            )
        if _term_particle_change(term) != 0:
            raise UnsupportedTransformError(
                "Quantax NQS v1 only supports particle-number-conserving operators."
            )


def compile_quantax_hamiltonian(hamiltonian) -> QuantaxCompilation:
    """Compile a symbolic Hamiltonian into Quantax operator data."""
    operator = coerce_symbolic_operator(hamiltonian)
    _validate_quantax_operator(operator)

    grouped_terms: dict[str, list[list[complex | int]]] = {}
    hopping_graph: set[tuple[int, int]] = set()
    for term, coeff in operator.iter_terms():
        opstr = "".join("+" if ladder.action == "create" else "-" for ladder in term.factors)
        indices = [_to_quantax_mode(operator.space, ladder.mode) for ladder in term.factors]
        grouped_terms.setdefault(opstr, []).append([coeff, *indices])

        if len(term.factors) == 2 and opstr == "+-":
            left_mode = operator.space.unpack_mode(term.factors[0].mode)
            right_mode = operator.space.unpack_mode(term.factors[1].mode)
            if left_mode.spin == right_mode.spin:
                left_spatial = _to_spatial_orbital(operator.space, term.factors[0].mode)
                right_spatial = _to_spatial_orbital(operator.space, term.factors[1].mode)
                if left_spatial != right_spatial:
                    hopping_graph.add(tuple(sorted((left_spatial, right_spatial))))

    static_list: list[list[object]] = []
    for opstr in sorted(grouped_terms):
        interactions = sorted(
            grouped_terms[opstr],
            key=lambda item: tuple(int(index) for index in item[1:]),
        )
        static_list.append([opstr, interactions])

    return QuantaxCompilation(
        op_list=static_list,
        constant_shift=operator.constant,
        num_spatial_orbitals=operator.space.num_spatial_orbitals,
        hopping_graph=tuple(sorted(hopping_graph)),
    )
