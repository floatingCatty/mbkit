"""Structured one- and two-body coefficient containers.

`Operator` is the most general symbolic representation in `mbkit`. This module
provides the next layer up: a container specialized for constants plus normal-
ordered one-body and two-body spin-orbital tensors. That structured form is what
solver backends and interop bridges typically want.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np

from .operator import Ladder, Operator, Term


def _validate_square_matrix(name: str, array: np.ndarray, nmode: int) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.complex128)
    if matrix.shape[-2:] != (nmode, nmode):
        raise ValueError(f"{name} must have trailing shape ({nmode}, {nmode}), got {matrix.shape}.")
    return matrix


def _validate_rank4_tensor(name: str, array: np.ndarray, nmode: int) -> np.ndarray:
    tensor = np.asarray(array, dtype=np.complex128)
    if tensor.shape[-4:] != (nmode, nmode, nmode, nmode):
        raise ValueError(
            f"{name} must have trailing shape ({nmode}, {nmode}, {nmode}, {nmode}), got {tensor.shape}."
        )
    return tensor


def _copy_constant(value):
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        return complex(array.item())
    return array.copy()


def _batch_shape_of_constant(value) -> tuple[int, ...]:
    array = np.asarray(value, dtype=np.complex128)
    return () if array.ndim == 0 else tuple(array.shape)


def _spin_indices(space, spin: str) -> np.ndarray:
    offset = 0 if spin == "up" else 1
    return np.arange(offset, space.num_spin_orbitals, 2, dtype=int)


def _take_block(array: np.ndarray, indices: tuple[np.ndarray, ...]) -> np.ndarray:
    result = np.asarray(array)
    start_axis = result.ndim - len(indices)
    for axis_offset, idx in enumerate(indices):
        result = np.take(result, idx, axis=start_axis + axis_offset)
    return result


def _same_space(left, right) -> bool:
    return (
        left is right
        or (
            left.num_sites == right.num_sites
            and tuple(left.orbitals) == tuple(right.orbitals)
            and tuple(left.spins) == tuple(right.spins)
        )
    )


def _expand_transform(space, transform) -> np.ndarray:
    nmode = space.num_spin_orbitals
    nspatial = space.num_spatial_orbitals

    if isinstance(transform, (tuple, list)) and len(transform) == 2:
        alpha = np.asarray(transform[0], dtype=np.complex128)
        beta = np.asarray(transform[1], dtype=np.complex128)
        if alpha.shape != (nspatial, nspatial) or beta.shape != (nspatial, nspatial):
            raise ValueError(
                "Spin-block transforms must each have shape "
                f"({nspatial}, {nspatial})."
            )
        full = np.zeros((nmode, nmode), dtype=np.complex128)
        alpha_idx = _spin_indices(space, "up")
        beta_idx = _spin_indices(space, "down")
        full[np.ix_(alpha_idx, alpha_idx)] = alpha
        full[np.ix_(beta_idx, beta_idx)] = beta
        return full

    matrix = np.asarray(transform, dtype=np.complex128)
    if matrix.shape == (nspatial, nspatial):
        alpha_idx = _spin_indices(space, "up")
        beta_idx = _spin_indices(space, "down")
        full = np.zeros((nmode, nmode), dtype=np.complex128)
        full[np.ix_(alpha_idx, alpha_idx)] = matrix
        full[np.ix_(beta_idx, beta_idx)] = matrix
        return full
    if matrix.shape == (nmode, nmode):
        return matrix
    raise ValueError(
        "basis transform must have shape "
        f"({nspatial}, {nspatial}), ({nmode}, {nmode}), or be a pair of spatial transforms."
    )


def _normalize_batch_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise ValueError(f"axis {axis} is out of bounds for batch rank {ndim}.")
    return axis


def _component_einsum_operands(attr: str, operands, indices: tuple[int, ...] | None):
    selected = operands if indices is None else tuple(operands[index] for index in indices)
    arrays = []
    for operand in selected:
        if isinstance(operand, ElectronicIntegrals):
            arrays.append(getattr(operand, attr))
        else:
            arrays.append(np.asarray(operand))
    return arrays


@dataclass(frozen=True)
class ElectronicIntegrals:
    """Structured constant, one-body, and two-body spin-orbital coefficients.

    The design mirrors mature packages such as OpenFermion and Qiskit Nature:
    keep a fully general symbolic operator available, but also offer a typed
    tensor container when the Hamiltonian has recognized electronic structure.
    """

    space: object
    constant: complex = 0.0
    one_body: np.ndarray | None = None
    two_body: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate tensor shapes and infer any leading batch dimensions."""
        nmode = self.space.num_spin_orbitals
        one_body = (
            np.zeros((nmode, nmode), dtype=np.complex128)
            if self.one_body is None
            else _validate_square_matrix("one_body", self.one_body, nmode)
        )
        two_body = (
            np.zeros((nmode, nmode, nmode, nmode), dtype=np.complex128)
            if self.two_body is None
            else _validate_rank4_tensor("two_body", self.two_body, nmode)
        )
        constant = _copy_constant(self.constant)

        batch_shapes = {
            tuple(one_body.shape[:-2]),
            tuple(two_body.shape[:-4]),
            _batch_shape_of_constant(constant),
        }
        batch_shapes.discard(())
        if len(batch_shapes) > 1:
            raise ValueError(f"Inconsistent batch shapes in ElectronicIntegrals: {sorted(batch_shapes)!r}.")
        batch_shape = next(iter(batch_shapes), ())

        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "one_body", one_body.copy())
        object.__setattr__(self, "two_body", two_body.copy())
        object.__setattr__(self, "_batch_shape", batch_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def one_body_alpha(self) -> np.ndarray:
        idx = _spin_indices(self.space, "up")
        return _take_block(self.one_body, (idx, idx))

    @property
    def one_body_beta(self) -> np.ndarray:
        idx = _spin_indices(self.space, "down")
        return _take_block(self.one_body, (idx, idx))

    @property
    def two_body_aa(self) -> np.ndarray:
        idx = _spin_indices(self.space, "up")
        return _take_block(self.two_body, (idx, idx, idx, idx))

    @property
    def two_body_ab(self) -> np.ndarray:
        return _take_block(
            self.two_body,
            (
                _spin_indices(self.space, "up"),
                _spin_indices(self.space, "down"),
                _spin_indices(self.space, "up"),
                _spin_indices(self.space, "down"),
            ),
        )

    @property
    def two_body_bb(self) -> np.ndarray:
        idx = _spin_indices(self.space, "down")
        return _take_block(self.two_body, (idx, idx, idx, idx))

    @classmethod
    def zeros(cls, space) -> "ElectronicIntegrals":
        return cls(space)

    @classmethod
    def from_raw(
        cls,
        space,
        *,
        one_body: np.ndarray,
        two_body: np.ndarray | None = None,
        constant: complex = 0.0,
    ) -> "ElectronicIntegrals":
        return cls(space, constant=constant, one_body=one_body, two_body=two_body)

    @classmethod
    def from_restricted(
        cls,
        space,
        *,
        one_body: np.ndarray,
        two_body: np.ndarray | None = None,
        constant: complex = 0.0,
    ) -> "ElectronicIntegrals":
        """Build spin-orbital tensors from restricted spatial-orbital data."""
        nspatial = space.num_spatial_orbitals
        nmode = space.num_spin_orbitals
        one_spatial = np.asarray(one_body, dtype=np.complex128)
        if one_spatial.shape[-2:] != (nspatial, nspatial):
            raise ValueError(f"restricted one_body must have trailing shape ({nspatial}, {nspatial}).")

        batch_shape = tuple(one_spatial.shape[:-2])
        alpha_idx = _spin_indices(space, "up")
        beta_idx = _spin_indices(space, "down")

        full_one = np.zeros(batch_shape + (nmode, nmode), dtype=np.complex128)
        full_one[(...,) + np.ix_(alpha_idx, alpha_idx)] = one_spatial
        full_one[(...,) + np.ix_(beta_idx, beta_idx)] = one_spatial

        full_two = np.zeros(batch_shape + (nmode, nmode, nmode, nmode), dtype=np.complex128)
        if two_body is not None:
            two_spatial = np.asarray(two_body, dtype=np.complex128)
            if two_spatial.shape[-4:] != (nspatial, nspatial, nspatial, nspatial):
                raise ValueError(
                    "restricted two_body must have trailing shape "
                    f"({nspatial}, {nspatial}, {nspatial}, {nspatial})."
                )
            if tuple(two_spatial.shape[:-4]) != batch_shape:
                raise ValueError("restricted one_body and two_body must share the same batch shape.")
            full_two[(...,) + np.ix_(alpha_idx, alpha_idx, alpha_idx, alpha_idx)] = two_spatial
            full_two[(...,) + np.ix_(beta_idx, beta_idx, beta_idx, beta_idx)] = two_spatial
            full_two[(...,) + np.ix_(alpha_idx, beta_idx, alpha_idx, beta_idx)] = two_spatial
            full_two[(...,) + np.ix_(beta_idx, alpha_idx, beta_idx, alpha_idx)] = np.transpose(
                two_spatial,
                (*range(two_spatial.ndim - 4), -2, -1, -4, -3),
            )

        return cls(space, constant=constant, one_body=full_one, two_body=full_two)

    @classmethod
    def from_unrestricted(
        cls,
        space,
        *,
        one_body_alpha: np.ndarray,
        one_body_beta: np.ndarray | None = None,
        two_body_aa: np.ndarray | None = None,
        two_body_ab: np.ndarray | None = None,
        two_body_bb: np.ndarray | None = None,
        constant: complex = 0.0,
    ) -> "ElectronicIntegrals":
        """Build spin-orbital tensors from unrestricted spin-block data."""
        nspatial = space.num_spatial_orbitals
        nmode = space.num_spin_orbitals
        alpha_idx = _spin_indices(space, "up")
        beta_idx = _spin_indices(space, "down")

        one_a = np.asarray(one_body_alpha, dtype=np.complex128)
        if one_a.shape[-2:] != (nspatial, nspatial):
            raise ValueError(f"one_body_alpha must have trailing shape ({nspatial}, {nspatial}).")
        batch_shape = tuple(one_a.shape[:-2])

        one_b = one_a if one_body_beta is None else np.asarray(one_body_beta, dtype=np.complex128)
        if one_b.shape[-2:] != (nspatial, nspatial):
            raise ValueError(f"one_body_beta must have trailing shape ({nspatial}, {nspatial}).")
        if tuple(one_b.shape[:-2]) != batch_shape:
            raise ValueError("one_body_alpha and one_body_beta must share the same batch shape.")

        full_one = np.zeros(batch_shape + (nmode, nmode), dtype=np.complex128)
        full_one[(...,) + np.ix_(alpha_idx, alpha_idx)] = one_a
        full_one[(...,) + np.ix_(beta_idx, beta_idx)] = one_b

        full_two = np.zeros(batch_shape + (nmode, nmode, nmode, nmode), dtype=np.complex128)
        if two_body_aa is not None:
            aa = np.asarray(two_body_aa, dtype=np.complex128)
            if aa.shape[-4:] != (nspatial, nspatial, nspatial, nspatial):
                raise ValueError(f"two_body_aa must have trailing shape ({nspatial}, {nspatial}, {nspatial}, {nspatial}).")
            if tuple(aa.shape[:-4]) != batch_shape:
                raise ValueError("two_body_aa must share the same batch shape as the one-body blocks.")
            full_two[(...,) + np.ix_(alpha_idx, alpha_idx, alpha_idx, alpha_idx)] = aa
        if two_body_ab is not None:
            ab = np.asarray(two_body_ab, dtype=np.complex128)
            if ab.shape[-4:] != (nspatial, nspatial, nspatial, nspatial):
                raise ValueError(f"two_body_ab must have trailing shape ({nspatial}, {nspatial}, {nspatial}, {nspatial}).")
            if tuple(ab.shape[:-4]) != batch_shape:
                raise ValueError("two_body_ab must share the same batch shape as the one-body blocks.")
            full_two[(...,) + np.ix_(alpha_idx, beta_idx, alpha_idx, beta_idx)] = ab
            full_two[(...,) + np.ix_(beta_idx, alpha_idx, beta_idx, alpha_idx)] = np.transpose(
                ab,
                (*range(ab.ndim - 4), -2, -1, -4, -3),
            )
        if two_body_bb is not None:
            bb = np.asarray(two_body_bb, dtype=np.complex128)
            if bb.shape[-4:] != (nspatial, nspatial, nspatial, nspatial):
                raise ValueError(f"two_body_bb must have trailing shape ({nspatial}, {nspatial}, {nspatial}, {nspatial}).")
            if tuple(bb.shape[:-4]) != batch_shape:
                raise ValueError("two_body_bb must share the same batch shape as the one-body blocks.")
            full_two[(...,) + np.ix_(beta_idx, beta_idx, beta_idx, beta_idx)] = bb

        return cls(space, constant=constant, one_body=full_one, two_body=full_two)

    @classmethod
    def einsum(
        cls,
        subscripts: dict[str, str | tuple[str, tuple[int, ...]]],
        *operands,
        space,
        constant: complex = 0.0,
    ) -> "ElectronicIntegrals":
        """Apply `numpy.einsum` component-wise to one-/two-body tensors.

        This is primarily a developer-facing utility used for basis transforms
        and higher-level tensor algebra.
        """
        one_body = None
        two_body = None

        if "one_body" in subscripts:
            one_spec = subscripts["one_body"]
            if isinstance(one_spec, tuple):
                one_expr, one_indices = one_spec
            else:
                one_expr, one_indices = one_spec, None
            one_body = np.einsum(
                one_expr,
                *_component_einsum_operands("one_body", operands, one_indices),
                optimize=True,
            )

        if "two_body" in subscripts:
            two_spec = subscripts["two_body"]
            if isinstance(two_spec, tuple):
                two_expr, two_indices = two_spec
            else:
                two_expr, two_indices = two_spec, None
            two_body = np.einsum(
                two_expr,
                *_component_einsum_operands("two_body", operands, two_indices),
                optimize=True,
            )

        return cls(space, constant=constant, one_body=one_body, two_body=two_body)

    @classmethod
    def stack(cls, integrals: list["ElectronicIntegrals"], *, axis: int = 0) -> "ElectronicIntegrals":
        """Stack a list of integral containers along a new batch axis."""
        if not integrals:
            raise ValueError("stack() requires at least one ElectronicIntegrals instance.")
        space = integrals[0].space
        if not all(_same_space(space, item.space) for item in integrals):
            raise ValueError("All ElectronicIntegrals passed to stack() must share the same ElectronicSpace.")
        constants = np.stack([np.asarray(item.constant, dtype=np.complex128) for item in integrals], axis=axis)
        one_body = np.stack([item.one_body for item in integrals], axis=axis)
        two_body = np.stack([item.two_body for item in integrals], axis=axis)
        return cls(space, constant=constants, one_body=one_body, two_body=two_body)

    def split(self, indices_or_sections, *, axis: int = 0) -> list["ElectronicIntegrals"]:
        """Split a batched container into a list of smaller containers."""
        if not self.batch_shape:
            raise ValueError("split() requires a stacked ElectronicIntegrals object with non-empty batch_shape.")
        axis = _normalize_batch_axis(axis, len(self.batch_shape))
        constants = np.split(np.asarray(self.constant, dtype=np.complex128), indices_or_sections, axis=axis)
        one_bodies = np.split(self.one_body, indices_or_sections, axis=axis)
        two_bodies = np.split(self.two_body, indices_or_sections, axis=axis)

        out = []
        for constant, one_body, two_body in zip(constants, one_bodies, two_bodies):
            if constant.shape[axis] == 1:
                constant = np.squeeze(constant, axis=axis)
                one_body = np.squeeze(one_body, axis=axis)
                two_body = np.squeeze(two_body, axis=axis)
            out.append(type(self)(self.space, constant=constant, one_body=one_body, two_body=two_body))
        return out

    def basis_transform(self, transform) -> "ElectronicIntegrals":
        """Apply an orbital basis transform to both one- and two-body tensors."""
        transform_matrix = _expand_transform(self.space, transform)
        one_body = np.einsum(
            "pi,...pq,qj->...ij",
            transform_matrix.conj(),
            self.one_body,
            transform_matrix,
            optimize=True,
        )
        two_body = np.einsum(
            "pi,qj,...pqrs,rk,sl->...ijkl",
            transform_matrix.conj(),
            transform_matrix.conj(),
            self.two_body,
            transform_matrix,
            transform_matrix,
            optimize=True,
        )
        return type(self)(self.space, constant=self.constant, one_body=one_body, two_body=two_body)

    def to_operator(self, *, atol: float = 1e-12) -> Operator:
        """Convert the structured tensors back into a symbolic `Operator`."""
        if self.batch_shape:
            raise ValueError("to_operator() requires an unbatched ElectronicIntegrals object.")
        terms: dict[Term, complex] = {}
        nmode = self.space.num_spin_orbitals

        for p in range(nmode):
            for q in range(nmode):
                coeff = self.one_body[p, q]
                if abs(coeff) <= atol:
                    continue
                term = Term((Ladder(p, "create"), Ladder(q, "destroy")))
                terms[term] = terms.get(term, 0.0) + coeff

        for p in range(nmode):
            for q in range(nmode):
                for r in range(nmode):
                    for s in range(nmode):
                        coeff = self.two_body[p, q, r, s]
                        if abs(coeff) <= atol:
                            continue
                        term = Term(
                            (
                                Ladder(p, "create"),
                                Ladder(q, "create"),
                                Ladder(r, "destroy"),
                                Ladder(s, "destroy"),
                            )
                        )
                        terms[term] = terms.get(term, 0.0) + coeff

        return Operator(self.space, constant=self.constant, terms=terms).simplify(atol=atol)

    def equiv(self, other, *, atol: float = 1e-12) -> bool:
        """Check tensor-level equivalence up to numerical tolerance."""
        if not isinstance(other, ElectronicIntegrals):
            return False
        if not _same_space(self.space, other.space):
            return False
        return (
            np.allclose(np.asarray(self.constant), np.asarray(other.constant), atol=atol)
            and np.allclose(self.one_body, other.one_body, atol=atol)
            and np.allclose(self.two_body, other.two_body, atol=atol)
        )

    def __add__(self, other):
        if not isinstance(other, ElectronicIntegrals):
            return NotImplemented
        if not _same_space(self.space, other.space):
            raise ValueError("Cannot add ElectronicIntegrals from different ElectronicSpace objects.")
        return type(self)(
            self.space,
            constant=np.asarray(self.constant) + np.asarray(other.constant),
            one_body=self.one_body + other.one_body,
            two_body=self.two_body + other.two_body,
        )

    def __sub__(self, other):
        if not isinstance(other, ElectronicIntegrals):
            return NotImplemented
        return self + ((-1) * other)

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        return type(self)(
            self.space,
            constant=np.asarray(self.constant) * scalar,
            one_body=self.one_body * scalar,
            two_body=self.two_body * scalar,
        )

    def __rmul__(self, scalar):
        return self * scalar
