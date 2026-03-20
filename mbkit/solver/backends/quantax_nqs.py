from __future__ import annotations

from functools import partial
from importlib import import_module
from pathlib import Path
import math
import sys

import numpy as np

from ...operator import Ladder, Operator as SymbolicOperator, Term
from ...operator.transforms import UnsupportedTransformError
from ...utils.solver import coerce_symbolic_operator
from ..base import SolverBackend
from ..capabilities import BackendCapabilities
from ..compile import compile_quantax_hamiltonian


def _load_quantax():
    try:
        return import_module("quantax"), "installed", None
    except Exception as installed_exc:
        sys.modules.pop("quantax", None)

    extern_root = Path(__file__).resolve().parents[3] / "extern" / "quantax"
    inserted = False
    if str(extern_root) not in sys.path:
        sys.path.insert(0, str(extern_root))
        inserted = True

    try:
        return import_module("quantax"), "extern", None
    except Exception as extern_exc:
        sys.modules.pop("quantax", None)
        if inserted:
            try:
                sys.path.remove(str(extern_root))
            except ValueError:
                pass
        return None, None, extern_exc


qtx, _QUANTAX_SOURCE, _QUANTAX_IMPORT_ERROR = _load_quantax()
if _QUANTAX_IMPORT_ERROR is None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu

    _common_samplers = import_module("quantax.sampler.common_samplers")
    _metropolis = import_module("quantax.sampler.metropolis")
    _common_symmetries = import_module("quantax.symmetry.common_symmetries")
    _utils = import_module("quantax.utils")
    _array_utils = import_module("quantax.utils.array")
    _propose_exchange = _common_samplers._propose_exchange
    Metropolis = _metropolis.Metropolis
    to_replicate_array = _array_utils.to_replicate_array
    ints_to_array = _utils.ints_to_array

    try:
        import_module("quspin")

        _QUSPIN_IMPORT_ERROR = None
    except Exception as exc:
        _QUSPIN_IMPORT_ERROR = exc
else:
    eqx = jax = jnp = jtu = None
    _propose_exchange = None
    Metropolis = object
    to_replicate_array = None
    ints_to_array = None
    _QUSPIN_IMPORT_ERROR = _QUANTAX_IMPORT_ERROR


def _quantax_dependency_message() -> str:
    return (
        "Quantax-backed NQSSolver requires the optional `mbkit[nqs]` dependencies. "
        "Install them with `pip install \"mbkit[nqs]\"`. Quantax also depends on JAX, "
        "and JAX installation is platform-specific, so follow the JAX install instructions "
        "for your OS and accelerator before using the NQS backend."
    )


def _require_quantax() -> None:
    if _QUANTAX_IMPORT_ERROR is None:
        return
    raise ImportError(_quantax_dependency_message()) from _QUANTAX_IMPORT_ERROR


def _require_quspin_for_exact_sampling() -> None:
    if _QUSPIN_IMPORT_ERROR is None:
        return
    raise ImportError(
        "Quantax exact sampling requires QuSpin in addition to the NQS stack. "
        "Install `pip install \"mbkit[nqs]\"` and ensure QuSpin is available."
    ) from _QUSPIN_IMPORT_ERROR


def _operator_is_complex(operator: SymbolicOperator, *, atol: float = 1e-12) -> bool:
    if abs(complex(operator.constant).imag) > atol:
        return True
    return any(abs(complex(coeff).imag) > atol for _, coeff in operator.iter_terms())


def _space_signature(space) -> tuple[object, ...]:
    return (
        int(space.num_sites),
        tuple(space.orbitals),
        tuple(space.spins),
    )


def _term_particle_change(term) -> int:
    delta = 0
    for ladder in term.factors:
        delta += 1 if ladder.action == "create" else -1
    return delta


def _term_spin_change(space, term) -> int:
    delta = 0
    for ladder in term.factors:
        spin_sign = 1 if space.unpack_mode(ladder.mode).spin == "up" else -1
        delta += spin_sign if ladder.action == "create" else -spin_sign
    return delta


def _spin_changes(operator: SymbolicOperator) -> frozenset[int]:
    changes = set()
    if abs(operator.constant) > 1e-15:
        changes.add(0)
    for term, _ in operator.iter_terms():
        changes.add(_term_spin_change(operator.space, term))
    return frozenset(changes)


def _term_preserves_sector(space, term, *, sector_kind: str) -> bool:
    if _term_particle_change(term) != 0:
        return False
    if sector_kind == "spin_resolved":
        return _term_spin_change(space, term) == 0
    return True


def _sector_preserving_part(operator: SymbolicOperator, *, sector_kind: str) -> SymbolicOperator:
    preserved_terms = {
        term: coeff
        for term, coeff in operator.iter_terms()
        if _term_preserves_sector(operator.space, term, sector_kind=sector_kind)
    }
    return SymbolicOperator(operator.space, constant=operator.constant, terms=preserved_terms)


def _validate_quantax_problem(operator: SymbolicOperator) -> None:
    space = operator.space
    if space.num_spins != 2 or tuple(space.spins) != ("up", "down"):
        raise UnsupportedTransformError(
            "Quantax NQS v1 currently requires a two-spin ElectronicSpace."
        )
    for term, _ in operator.iter_terms():
        if len(term.factors) % 2 != 0:
            raise UnsupportedTransformError(
                "Quantax NQS v1 does not support odd-parity operator terms."
            )
        if _term_particle_change(term) != 0:
            raise UnsupportedTransformError(
                "Quantax NQS v1 supports only particle-number-conserving Hamiltonians."
            )


def _resolve_jax_dtype(operator: SymbolicOperator, requested_dtype):
    op_is_complex = _operator_is_complex(operator)
    if requested_dtype is None:
        return jnp.complex128 if op_is_complex else jnp.float64

    dtype = np.dtype(requested_dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return jnp.complex64 if dtype.itemsize <= 8 else jnp.complex128
    if np.issubdtype(dtype, np.floating):
        if op_is_complex:
            raise ValueError("Complex Hamiltonians require a complex Quantax dtype.")
        return jnp.float32 if dtype.itemsize <= 4 else jnp.float64
    raise TypeError("dtype must be a floating or complex floating dtype.")


def _resolve_device_multiple(nsamples: int) -> int:
    value = int(nsamples)
    if value <= 0:
        raise ValueError("`nsamples` must be positive.")
    if value % jax.device_count() != 0:
        raise ValueError(
            "`nsamples` must be a multiple of the number of JAX devices; "
            f"got {value} samples for {jax.device_count()} devices."
        )
    return value


def _allowed_spin_partitions(total_particles: int, num_spatial_orbitals: int) -> tuple[tuple[int, int], ...]:
    min_up = max(0, total_particles - num_spatial_orbitals)
    max_up = min(total_particles, num_spatial_orbitals)
    return tuple((n_up, total_particles - n_up) for n_up in range(min_up, max_up + 1))


def _resolve_quantax_particle_sector(*, space=None, n_electrons=None, n_particles=None):
    if n_electrons is not None and n_particles is not None:
        raise ValueError("Provide only one of `n_electrons` or `n_particles`.")

    spec = n_particles if n_particles is not None else n_electrons
    if spec is None and space is not None:
        if getattr(space, "n_electrons_per_spin", None) is not None:
            spec = tuple(space.n_electrons_per_spin)
        elif getattr(space, "n_electrons", None) is not None:
            spec = int(space.n_electrons)
    if spec is None:
        raise ValueError("An electron count must be supplied via `n_electrons` or `n_particles`.")

    if isinstance(spec, (int, np.integer)):
        total = int(spec)
        return total, total, "total"

    if isinstance(spec, tuple) and len(spec) == 2:
        pair = (int(spec[0]), int(spec[1]))
        return pair, sum(pair), "spin_resolved"

    if isinstance(spec, list):
        if len(spec) == 1 and isinstance(spec[0], tuple) and len(spec[0]) == 2:
            pair = (int(spec[0][0]), int(spec[0][1]))
            return pair, sum(pair), "spin_resolved"

        if len(spec) == 2 and all(isinstance(value, (int, np.integer)) for value in spec):
            pair = (int(spec[0]), int(spec[1]))
            return pair, sum(pair), "spin_resolved"

        if spec and all(isinstance(entry, tuple) and len(entry) == 2 for entry in spec):
            partitions = tuple((int(entry[0]), int(entry[1])) for entry in spec)
            totals = {sum(entry) for entry in partitions}
            if len(totals) != 1:
                raise ValueError(
                    "Quantax NQS multi-sector input must keep a fixed total particle number."
                )
            total = totals.pop()
            if space is None:
                raise ValueError(
                    "Quantax NQS needs the ElectronicSpace to interpret multi-sector particle input."
                )
            allowed = set(_allowed_spin_partitions(total, space.num_spatial_orbitals))
            if set(partitions) == allowed:
                return total, total, "total"
            raise ValueError(
                "Quantax NQS does not support arbitrary subsets of spin sectors. "
                "Use an integer total particle count, or provide the full list of "
                "spin partitions for that total particle number."
            )

    raise TypeError(
        "Electron counts must be an int, a `(n_alpha, n_beta)` tuple, "
        "or a full list of spin partitions representing one fixed total particle number."
    )


def _hilbert_dimension(num_spatial_orbitals: int, n_particles: int | tuple[int, int]) -> int:
    if isinstance(n_particles, tuple):
        n_up, n_down = n_particles
        return math.comb(num_spatial_orbitals, n_up) * math.comb(num_spatial_orbitals, n_down)
    return math.comb(2 * num_spatial_orbitals, int(n_particles))


def _expand_site_coordinates(space):
    lattice = getattr(space, "lattice", None)
    if lattice is None or lattice.site_positions is None:
        return None

    base_positions = lattice.site_positions
    ndim = max((len(position) for position in base_positions), default=0)
    add_orbital_axis = space.num_orbitals_per_site > 1

    coords = []
    for site in range(space.num_sites):
        base = tuple(float(value) for value in base_positions[site])
        padded = base + (0.0,) * (ndim - len(base))
        for orbital_index in range(space.num_orbitals_per_site):
            if add_orbital_axis:
                coords.append(padded + (float(orbital_index),))
            else:
                coords.append(padded)
    return np.asarray(coords, dtype=float)


def _build_neighbor_table(
    num_spatial_orbitals: int,
    edges: tuple[tuple[int, int], ...],
    *,
    allow_spin_mixing: bool = False,
) -> np.ndarray:
    nmode = 2 * num_spatial_orbitals
    adjacency = [set() for _ in range(nmode)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
        adjacency[left + num_spatial_orbitals].add(right + num_spatial_orbitals)
        adjacency[right + num_spatial_orbitals].add(left + num_spatial_orbitals)

    if allow_spin_mixing:
        for spatial in range(num_spatial_orbitals):
            up_mode = spatial
            down_mode = spatial + num_spatial_orbitals
            adjacency[up_mode].add(down_mode)
            adjacency[down_mode].add(up_mode)

    for mode, neighbors in enumerate(adjacency):
        if neighbors:
            continue
        if allow_spin_mixing:
            neighbors.update(other for other in range(nmode) if other != mode)
        else:
            block_offset = 0 if mode < num_spatial_orbitals else num_spatial_orbitals
            local_index = mode - block_offset
            neighbors.update(
                block_offset + other
                for other in range(num_spatial_orbitals)
                if other != local_index
            )

    max_neighbors = max((len(neighbors) for neighbors in adjacency), default=0)
    if max_neighbors == 0:
        return np.full((nmode, 1), -1, dtype=np.int32)

    neighbor_table = np.full((nmode, max_neighbors), -1, dtype=np.int32)
    for row, neighbors in enumerate(adjacency):
        for column, neighbor in enumerate(sorted(neighbors)):
            neighbor_table[row, column] = neighbor
    return neighbor_table


def _callable_name(value) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "__name__"):
        return value.__name__
    return type(value).__name__


def _real_component_dtype(dtype):
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return jnp.float32 if dtype == jnp.complex64 else jnp.float64
    return dtype


def _random_orthonormal_columns(n_rows: int, n_cols: int, dtype):
    if n_cols <= 0:
        return jnp.zeros((n_rows, 0), dtype=dtype)

    real_dtype = _real_component_dtype(dtype)
    key_real = qtx.get_subkeys()
    matrix = jax.random.normal(key_real, (n_rows, n_cols), dtype=real_dtype)
    if jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
        key_imag = qtx.get_subkeys()
        matrix = matrix + 1j * jax.random.normal(key_imag, (n_rows, n_cols), dtype=real_dtype)
    q, _ = jnp.linalg.qr(matrix, mode="reduced")
    return q.astype(dtype)


def _default_total_sector_det_orbitals(dtype):
    sites = qtx.get_sites()
    return _random_orthonormal_columns(sites.Nfmodes, sites.Ntotal, dtype)


def _default_total_sector_pf_backflow_orbitals(dtype):
    sites = qtx.get_sites()
    return _random_orthonormal_columns(sites.Nfmodes, sites.Nfmodes, dtype)


def _single_device_model(model):
    if jax.device_count() != 1:
        return model
    return jtu.tree_map(
        lambda value: jax.device_put(np.asarray(value)) if isinstance(value, jax.Array) else value,
        model,
    )


def _reset_quantax_symmetry_cache() -> None:
    if _QUANTAX_IMPORT_ERROR is not None:
        return
    _common_symmetries._Identity = None
    _common_symmetries._Z2Inverse = {}


if _QUANTAX_IMPORT_ERROR is None:

    class GraphParticleHop(Metropolis):
        """Particle-hop sampler driven by an explicit orbital graph."""

        def __init__(
            self,
            state,
            nsamples: int,
            neighbors: np.ndarray,
            *,
            reweight: float = 2.0,
            thermal_steps: int | None = None,
            sweep_steps: int | None = None,
            initial_spins=None,
        ):
            sites = qtx.get_sites()
            if sites.Nparticles is None:
                raise ValueError(
                    "The number of particles must be fixed in Quantax Sites for `GraphParticleHop`."
                )

            if 2 * sites.Ntotal <= state.Nmodes:
                self._hopping_particle = 1
            else:
                self._hopping_particle = -1

            self._neighbors = to_replicate_array(neighbors).astype(jnp.int32)
            super().__init__(
                state=state,
                nsamples=nsamples,
                reweight=reweight,
                thermal_steps=thermal_steps,
                sweep_steps=sweep_steps,
                initial_spins=initial_spins,
            )

        @property
        def particle_type(self):
            return (qtx.PARTICLE_TYPE.spinful_fermion,)

        @property
        def nflips(self) -> int:
            return 2

        @partial(jax.jit, static_argnums=0)
        def propose(self, key, old_spins):
            return _propose_exchange(
                key,
                old_spins,
                self._hopping_particle,
                self._neighbors,
            )


class QuantaxNQSBackend(SolverBackend):
    """Variational neural-quantum-state backend implemented with Quantax."""

    solver_family = "nqs"
    backend_name = "quantax"
    capabilities = BackendCapabilities(
        supports_ground_state=True,
        supports_rdm1=True,
        supports_statistics=True,
        supports_native_expectation=True,
        supports_complex=True,
        preferred_problem_type="variational_nqs",
    )

    def __init__(
        self,
        *,
        model: str | object = "pf_backflow",
        optimizer: str | object = "march",
        linear_solver: str | object = "auto_pinv_eig",
        sampler: str | object = "auto",
        nsamples: int = 4096,
        n_steps: int = 200,
        step_size: float = 1e-2,
        seed: int = 42,
        exact_sampler_cutoff: int = 4096,
        model_kwargs: dict[str, object] | None = None,
        optimizer_kwargs: dict[str, object] | None = None,
        sampler_kwargs: dict[str, object] | None = None,
        max_parallel=None,
        dtype=None,
    ) -> None:
        _require_quantax()

        self.model_spec = model
        self.optimizer_spec = optimizer
        self.linear_solver_spec = linear_solver
        self.sampler_spec = sampler
        self.nsamples = int(nsamples)
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)
        self.seed = int(seed)
        self.exact_sampler_cutoff = int(exact_sampler_cutoff)
        self.model_kwargs = dict(model_kwargs or {})
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.sampler_kwargs = dict(sampler_kwargs or {})
        self.max_parallel = max_parallel
        self.requested_dtype = dtype

        if self.n_steps < 0:
            raise ValueError("`n_steps` must be non-negative.")
        if self.exact_sampler_cutoff < 1:
            raise ValueError("`exact_sampler_cutoff` must be positive.")

        self.operator = None
        self.space = None
        self.n_particles = None
        self.n_electrons = None
        self._particle_sector_kind = None
        self._spin_mixing_enabled = False
        self.compilation = None
        self.quantax_hamiltonian = None
        self.quantax_sites = None
        self.state = None
        self.optimizer = None
        self.sampler = None
        self.measurement_samples = None
        self._resolved_dtype = None
        self._resolved_model_name = None
        self._resolved_optimizer_name = None
        self._resolved_linear_solver_name = None
        self._resolved_sampler_name = None
        self._exact_sampler_symm = None
        self._last_energy = None
        self._last_variance = None
        self._history = {"energy": [], "variance": []}

    def _require_solution(self) -> None:
        if self.state is None or self.operator is None or self.measurement_samples is None:
            raise RuntimeError("Call solve() before requesting observables.")

    def _validate_observable_space(self, operator: SymbolicOperator) -> None:
        if _space_signature(operator.space) != _space_signature(self.space):
            raise ValueError("Observable ElectronicSpace does not match the solved Hamiltonian space.")

    def _build_linear_solver(self, state):
        solver_kwargs = dict(self.optimizer_kwargs).pop("linear_solver_kwargs", {})
        solver_spec = self.linear_solver_spec
        if callable(solver_spec) and not isinstance(solver_spec, str):
            self._resolved_linear_solver_name = _callable_name(solver_spec)
            return solver_spec

        solver_name = str(solver_spec).strip().lower()
        factories = {
            "auto_pinv_eig": qtx.optimizer.auto_pinv_eig,
            "minnorm_pinv_eig": qtx.optimizer.minnorm_pinv_eig,
            "lstsq_pinv_eig": qtx.optimizer.lstsq_pinv_eig,
            "pinvh_solve": qtx.optimizer.pinvh_solve,
            "block_pinv_eig": qtx.optimizer.block_pinv_eig,
        }
        try:
            factory = factories[solver_name]
        except KeyError as exc:
            raise ValueError(
                "Unknown Quantax linear solver "
                f"{solver_spec!r}. Supported values are {sorted(factories)!r}."
            ) from exc

        self._resolved_linear_solver_name = solver_name
        if solver_name == "block_pinv_eig":
            return factory(state, **solver_kwargs)
        return factory(**solver_kwargs)

    def _build_model(self):
        model_kwargs = dict(self.model_kwargs)
        if not isinstance(self.model_spec, str):
            if model_kwargs:
                raise TypeError("Custom Quantax model instances do not accept `model_kwargs`.")
            self._resolved_model_name = _callable_name(self.model_spec)
            return self.model_spec

        model_name = self.model_spec.strip().lower()
        self._resolved_model_name = model_name
        dtype = self._resolved_dtype

        if model_name == "general_det":
            if isinstance(qtx.get_sites().Nparticles, int) and "U" not in model_kwargs:
                model_kwargs["U"] = _default_total_sector_det_orbitals(dtype)
            return qtx.model.GeneralDet(dtype=dtype, **model_kwargs)
        if model_name == "general_pf":
            return qtx.model.GeneralPf(dtype=dtype, **model_kwargs)

        if model_name not in {"det_backflow", "pf_backflow"}:
            raise ValueError(
                "Unknown Quantax model "
                f"{self.model_spec!r}. Supported values are "
                "('general_det', 'general_pf', 'det_backflow', 'pf_backflow')."
            )

        sites = qtx.get_sites()
        net = model_kwargs.pop("net", None)
        d = int(model_kwargs.pop("d", max(8, min(32, 2 * math.ceil(math.sqrt(sites.Nfmodes))))))
        if sites.is_spinful and d % 2 != 0:
            d += 1
        hidden_width = int(model_kwargs.pop("hidden_width", max(64, 4 * sites.Nfmodes)))
        depth = int(model_kwargs.pop("depth", 1))
        activation = model_kwargs.pop("activation", jax.nn.gelu)
        use_final_bias = bool(model_kwargs.pop("use_final_bias", False))

        if net is None:
            effective_channels = d // 2 if sites.is_spinful else d
            out_size = effective_channels * sites.Nfmodes
            net = eqx.nn.MLP(
                in_size=sites.Nmodes,
                out_size=out_size,
                width_size=hidden_width,
                depth=depth,
                activation=activation,
                final_activation=lambda value: value,
                use_final_bias=use_final_bias,
                dtype=dtype,
                key=qtx.get_subkeys(),
            )

        if model_name == "det_backflow":
            if isinstance(qtx.get_sites().Nparticles, int) and "U0" not in model_kwargs:
                model_kwargs["U0"] = _default_total_sector_det_orbitals(dtype)
            return qtx.model.DetBackflow(net=net, d=d, dtype=dtype, **model_kwargs)
        if isinstance(qtx.get_sites().Nparticles, int) and "U0" not in model_kwargs:
            model_kwargs["U0"] = _default_total_sector_pf_backflow_orbitals(dtype)
        return qtx.model.PfBackflow(net=net, d=d, dtype=dtype, **model_kwargs)

    def _build_optimizer(self):
        optimizer_kwargs = dict(self.optimizer_kwargs)
        solver_override = optimizer_kwargs.pop("solver", None)
        optimizer_kwargs.pop("linear_solver_kwargs", None)

        if solver_override is None:
            solver = self._build_linear_solver(self.state)
        else:
            solver = solver_override
            self._resolved_linear_solver_name = _callable_name(solver_override)

        if not isinstance(self.optimizer_spec, str):
            self._resolved_optimizer_name = _callable_name(self.optimizer_spec)
            return self.optimizer_spec(
                state=self.state,
                hamiltonian=self.quantax_hamiltonian,
                solver=solver,
                **optimizer_kwargs,
            )

        optimizer_name = self.optimizer_spec.strip().lower()
        self._resolved_optimizer_name = optimizer_name
        optimizer_map = {
            "march": qtx.optimizer.MARCH,
            "sr": qtx.optimizer.SR,
            "spring": qtx.optimizer.SPRING,
            "adam_sr": qtx.optimizer.AdamSR,
        }
        try:
            optimizer_cls = optimizer_map[optimizer_name]
        except KeyError as exc:
            raise ValueError(
                "Unknown Quantax optimizer "
                f"{self.optimizer_spec!r}. Supported values are {sorted(optimizer_map)!r}."
            ) from exc

        return optimizer_cls(
            state=self.state,
            hamiltonian=self.quantax_hamiltonian,
            solver=solver,
            **optimizer_kwargs,
        )

    def _resolve_sampler_name(self) -> str:
        if not isinstance(self.sampler_spec, str):
            return _callable_name(self.sampler_spec)

        sampler_name = self.sampler_spec.strip().lower()
        if sampler_name != "auto":
            return sampler_name

        hilbert_dim = _hilbert_dimension(
            self.space.num_spatial_orbitals,
            self.n_particles,
        )
        if hilbert_dim <= self.exact_sampler_cutoff and _QUSPIN_IMPORT_ERROR is None:
            return "exact"
        return "particle_hop"

    def _build_sampler(self):
        if not isinstance(self.sampler_spec, str):
            if self.sampler_kwargs:
                raise TypeError("Custom Quantax sampler instances do not accept `sampler_kwargs`.")
            self._resolved_sampler_name = _callable_name(self.sampler_spec)
            self._exact_sampler_symm = None
            return self.sampler_spec

        sampler_name = self._resolve_sampler_name()
        sampler_kwargs = dict(self.sampler_kwargs)
        nsamples = _resolve_device_multiple(self.nsamples)
        self._resolved_sampler_name = sampler_name
        self._exact_sampler_symm = None

        if sampler_name == "exact":
            _require_quspin_for_exact_sampling()
            reweight = float(sampler_kwargs.pop("reweight", 2.0))
            symm = sampler_kwargs.pop("symm", None)
            self._exact_sampler_symm = self.state.symm if symm is None else symm
            if sampler_kwargs:
                raise TypeError(
                    f"Unsupported sampler kwargs for ExactSampler: {sorted(sampler_kwargs)!r}."
                )
            return qtx.sampler.ExactSampler(
                self.state,
                nsamples=nsamples,
                reweight=reweight,
                symm=symm,
            )

        if sampler_name == "random":
            if sampler_kwargs:
                raise TypeError(
                    f"Unsupported sampler kwargs for RandomSampler: {sorted(sampler_kwargs)!r}."
                )
            return qtx.sampler.RandomSampler(
                self.state,
                nsamples=nsamples,
            )

        if sampler_name == "particle_hop":
            neighbors = _build_neighbor_table(
                self.space.num_spatial_orbitals,
                self.compilation.hopping_graph,
                allow_spin_mixing=self._spin_mixing_enabled,
            )
            sampler = GraphParticleHop(
                self.state,
                nsamples=nsamples,
                neighbors=neighbors,
                reweight=float(sampler_kwargs.pop("reweight", 2.0)),
                thermal_steps=sampler_kwargs.pop("thermal_steps", None),
                sweep_steps=sampler_kwargs.pop("sweep_steps", None),
                initial_spins=sampler_kwargs.pop("initial_spins", None),
            )
            if sampler_kwargs:
                raise TypeError(
                    f"Unsupported sampler kwargs for GraphParticleHop: {sorted(sampler_kwargs)!r}."
                )
            return sampler

        raise ValueError(
            "Unknown Quantax sampler "
            f"{self.sampler_spec!r}. Supported values are ('auto', 'exact', 'random', 'particle_hop')."
        )

    def _build_quantax_hamiltonian(self):
        if self.compilation.op_list:
            return qtx.operator.Operator(self.compilation.op_list)
        return qtx.operator.Operator([])

    def _deterministic_exact_samples(self, nsamples: int | None = None):
        symm = self.state.symm if self._exact_sampler_symm is None else self._exact_sampler_symm
        if symm is not self.state.symm:
            return self.sampler.sweep()

        dense_state = self.state.todense(symm)
        basis_ints = np.asarray(symm.basis.states)
        psi = np.asarray(dense_state.psi)
        prob = np.abs(psi) ** 2
        mask = prob > 0.0
        basis_ints = basis_ints[mask]
        psi = psi[mask]
        prob = prob[mask]
        prob = prob / prob.sum()

        nstates = basis_ints.size
        total_samples = nstates if nsamples is None else max(int(nsamples), nstates)
        counts = np.ones(nstates, dtype=np.int32)
        remaining = total_samples - nstates
        if remaining > 0:
            ideal_extra = remaining * prob
            extra = np.floor(ideal_extra).astype(np.int32)
            counts += extra
            leftover = remaining - int(extra.sum())
            if leftover > 0:
                order = np.argsort(-(ideal_extra - extra))
                counts[order[:leftover]] += 1

        basis_ints = np.repeat(basis_ints, counts)
        psi = np.repeat(psi, counts)
        prob_per_sample = np.repeat(prob / counts, counts)
        spins = ints_to_array(basis_ints)
        reweight_factor = total_samples * prob_per_sample
        return qtx.sampler.Samples(
            jnp.asarray(spins),
            jnp.asarray(psi),
            None,
            jnp.asarray(reweight_factor),
        )

    def _training_samples(self):
        if self._resolved_sampler_name == "exact":
            return self._deterministic_exact_samples(self.nsamples)
        return self.sampler.sweep()

    def _measure(self, operator: SymbolicOperator) -> tuple[complex, float, int]:
        self._validate_observable_space(operator)
        reduced = _sector_preserving_part(
            operator,
            sector_kind=self._particle_sector_kind,
        )
        compilation = compile_quantax_hamiltonian(reduced)
        mean = complex(compilation.constant_shift)
        variance = 0.0

        if compilation.op_list:
            quantax_operator = qtx.operator.Operator(compilation.op_list)
            Oloc = quantax_operator.Oloc(self.state, self.measurement_samples)
            reweight = np.asarray(self.measurement_samples.reweight_factor)
            Oloc = np.asarray(Oloc)
            weighted = Oloc * reweight
            measured_mean = np.mean(weighted)
            mean += complex(measured_mean)
            variance = float(np.real(np.mean(np.abs(Oloc) ** 2 * reweight) - np.abs(measured_mean) ** 2))

        return mean, max(variance, 0.0), int(self.measurement_samples.nsamples)

    def solve(self, hamiltonian, *, n_electrons=None, n_particles=None):
        operator = coerce_symbolic_operator(hamiltonian)
        _validate_quantax_problem(operator)

        self.operator = operator
        self.space = operator.space
        self.n_particles, self.n_electrons, self._particle_sector_kind = _resolve_quantax_particle_sector(
            space=operator.space,
            n_electrons=n_electrons,
            n_particles=n_particles,
        )
        if self.n_particles is None:
            raise ValueError("Quantax NQS v1 requires a fixed particle sector.")

        spin_changes = _spin_changes(operator)
        self._spin_mixing_enabled = any(change != 0 for change in spin_changes)
        if self._spin_mixing_enabled and self._particle_sector_kind != "total":
            raise ValueError(
                "Spin-changing Hamiltonians require a total particle-count sector in Quantax NQS. "
                "Pass `n_electrons=<int>` or `n_particles=<int>`, or provide the full list of spin "
                "partitions for that total particle number."
            )

        norb = self.space.num_spatial_orbitals
        if isinstance(self.n_particles, tuple):
            n_up, n_down = self.n_particles
            if not (0 <= n_up <= norb and 0 <= n_down <= norb):
                raise ValueError(
                    "Requested particle counts must fit inside the spatial-orbital space; "
                    f"got {(n_up, n_down)} for {norb} spatial orbitals."
                )
        else:
            if not (0 <= int(self.n_particles) <= 2 * norb):
                raise ValueError(
                    "Requested total particle count must fit inside the spin-orbital space; "
                    f"got {self.n_particles} for {2 * norb} spin orbitals."
                )

        self.compilation = compile_quantax_hamiltonian(operator)
        self._resolved_dtype = _resolve_jax_dtype(operator, self.requested_dtype)
        self._history = {"energy": [], "variance": []}
        self._last_energy = None
        self._last_variance = None

        qtx.sites.Sites._SITES = None
        qtx.set_default_dtype(self._resolved_dtype)
        qtx.set_random_seed(self.seed)
        _reset_quantax_symmetry_cache()

        self.quantax_sites = qtx.sites.Sites(
            self.space.num_spatial_orbitals,
            particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
            Nparticles=self.n_particles,
            double_occ=True,
            coord=_expand_site_coordinates(self.space),
        )
        self.quantax_hamiltonian = self._build_quantax_hamiltonian()
        model = self._build_model()
        self.state = qtx.state.Variational(
            model,
            max_parallel=self.max_parallel,
        )
        if jax.device_count() == 1:
            self.state._model = _single_device_model(model)
        self.optimizer = self._build_optimizer()
        self.sampler = self._build_sampler()

        for _ in range(self.n_steps):
            samples = self._training_samples()
            step = self.optimizer.get_step(samples)
            self.state.update(step * self.step_size)
            if jax.device_count() == 1:
                self.state._model = _single_device_model(self.state._model)
            self._history["energy"].append(float(self.optimizer.energy))
            self._history["variance"].append(float(self.optimizer.VarE))

        if self._resolved_sampler_name == "exact":
            self.measurement_samples = self._deterministic_exact_samples(self.nsamples)
        else:
            self.measurement_samples = self.sampler.sweep()
        self._last_energy, self._last_variance, _ = self._measure(self.operator)
        return self

    def _expectation(self, operator: SymbolicOperator):
        value, _, _ = self._measure(operator)
        return np.real_if_close(value)

    def expect(self, operator, *, stats: bool = False):
        self._require_solution()
        operator = coerce_symbolic_operator(operator)
        value, variance, n_samples = self._measure(operator)
        value = np.real_if_close(value)
        if not stats:
            return value
        stderr = float(math.sqrt(max(variance, 0.0) / max(n_samples, 1)))
        return {
            "mean": value,
            "stderr": stderr,
            "variance": variance,
            "n_samples": n_samples,
            "kind": "stochastic",
            "backend": self.backend_name,
        }

    def energy(self):
        self._require_solution()
        return np.real_if_close(self._last_energy)

    def rdm1(self):
        self._require_solution()
        nmode = self.space.num_spin_orbitals
        rdm = np.zeros((nmode, nmode), dtype=np.complex128)
        for p in range(nmode):
            for q in range(nmode):
                operator = SymbolicOperator(
                    self.space,
                    terms={Term((Ladder(p, "create"), Ladder(q, "destroy"))): 1.0},
                )
                rdm[p, q] = self._expectation(operator)
        return np.real_if_close(rdm)

    def docc(self):
        self._require_solution()
        values = []
        for site in range(self.space.num_sites):
            for orbital in self.space.orbitals:
                operator = (
                    self.space.number(site, orbital=orbital, spin="up")
                    @ self.space.number(site, orbital=orbital, spin="down")
                )
                values.append(np.real_if_close(self._expectation(operator)).item())
        return np.asarray(values, dtype=float)

    def s2(self):
        self._require_solution()
        return np.real_if_close(self._expectation(self.space.spin_squared_term()))

    def diagnostics(self) -> dict[str, object]:
        data = super().diagnostics()
        data.update(
            {
                "model": self._resolved_model_name,
                "optimizer": self._resolved_optimizer_name,
                "linear_solver": self._resolved_linear_solver_name,
                "sampler": self._resolved_sampler_name,
                "n_particles": self.n_particles,
                "particle_sector_kind": self._particle_sector_kind,
                "spin_mixing_enabled": self._spin_mixing_enabled,
                "nsamples": self.nsamples,
                "n_steps": self.n_steps,
                "step_size": self.step_size,
                "seed": self.seed,
                "exact_sampler_cutoff": self.exact_sampler_cutoff,
                "dtype": None if self._resolved_dtype is None else np.dtype(self._resolved_dtype).name,
                "quantax_source": _QUANTAX_SOURCE,
                "device_count": None if jax is None else int(jax.device_count()),
                "last_energy": None if self._last_energy is None else float(np.real(self._last_energy)),
                "last_variance": self._last_variance,
                "constant_shift": None
                if self.compilation is None
                else np.real_if_close(self.compilation.constant_shift),
                "training_history": {
                    "energy": list(self._history["energy"]),
                    "variance": list(self._history["variance"]),
                },
            }
        )
        return data

    def E(self):
        return self.energy()

    def RDM(self):
        return self.rdm1()

    def S2(self):
        return self.s2()
