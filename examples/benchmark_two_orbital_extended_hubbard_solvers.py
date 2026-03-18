"""Benchmark solver robustness on the two-orbital extended Hubbard example.

This script builds the same Hamiltonian as
`examples/two_orbital_extended_hubbard.py` and compares several solver backends
on:

1. ground-state energy only
2. ground-state energy plus full one-body density matrix (`rdm1`)

Each solver run happens in its own subprocess so that a timeout, crash, or OOM
in one backend does not abort the full benchmark. This makes the script useful
both as a comparison tool and as a robustness check while the solver stack is
still evolving.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Make the example runnable directly from the source tree without requiring an
# editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.two_orbital_extended_hubbard import build_reference_two_orbital_extended_hubbard


SOLVER_SPECS = {
    "ed_total_n4": {
        "family": "ed",
        "label": "ED total N=4",
        "solve_kwargs": {"n_particles": [(nup, 4 - nup) for nup in range(5)]},
    },
    "ed_sz0": {
        "family": "ed",
        "label": "ED Sz=0",
        "solve_kwargs": {"n_particles": [(2, 2)]},
    },
    "block2": {
        "family": "dmrg",
        "label": "block2 DMRG",
        "constructor_kwargs": {
            "bond_dim": 200,
            "bond_mul": 2,
            "n_sweep": 18,
            "nupdate": 4,
            "iprint": 0,
            "scratch_dir": "/tmp/mbkit_block2_ext_hubbard_bench",
        },
        "solve_kwargs": {"n_particles": (2, 2)},
    },
    "quimb": {
        "family": "dmrg",
        "label": "quimb DMRG",
        "constructor_kwargs": {
            "backend": "quimb",
            "bond_dims": [32, 64, 128],
            "cutoffs": 1e-10,
            "solve_tol": 1e-8,
            "max_sweeps": 10,
            "verbosity": 0,
        },
        "solve_kwargs": {"n_particles": (2, 2)},
    },
    "tenpy": {
        "family": "dmrg",
        "label": "TeNPy DMRG",
        "constructor_kwargs": {
            "backend": "tenpy",
            "chi_max": 128,
            "svd_min": 1e-12,
            "max_sweeps": 10,
            "max_E_err": 1e-8,
        },
        "solve_kwargs": {"n_particles": (2, 2)},
    },
    "pyscf": {
        "family": "qc",
        "label": "PySCF FCI",
        "solve_kwargs": {"n_particles": (2, 2)},
    },
}


def _build_solver(name: str):
    from mbkit import DMRGSolver, EDSolver, PySCFSolver

    spec = SOLVER_SPECS[name]
    kwargs = dict(spec.get("constructor_kwargs", {}))
    family = spec["family"]
    if family == "ed":
        return EDSolver(**kwargs)
    if family == "dmrg":
        return DMRGSolver(**kwargs)
    if family == "qc":
        return PySCFSolver(**kwargs)
    raise KeyError(f"Unknown solver family {family!r} for spec {name!r}.")


def _serialize_matrix(matrix: np.ndarray) -> dict[str, object]:
    array = np.asarray(matrix)
    array = np.real_if_close(array)
    if np.iscomplexobj(array):
        return {
            "dtype": "complex",
            "real": np.asarray(array.real).tolist(),
            "imag": np.asarray(array.imag).tolist(),
        }
    return {
        "dtype": "real",
        "real": np.asarray(array, dtype=float).tolist(),
    }


def _deserialize_matrix(payload: dict[str, object]) -> np.ndarray:
    real = np.asarray(payload["real"], dtype=float)
    if payload.get("dtype") == "complex":
        imag = np.asarray(payload["imag"], dtype=float)
        return real + 1j * imag
    return real


def _run_single_solver(name: str, *, mode: str) -> dict[str, object]:
    solver = _build_solver(name)
    _space, hamiltonian = build_reference_two_orbital_extended_hubbard(2, 2)
    solve_kwargs = dict(SOLVER_SPECS[name].get("solve_kwargs", {}))

    t0 = time.perf_counter()
    solver.solve(hamiltonian, **solve_kwargs)
    solve_time = time.perf_counter() - t0

    result: dict[str, object] = {
        "solver": name,
        "label": SOLVER_SPECS[name]["label"],
        "mode": mode,
        "status": "ok",
        "energy": float(np.real_if_close(solver.energy())),
        "solve_time_s": solve_time,
    }

    if mode == "energy_rdm1":
        t1 = time.perf_counter()
        rdm1 = np.asarray(solver.rdm1())
        result["rdm1_time_s"] = time.perf_counter() - t1
        result["rdm1_trace"] = float(np.real_if_close(np.trace(rdm1)))
        result["rdm1"] = _serialize_matrix(rdm1)

    return result


def _run_child(name: str, *, mode: str) -> int:
    try:
        payload = _run_single_solver(name, mode=mode)
    except Exception as exc:  # pragma: no cover - benchmark path
        payload = {
            "solver": name,
            "label": SOLVER_SPECS[name]["label"],
            "mode": mode,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload), flush=True)
        return 1

    print(json.dumps(payload), flush=True)
    return 0


def _run_isolated(name: str, *, mode: str, timeout: float) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        name,
        "--mode",
        mode,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "solver": name,
            "label": SOLVER_SPECS[name]["label"],
            "mode": mode,
            "status": "timeout",
            "timeout_s": timeout,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    payload = None
    if lines:
        last_line = lines[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        status = "ok" if completed.returncode == 0 else "process_error"
        return {
            "solver": name,
            "label": SOLVER_SPECS[name]["label"],
            "mode": mode,
            "status": status,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    payload["returncode"] = completed.returncode
    if completed.returncode == 137 or completed.returncode == -9:
        payload["status"] = "killed"
    elif completed.returncode != 0 and payload.get("status") == "ok":
        payload["status"] = "process_error"
    return payload


def _format_time(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}s"


def _print_table(title: str, rows: list[dict[str, object]], *, include_rdm1: bool) -> None:
    print(f"\n{title}")
    headers = ["solver", "status", "energy", "dE vs ED Sz=0", "solve time"]
    if include_rdm1:
        headers.extend(["rdm1 time", "trace(rdm1)", "||rdm1-rdm1_ref||_F"])

    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))

    for row in rows:
        values = [
            str(row["label"]),
            str(row["status"]),
            row.get("energy_str", "-"),
            row.get("delta_e_str", "-"),
            _format_time(row.get("solve_time_s")),
        ]
        if include_rdm1:
            values.extend(
                [
                    _format_time(row.get("rdm1_time_s")),
                    row.get("rdm1_trace_str", "-"),
                    row.get("delta_rdm1_str", "-"),
                ]
            )
        print(" | ".join(values))


def _prepare_rows(results: list[dict[str, object]], *, include_rdm1: bool) -> list[dict[str, object]]:
    reference = next((row for row in results if row["solver"] == "ed_sz0" and row["status"] == "ok"), None)
    reference_energy = reference.get("energy") if reference is not None else None
    reference_rdm1 = None
    if include_rdm1 and reference is not None and "rdm1" in reference:
        reference_rdm1 = _deserialize_matrix(reference["rdm1"])

    rows = []
    for row in results:
        prepared = dict(row)
        if row.get("status") == "ok" and "energy" in row:
            prepared["energy_str"] = f"{float(row['energy']):.12f}"
            if reference_energy is not None:
                prepared["delta_e_str"] = f"{float(row['energy']) - float(reference_energy):+.3e}"
        else:
            prepared["energy_str"] = "-"
            prepared["delta_e_str"] = "-"

        if include_rdm1:
            if row.get("status") == "ok" and "rdm1_trace" in row:
                prepared["rdm1_trace_str"] = f"{float(row['rdm1_trace']):.6f}"
            else:
                prepared["rdm1_trace_str"] = "-"

            if row.get("status") == "ok" and "rdm1" in row and reference_rdm1 is not None:
                rdm1 = _deserialize_matrix(row["rdm1"])
                prepared["delta_rdm1_str"] = f"{np.linalg.norm(rdm1 - reference_rdm1):.3e}"
            else:
                prepared["delta_rdm1_str"] = "-"

        rows.append(prepared)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=sorted(SOLVER_SPECS), help="Internal child-run mode.")
    parser.add_argument(
        "--mode",
        choices=("energy", "energy_rdm1"),
        default="energy_rdm1",
        help="Benchmark mode for child or parent runs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-solver timeout in seconds for subprocess-isolated runs.",
    )
    parser.add_argument(
        "--skip-heavy-rdm1",
        action="store_true",
        help="Skip the full rdm1 phase for the heavier tensor-network backends (quimb, tenpy).",
    )
    args = parser.parse_args(argv)

    if args.child:
        return _run_child(args.child, mode=args.mode)

    print("Building benchmark around examples/two_orbital_extended_hubbard.py")
    print(f"Per-solver timeout: {args.timeout:.1f}s")

    energy_results = []
    for name in SOLVER_SPECS:
        result = _run_isolated(name, mode="energy", timeout=args.timeout)
        energy_results.append(result)

    rdm1_results = []
    for name in SOLVER_SPECS:
        if args.skip_heavy_rdm1 and name in {"quimb", "tenpy"}:
            rdm1_results.append(
                {
                    "solver": name,
                    "label": SOLVER_SPECS[name]["label"],
                    "mode": "energy_rdm1",
                    "status": "skipped",
                }
            )
            continue
        result = _run_isolated(name, mode="energy_rdm1", timeout=args.timeout)
        rdm1_results.append(result)

    energy_rows = _prepare_rows(energy_results, include_rdm1=False)
    rdm1_rows = _prepare_rows(rdm1_results, include_rdm1=True)

    _print_table("Energy-only Benchmark", energy_rows, include_rdm1=False)
    _print_table("Energy + RDM1 Benchmark", rdm1_rows, include_rdm1=True)

    ref_total = next((row for row in energy_results if row["solver"] == "ed_total_n4" and row["status"] == "ok"), None)
    ref_sz0 = next((row for row in energy_results if row["solver"] == "ed_sz0" and row["status"] == "ok"), None)
    if ref_total is not None and ref_sz0 is not None:
        delta = float(ref_total["energy"]) - float(ref_sz0["energy"])
        print(f"\nED total-N=4 minus ED Sz=0 energy: {delta:+.3e}")

    print("\nNotes:")
    print("- `ED total N=4` searches all `(n_up, n_down)` sectors with total electron count 4.")
    print("- `ED Sz=0` fixes the sector to `(n_up, n_down) = (2, 2)` and is used as the main `rdm1` reference.")
    print("- Subprocess isolation means a timeout, crash, or OOM in one backend does not abort the benchmark.")
    print("- A backend marked `killed` or `timeout` is a real robustness result for this example size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
