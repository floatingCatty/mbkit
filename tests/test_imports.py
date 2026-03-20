import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_imports_do_not_require_hubbard():
    code = """
import sys
sys.modules['hubbard'] = None
import mbkit
import mbkit.operator
import mbkit.operator.interop
import mbkit.operator.integrals
import mbkit.operator.quadratic
import mbkit.operator.transforms.qc
import mbkit.operator.transforms
import mbkit.solver
import mbkit.solver.base
import mbkit.solver.capabilities
import mbkit.solver.compile
import mbkit.solver.compile.local_terms
import mbkit.solver.nqs_solver
import mbkit.solver.registry
import mbkit.utils
import mbkit.utils.construction
import mbkit.utils.dependencies
import mbkit.utils.selection
import mbkit.utils.solver
from mbkit import chain, cubic, general, honeycomb, kagome, ladder, rectangular, square, triangular
from mbkit import UHFSolver, MP2Solver, CCSDSolver, CCSDTSolver, NQSSolver
print('ok')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
