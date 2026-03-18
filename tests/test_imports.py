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
import mbkit.operator.integrals
import mbkit.operator.models
import mbkit.operator.models.hopping
import mbkit.operator.models.observables
import mbkit.operator.models.soc
import mbkit.operator.transforms.qc
import mbkit.operator.transforms
import mbkit.solver
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
