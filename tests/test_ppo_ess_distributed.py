# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

_TORCHRUN_TIMEOUT_SECONDS = 120


def _output_text(output: bytes | str | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def test_compute_sequence_ess_is_global_over_data_parallel_group():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "tests/torchrun/run_ppo_ess_distributed.py",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=_TORCHRUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _output_text(exc.stdout)
        stderr = _output_text(exc.stderr)
        raise AssertionError(
            f"torchrun timed out after {_TORCHRUN_TIMEOUT_SECONDS} seconds\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        ) from exc

    assert result.returncode == 0, result.stdout + result.stderr
