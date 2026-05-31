# SPDX-License-Identifier: Apache-2.0

import math
import sys
from importlib import util
from pathlib import Path

import torch
import torch.distributed as dist


def _load_compute_sequence_ess():
    # Keep this focused math test from paying heavyweight PPO package imports
    # in each torchrun child process.
    ess_path = Path(__file__).resolve().parents[2] / "areal/trainer/ppo/ess.py"
    spec = util.spec_from_file_location("_ppo_ess_under_test", ess_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.compute_sequence_ess


compute_sequence_ess = _load_compute_sequence_ess()


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        log_weight = math.log(100.0) if rank == 0 else 0.0
        logprobs = torch.zeros((1, 1))
        prox_logp = torch.tensor([[log_weight]])
        loss_mask = torch.ones_like(logprobs)

        stats = compute_sequence_ess(
            prox_logp=prox_logp,
            logprobs=logprobs,
            loss_mask=loss_mask,
            process_group=dist.group.WORLD,
        )

        assert stats is not None
        expected_ess = (101.0**2) / (100.0**2 + 1.0)
        assert stats.valid_sequence_count == 2
        torch.testing.assert_close(stats.ess, expected_ess)
        torch.testing.assert_close(stats.ess_ratio, expected_ess / 2.0)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
