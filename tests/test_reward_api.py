import asyncio
import time

import pytest

from areal.api.reward_api import AsyncRewardWrapper


def _pathological_math_reward() -> float:
    from areal.reward import MathVerifyWorker

    return MathVerifyWorker(timeout=1).verify(r"$2^{2^{2008}}$", "1")


def _record_attempt_and_sleep(path: str, sleep_seconds: float) -> float:
    with open(path, "a") as f:
        f.write("attempt\n")
        f.flush()
    time.sleep(sleep_seconds)
    return 1.0


def _count_attempts(path) -> int:
    if not path.exists():
        return 0
    return path.read_text().count("attempt\n")


def _shutdown_reward_executors():
    AsyncRewardWrapper._atexit_shutdown_all()


@pytest.mark.asyncio
async def test_async_reward_wrapper_math_verify_timeout_finishes_before_outer_timeout():
    """Math-verify should finish inside the reward process before outer timeout."""
    wrapper = AsyncRewardWrapper(
        _pathological_math_reward,
        timeout_seconds=5,
        max_workers=1,
        max_retries=0,
    )

    try:
        start = time.monotonic()
        reward = await wrapper()
        elapsed = time.monotonic() - start

        assert reward == 0.0
        assert elapsed < 3
    finally:
        _shutdown_reward_executors()


@pytest.mark.asyncio
async def test_async_reward_wrapper_timeout_does_not_retry(tmp_path):
    """Timed-out reward calls should not submit duplicate attempts."""
    attempts_path = tmp_path / "attempts.txt"
    wrapper = AsyncRewardWrapper(
        _record_attempt_and_sleep,
        timeout_seconds=0.05,
        max_workers=4,
        max_retries=3,
    )

    try:
        reward = await wrapper(str(attempts_path), 0.5)
        assert reward == 0

        deadline = time.monotonic() + 2
        while _count_attempts(attempts_path) == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.3)
        assert _count_attempts(attempts_path) == 1
    finally:
        _shutdown_reward_executors()
