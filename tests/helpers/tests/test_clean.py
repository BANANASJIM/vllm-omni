# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib

import pytest

from tests.helpers import clean

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _BusyPlatform:
    device_control_env_var = "CUDA_VISIBLE_DEVICES"

    @staticmethod
    def device(_device: int):
        return contextlib.nullcontext()

    @staticmethod
    def mem_get_info() -> tuple[int, int]:
        return 0, 2**30

    @staticmethod
    def empty_cache() -> None:
        pass


def test_memory_clear_timeout_uses_monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monotonic_values = iter([10.0, 10.5, 11.0])

    def _unexpected_wall_clock() -> float:
        raise AssertionError("relative timeout consulted the wall clock")

    monkeypatch.delenv(_BusyPlatform.device_control_env_var, raising=False)
    monkeypatch.setattr(clean, "current_omni_platform", _BusyPlatform())
    monkeypatch.setattr(clean.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(clean.time, "time", _unexpected_wall_clock)
    monkeypatch.setattr(clean.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(clean.gc, "collect", lambda: 0)

    with pytest.raises(ValueError, match=r"after 1\.0 seconds"):
        clean.wait_for_gpu_memory_to_clear(
            devices=[0],
            threshold_ratio=0.5,
            timeout_s=1.0,
        )
