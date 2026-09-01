# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib
import threading

import pytest

from tests.helpers.monitor import DeviceMemoryMonitor
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_monitor_rejects_non_positive_or_non_finite_interval(interval: float):
    with pytest.raises(ValueError, match="interval must be a positive finite number"):
        DeviceMemoryMonitor(device_index=0, interval=interval)


def test_stop_interrupts_monitor_poll_interval(monkeypatch: pytest.MonkeyPatch):
    sampled = threading.Event()

    monkeypatch.setattr(current_omni_platform, "device", lambda _index: contextlib.nullcontext())

    def _sample_memory() -> tuple[int, int]:
        sampled.set()
        return 0, 0

    monkeypatch.setattr(current_omni_platform, "mem_get_info", _sample_memory)

    monitor = DeviceMemoryMonitor(device_index=0, interval=60.0)
    monitor.start()
    assert sampled.wait(timeout=1.0)

    monitor.stop()

    assert monitor._thread is not None
    assert not monitor._thread.is_alive()
