# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.dfx.reliability import helpers

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RollbackClock:
    def __init__(self) -> None:
        self.wall_calls = 0
        self.monotonic_calls = 0
        self._monotonic_values = iter((0.0, 0.0, 1.0))

    def time(self) -> float:
        self.wall_calls += 1
        return 100.0 if self.wall_calls == 1 else 90.0

    def monotonic(self) -> float:
        self.monotonic_calls += 1
        return next(self._monotonic_values)


class _FakeProc:
    stdout = object()

    def __init__(self) -> None:
        self._poll_results = iter((None, 0))
        self.terminate_calls = 0

    def poll(self) -> int | None:
        return next(self._poll_results)

    def terminate(self) -> None:
        self.terminate_calls += 1


def test_start_gpu_oom_hog_uses_monotonic_startup_deadline(monkeypatch) -> None:
    proc = _FakeProc()
    monkeypatch.setattr(helpers.subprocess, "Popen", lambda *args, **kwargs: proc)
    select_calls = []

    def fake_select(read, write, error, timeout):
        select_calls.append((read, write, error, timeout))
        return [], [], []

    monkeypatch.setattr(helpers.select, "select", fake_select)
    clock = _RollbackClock()
    monkeypatch.setattr(helpers, "time", clock)

    with pytest.raises(TimeoutError, match="OOM sidecar startup timeout"):
        helpers.start_gpu_oom_hog(
            target_mem_ratio=0.5,
            startup_timeout_sec=1,
            poll_interval_sec=0,
        )

    assert clock.wall_calls == 0
    assert clock.monotonic_calls == 3
    assert select_calls == [([proc.stdout], [], [], 0)]
    assert proc.terminate_calls == 1
