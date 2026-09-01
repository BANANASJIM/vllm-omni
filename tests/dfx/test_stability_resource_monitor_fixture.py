# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
import threading
from types import SimpleNamespace

import pytest

from tests.dfx.stability import conftest as stability_conftest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeProcess:
    def __init__(self, events: list[str], wait_times_out: bool) -> None:
        self.events = events
        self.wait_times_out = wait_times_out

    def poll(self) -> None:
        self.events.append("proc.poll")

    def terminate(self) -> None:
        self.events.append("proc.terminate")

    def kill(self) -> None:
        self.events.append("proc.kill")

    def wait(self, timeout: int | None = None) -> None:
        self.events.append(f"proc.wait:{timeout}")
        if timeout == 10 and self.wait_times_out:
            raise subprocess.TimeoutExpired(cmd="resource-monitor", timeout=timeout)


@pytest.mark.parametrize(
    ("wait_times_out", "expected_process_cleanup"),
    [
        (False, ["proc.terminate", "proc.wait:10"]),
        (True, ["proc.terminate", "proc.wait:10", "proc.kill", "proc.wait:None"]),
    ],
)
def test_resource_monitor_cleanup_runs_when_fixture_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    wait_times_out: bool,
    expected_process_cleanup: list[str],
) -> None:
    events: list[str] = []
    proc = FakeProcess(events, wait_times_out)
    reporter_threads: list[threading.Thread] = []
    join_timeouts: list[float | None] = []
    reporter_started = threading.Event()

    class RecordingThread(threading.Thread):
        def join(self, timeout: float | None = None) -> None:
            join_timeouts.append(timeout)
            super().join(timeout)

    def run_reporter(stop_event: threading.Event) -> None:
        reporter_threads.append(threading.current_thread())
        events.append("reporter.start")
        reporter_started.set()
        assert stop_event.wait(timeout=1)
        events.append("reporter.stop")

    def fail_wait_for_run_dir(*, timeout_sec: int):
        assert timeout_sec == 5
        assert reporter_started.wait(timeout=1)
        events.append("wait_for_run_dir")
        raise RuntimeError("setup failed")

    fake_threading = SimpleNamespace(Event=threading.Event, Thread=RecordingThread)
    monkeypatch.setattr(stability_conftest, "threading", fake_threading)
    monkeypatch.setattr(stability_conftest, "start_resource_monitor", lambda: proc)
    monkeypatch.setattr(stability_conftest, "report_latest_gpu_samples", run_reporter)
    monkeypatch.setattr(stability_conftest, "wait_for_run_dir", fail_wait_for_run_dir)
    monkeypatch.setattr(stability_conftest, "finalize_resource_monitor", lambda: events.append("finalize"))

    fixture = stability_conftest.stability_resource_monitor_per_test.__wrapped__(  # type: ignore[attr-defined]
        SimpleNamespace(node=SimpleNamespace(name="setup-failure"))
    )
    with pytest.raises(RuntimeError, match="setup failed"):
        next(fixture)

    assert join_timeouts == [None]
    assert len(reporter_threads) == 1
    assert not reporter_threads[0].is_alive()
    assert events == [
        "reporter.start",
        "wait_for_run_dir",
        "reporter.stop",
        "proc.poll",
        *expected_process_cleanup,
        "finalize",
    ]
