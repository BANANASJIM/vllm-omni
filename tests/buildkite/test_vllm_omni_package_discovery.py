# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard package discovery outside repo-root ``sys.path[0]``.

Covers the same failure class as:
- ``vllm ... --omni`` (upstream ``find_spec("vllm_omni")``) — see #5364
- ``python examples/.../end2end.py`` / multimodal client subprocesses, where
  ``sys.path[0]`` is the script directory (not the repo root), so a cwd-only
  layout can start OmniServer via ``python -m vllm_omni...`` and still fail
  example scripts with ``ModuleNotFoundError: No module named 'vllm_omni'``.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DISCOVERY_TIMEOUT_SECONDS = 60


def _run_discovery_subprocess(
    command: list[str],
    *,
    cwd: Path,
    timeout: float = _DISCOVERY_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _assert_subprocess_ok(proc: subprocess.CompletedProcess[str], *, detail: str) -> None:
    assert proc.returncode == 0, f"{detail} (exit={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_find_spec_vllm_omni() -> None:
    """Same probe as upstream ``vllm`` CLI before delegating to omni_main."""
    assert find_spec("vllm_omni") is not None
    assert find_spec("vllm_omni.entrypoints.cli.main") is not None


def test_find_spec_vllm_omni_from_non_repo_cwd(tmp_path: Path) -> None:
    """Console-script / ``python -c`` style: discovery when cwd is not the repo root."""
    code = (
        "from importlib.util import find_spec; "
        "assert find_spec('vllm_omni') is not None; "
        "assert find_spec('vllm_omni.entrypoints.cli.main') is not None"
    )
    proc = _run_discovery_subprocess(
        [sys.executable, "-c", code],
        cwd=tmp_path,
    )
    _assert_subprocess_ok(proc, detail=f"find_spec failed from cwd={tmp_path}")


def test_python_script_off_repo_root_sys_path(tmp_path: Path) -> None:
    """``python path/to/script.py``: ``sys.path[0]`` is the script dir, not the repo root.

    One case covers the end2end / online-client / orphan-dir layouts — they all
    share this discovery constraint. Use ``find_spec`` only (no heavy imports)
    so Simple Test stays fast.
    """
    script = tmp_path / "examples" / "offline_inference" / "hunyuan_image3" / "end2end_probe.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "\n".join(
            [
                "from importlib.util import find_spec",
                "import sys",
                'print("initial_sys_path0", sys.path[0])',
                'assert find_spec("vllm_omni") is not None, (sys.path[0], sys.path[:5])',
                'assert find_spec("vllm_omni.entrypoints.cli.main") is not None',
                'print("ok")',
                "",
            ]
        ),
        encoding="utf-8",
    )
    cwd = tmp_path / "run_cwd"
    cwd.mkdir()
    proc = _run_discovery_subprocess(
        [sys.executable, str(script)],
        cwd=cwd,
    )
    _assert_subprocess_ok(
        proc,
        detail=f"find_spec failed for python {script} with cwd={cwd}",
    )
    assert f"initial_sys_path0 {script.parent}" in proc.stdout


@pytest.mark.parametrize("as_script", [False, True], ids=["python-c", "python-script"])
def test_discovery_subprocess_watchdog_is_diagnostic(tmp_path: Path, as_script: bool) -> None:
    code = "import time; time.sleep(1)"
    if as_script:
        script = tmp_path / "slow_probe.py"
        script.write_text(code, encoding="utf-8")
        command = [sys.executable, str(script)]
    else:
        command = [sys.executable, "-c", code]

    with pytest.raises(subprocess.TimeoutExpired, match="timed out after") as exc_info:
        _run_discovery_subprocess(command, cwd=tmp_path, timeout=0.1)

    assert exc_info.value.cmd == command
    assert exc_info.value.timeout == pytest.approx(0.1)
