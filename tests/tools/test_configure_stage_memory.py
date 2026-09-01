# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import huggingface_hub
import pytest

from tools.configure_stage_memory import get_model_size_gib

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_model_size_gib_reports_query_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_model_info(model: str) -> None:
        assert model == "org/model"
        raise RuntimeError("offline registry")

    monkeypatch.setattr(huggingface_hub, "model_info", fail_model_info)

    assert get_model_size_gib("org/model") is None
    stderr = capsys.readouterr().err
    assert "Warning" in stderr
    assert "org/model" in stderr
    assert "offline registry" in stderr


def test_get_model_size_gib_converts_parameter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda _model: SimpleNamespace(safetensors=SimpleNamespace(total=2**30)),
    )

    assert get_model_size_gib("org/model") == 2.0
