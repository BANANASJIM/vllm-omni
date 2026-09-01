# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from tests.model_executor import helpers

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bootstrap_skips_missing_module_but_propagates_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    registration_error = RuntimeError("registration failed")

    def fake_import_module(name: str) -> None:
        calls.append(name)
        if name == "missing":
            raise ModuleNotFoundError(name)
        raise registration_error

    monkeypatch.setattr(helpers, "_VLLM_PREIMPORT_MODULES", ("missing", "broken"))
    monkeypatch.setattr(helpers.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="registration failed") as exc_info:
        helpers.bootstrap_vllm_layer_custom_op_modules()

    assert exc_info.value is registration_error
    assert calls == ["missing", "broken"]
