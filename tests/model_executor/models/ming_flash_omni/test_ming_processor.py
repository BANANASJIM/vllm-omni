# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any, cast

import pytest
from transformers.processing_utils import ProcessorMixin

from vllm_omni.transformers_utils.processors.ming import (
    MingFlashOmniProcessor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("existing_template", "tokenizer_template", "expected"),
    [
        ("existing", "fallback", "existing"),
        (None, "fallback", "fallback"),
    ],
)
def test_init_preserves_or_falls_back_chat_template(
    monkeypatch: pytest.MonkeyPatch,
    existing_template: str | None,
    tokenizer_template: str,
    expected: str,
) -> None:
    def fake_processor_init(processor: Any, **_kwargs: Any) -> None:
        processor.chat_template = existing_template

    monkeypatch.setattr(ProcessorMixin, "__init__", fake_processor_init)
    processor = MingFlashOmniProcessor(
        image_processor=object(),
        audio_processor=object(),
        tokenizer=SimpleNamespace(chat_template=tokenizer_template),
    )

    assert processor.chat_template == expected


def _apply_chat_template(conversation: list[dict[str, str]]) -> str:
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token="<eos>"),
        apply_system_template=lambda *_args, **_kwargs: "<system>",
    )
    return MingFlashOmniProcessor.apply_chat_template(
        cast(Any, processor),
        conversation,
    )


def test_apply_chat_template_rejects_invalid_role() -> None:
    with pytest.raises(ValueError) as exc_info:
        _apply_chat_template([{"role": "SYSTEM", "content": "Hello"}])
    assert str(exc_info.value) == "Invalid role: SYSTEM. Must be 'HUMAN' or 'ASSISTANT'"


def test_apply_chat_template_rejects_assistant_last_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        _apply_chat_template([{"role": "ASSISTANT", "content": "Hello"}])
    assert str(exc_info.value) == "Last message must be from HUMAN"


def test_apply_chat_template_preserves_valid_conversation() -> None:
    result = _apply_chat_template(
        [
            {"role": "HUMAN", "content": "First"},
            {"role": "ASSISTANT", "content": "Second"},
            {"role": "HUMAN", "content": "Third"},
        ]
    )

    assert result == (
        "<system><eos>"
        "<role>HUMAN</role>First<eos>"
        "<role>ASSISTANT</role>Second<eos>"
        "<role>HUMAN</role>Third<eos>"
        "<role>ASSISTANT</role>"
    )
