# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the MammothModa2 AR-to-DiT stage input processor."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ar_output(latent: torch.Tensor, *, request_id: str = "req-1") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[11, 12],
        outputs=[
            SimpleNamespace(
                cumulative_token_ids=[21, 22, 23],
                multimodal_output={"latent": latent},
            )
        ],
    )


def test_ar2dit_preserves_matching_tokens_and_hidden_states():
    latent = torch.arange(12, dtype=torch.float16).reshape(4, 3)

    result = ar2dit([_ar_output(latent)])

    additional_information = result[0]["additional_information"]
    assert additional_information["full_token_ids"] == [11, 12, 21, 22]
    hidden_states = additional_information["full_hidden_states"]
    assert hidden_states.shape == (4, 3)
    assert hidden_states.dtype == torch.float32
    torch.testing.assert_close(hidden_states, latent.float())


def test_ar2dit_rejects_hidden_state_length_mismatch():
    with pytest.raises(RuntimeError) as exc_info:
        ar2dit([_ar_output(torch.zeros(3, 2), request_id="req-corrupt")])

    message = str(exc_info.value)
    assert "AR stage hidden states length mismatch" in message
    assert "request_id=req-corrupt" in message
    assert "expected=4" in message
    assert "got=3" in message
