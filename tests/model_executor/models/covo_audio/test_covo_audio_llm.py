# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.covo_audio.covo_audio_llm import AudioAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("downsample", [0, 1, 3, 6])
def test_audio_adapter_rejects_invalid_downsample(downsample: int) -> None:
    with pytest.raises(ValueError, match="power of 2 and at least 2"):
        AudioAdapter(input_dim=4, output_dim=8, downsample=downsample)


@pytest.mark.parametrize(("downsample", "expected_layers"), [(2, 1), (4, 2), (8, 3)])
def test_audio_adapter_builds_one_layer_per_downsample_step(downsample: int, expected_layers: int) -> None:
    adapter = AudioAdapter(input_dim=4, output_dim=8, downsample=downsample)

    assert len(adapter.downsample_layers) == expected_layers
