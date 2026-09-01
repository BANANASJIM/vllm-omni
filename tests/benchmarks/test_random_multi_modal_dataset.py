# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64

import pytest

from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import OmniRandomMultiModalDataset

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def test_synthetic_audio_data_url_matches_wav_payload() -> None:
    dataset = OmniRandomMultiModalDataset(random_seed=0, dataset_path=None)

    item = dataset.generate_mm_item((0, 1, 1))
    url = item["audio_url"]["url"]

    assert url.startswith("data:audio/wav;base64,")
    audio_bytes = base64.b64decode(url.partition(",")[2])
    assert audio_bytes[:4] == b"RIFF"
    assert audio_bytes[8:12] == b"WAVE"
