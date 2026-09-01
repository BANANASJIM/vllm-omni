# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import urllib.request

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_audio_url_uses_media_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid_cuda(*_args, **_kwargs) -> None:
        raise RuntimeError("URL audio loading must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbid_cuda)

    from vllm.multimodal.media import MediaConnector

    from vllm_omni.model_executor.models.qwen3_tts import qwen3_tts_tokenizer as tokenizer_module

    url = "https://example.invalid/audio.wav"
    stereo_audio = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    fetched_urls: list[str] = []

    def _fetch_audio(_self: MediaConnector, audio_url: str) -> tuple[np.ndarray, int]:
        assert _self.allowed_local_media_path is None
        fetched_urls.append(audio_url)
        return stereo_audio, 8000

    class _FakeResampler:
        def __init__(self, target_sr: int) -> None:
            assert target_sr == 16000

        def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
            assert orig_sr == 8000
            return audio + 1.0

    def _forbid_urlopen(*_args, **_kwargs):
        pytest.fail("URL audio loading bypassed MediaConnector")

    monkeypatch.setattr(MediaConnector, "fetch_audio", _fetch_audio)
    monkeypatch.setattr(tokenizer_module, "AudioResampler", _FakeResampler)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_urlopen)

    tokenizer = tokenizer_module.Qwen3TTSTokenizer()

    audio = tokenizer.load_audio(url, target_sr=16000)

    assert fetched_urls == [url]
    np.testing.assert_array_equal(audio, np.array([2.0, 4.0], dtype=np.float32))
    assert audio.dtype == np.float32
