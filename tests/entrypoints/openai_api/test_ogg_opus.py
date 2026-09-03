# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import io
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai import serving_speech as serving_speech_module
from vllm_omni.entrypoints.openai.ogg_opus import OPUS_SAMPLE_RATE, OggOpusEncoder
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_serving() -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = None
    serving._get_tts_adapter = lambda: None
    serving._mark_ref_audio_artifact_ready_for_request = lambda _request_id: None
    serving._discard_ref_audio_artifact_warmup = lambda _request_id: None
    return serving


def _decode_samples(data: bytes) -> tuple[int, int]:
    with av.open(io.BytesIO(data)) as container:
        assert len(container.streams.audio) == 1
        frames = list(container.decode(audio=0))
    assert frames
    return sum(frame.samples for frame in frames), frames[0].sample_rate


@pytest.mark.parametrize("source_rate", [22050, 24000, 44100, 48000])
def test_ogg_opus_stream_is_single_decodable_stream_with_complete_tail(source_rate: int):
    sample_count = source_rate // 10 + 37
    samples = np.sin(np.arange(sample_count, dtype=np.float32) * (2 * np.pi * 440 / source_rate))
    encoder = OggOpusEncoder()

    parts = [encoder.encode(chunk, source_rate) for chunk in np.array_split(samples, 7)]
    parts.append(encoder.finish())
    encoded = b"".join(parts)

    assert encoded.startswith(b"OggS")
    assert encoded.count(b"OpusHead") == 1
    decoded_samples, decoded_rate = _decode_samples(encoded)
    assert decoded_rate == OPUS_SAMPLE_RATE
    assert decoded_samples / decoded_rate == pytest.approx(sample_count / source_rate, abs=1 / source_rate)


def test_ogg_opus_stream_emits_audio_page_before_finish():
    encoder = OggOpusEncoder()
    frame = np.zeros(960, dtype=np.float32)

    header = encoder.encode(frame, OPUS_SAMPLE_RATE)
    audio_page = encoder.encode(frame, OPUS_SAMPLE_RATE)

    assert header.startswith(b"OggS")
    assert b"OpusHead" in header
    assert audio_page.startswith(b"OggS")
    assert b"OpusHead" not in audio_page
    encoded = header + audio_page + encoder.finish()
    with av.open(io.BytesIO(encoded)) as container:
        packet_durations = [packet.duration for packet in container.demux(audio=0) if packet.size]
    assert packet_durations[0] == 960


@pytest.mark.asyncio
async def test_streaming_opus_chunks_report_opus_decode_rate():
    serving = _make_serving()

    async def generate():
        for _ in range(2):
            yield SimpleNamespace(
                multimodal_output={
                    "audio": torch.zeros(480, dtype=torch.float32),
                    "sr": 24000,
                }
            )

    chunks = [
        chunk
        async for chunk in serving._generate_audio_chunks(
            generate(),
            "request-id",
            response_format="opus",
            include_sample_rate=True,
        )
    ]

    assert chunks
    assert {sample_rate for _, sample_rate in chunks} == {OPUS_SAMPLE_RATE}
    _decode_samples(b"".join(data for data, _ in chunks))


@pytest.mark.asyncio
async def test_streaming_opus_closes_encoder_when_consumer_stops(monkeypatch: pytest.MonkeyPatch):
    encoders: list[OggOpusEncoder] = []

    class TrackingEncoder(OggOpusEncoder):
        def __init__(self) -> None:
            super().__init__()
            encoders.append(self)

    monkeypatch.setattr(serving_speech_module, "OggOpusEncoder", TrackingEncoder)
    serving = _make_serving()

    async def generate():
        while True:
            yield SimpleNamespace(
                multimodal_output={
                    "audio": torch.zeros(960, dtype=torch.float32),
                    "sr": OPUS_SAMPLE_RATE,
                }
            )

    stream = serving._generate_audio_chunks(generate(), "request-id", response_format="opus")
    await anext(stream)
    await stream.aclose()

    assert len(encoders) == 1
    assert encoders[0]._closed
