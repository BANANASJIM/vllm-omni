# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Incremental Ogg/Opus encoding for streaming speech responses."""

from fractions import Fraction

import av
import numpy as np

OPUS_SAMPLE_RATE = 48000


class _StreamingBuffer:
    """Keep only bytes the muxer has not handed to the HTTP stream yet."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self._position = 0

    def write(self, data: bytes) -> int:
        self._pending.extend(data)
        self._position += len(data)
        return len(data)

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        data = bytes(self._pending)
        self._pending.clear()
        return data


class OggOpusEncoder:
    """Encode one continuous mono Ogg/Opus logical stream."""

    def __init__(self) -> None:
        self._buffer = _StreamingBuffer()
        self._container = av.open(
            self._buffer,
            mode="w",
            format="ogg",
            options={"page_duration": "20000"},
        )
        self._stream = self._container.add_stream(
            "libopus",
            rate=OPUS_SAMPLE_RATE,
            options={"frame_duration": "20"},
        )
        self._stream.layout = "mono"
        self._container.start_encoding()
        self._resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=OPUS_SAMPLE_RATE,
        )
        self._source_rate: int | None = None
        self._source_pts = 0
        self._closed = False

    def encode(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode one source-rate PCM chunk and return newly muxed bytes."""
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return b""

        if self._source_rate is None:
            self._source_rate = sample_rate
        elif sample_rate != self._source_rate:
            raise ValueError(
                f"Audio sample rate changed during Opus streaming: {self._source_rate} Hz to {sample_rate} Hz"
            )

        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(chunk.reshape(1, -1)),
            format="flt",
            layout="mono",
        )
        frame.sample_rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)
        frame.pts = self._source_pts
        self._source_pts += int(chunk.size)

        for resampled_frame in self._resampler.resample(frame):
            self._encode_frame(resampled_frame)
        return self._take_bytes()

    def finish(self) -> bytes:
        """Drain the resampler and encoder, then finish the Ogg stream."""
        if self._closed:
            return b""
        try:
            for resampled_frame in self._resampler.resample(None):
                self._encode_frame(resampled_frame)
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        finally:
            try:
                self._container.close()
            finally:
                self._closed = True
        return self._take_bytes()

    def close(self) -> None:
        """Close without explicitly draining buffered audio."""
        if self._closed:
            return
        try:
            self._container.close()
        finally:
            self._closed = True

    def _encode_frame(self, frame: av.AudioFrame) -> None:
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def _take_bytes(self) -> bytes:
        return self._buffer.take()


__all__ = ["OPUS_SAMPLE_RATE", "OggOpusEncoder"]
