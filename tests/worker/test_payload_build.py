# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.worker.output import payload_build

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _build(mm_cpu: dict[str, object], *, idx: int, start: int, end: int) -> dict[str, object]:
    return payload_build.build_omni_mm_payload(
        combined_multimodal_outputs=None,
        mm_cpu=mm_cpu,
        rid=f"r{idx}",
        idx=idx,
        start=start,
        end=end,
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=3,
        scheduled_seq_len=3,
    )


@pytest.mark.parametrize("idx", [0, 2])
def test_dense_singleton_list_broadcasts_a_clone(idx: int) -> None:
    source = torch.tensor([11])

    routed = _build({"codes.audio": [source]}, idx=idx, start=idx, end=idx + 1)["codes.audio"]

    assert isinstance(routed, torch.Tensor)
    assert torch.equal(routed, source)
    assert routed.data_ptr() != source.data_ptr()


def test_dense_aligned_list_selects_and_clones_request_value() -> None:
    source = torch.tensor([22])

    routed = _build(
        {"codes.audio": [torch.tensor([11]), source]},
        idx=1,
        start=1,
        end=2,
    )["codes.audio"]

    assert isinstance(routed, torch.Tensor)
    assert torch.equal(routed, source)
    assert routed.data_ptr() != source.data_ptr()


@pytest.mark.parametrize(
    ("values", "idx"),
    [([torch.tensor([11]), torch.tensor([22])], 2), ([], 0)],
    ids=["out_of_range", "empty"],
)
def test_dense_misaligned_list_drops_only_that_key_and_logs(
    values: list[torch.Tensor],
    idx: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []

    def record_error(message: str, *args: object) -> None:
        errors.append(message % args)

    monkeypatch.setattr(payload_build.logger, "error", record_error)

    payload = _build({"codes.audio": values, "sr": 24000}, idx=idx, start=idx, end=idx + 1)

    assert payload == {"sr": 24000}
    assert len(errors) == 1
    assert f"request r{idx}" in errors[0]
    assert f"index {idx}" in errors[0]
    assert f"length {len(values)}" in errors[0]
    assert "dropping key codes.audio" in errors[0]


def test_dense_non_list_values_keep_existing_routing() -> None:
    aligned = torch.arange(6).reshape(3, 2)
    unaligned = torch.arange(4).reshape(2, 2)
    tuple_value = (torch.tensor([31]), torch.tensor([32]))

    payload = _build(
        {
            "aligned": aligned,
            "unaligned": unaligned,
            "tuple": tuple_value,
            "none": None,
        },
        idx=1,
        start=1,
        end=2,
    )

    assert torch.equal(payload["aligned"], aligned[1:2])
    assert torch.equal(payload["unaligned"], unaligned)
    assert isinstance(payload["unaligned"], torch.Tensor)
    assert payload["unaligned"].data_ptr() != unaligned.data_ptr()
    assert payload["tuple"] is tuple_value
    assert payload["none"] is None
