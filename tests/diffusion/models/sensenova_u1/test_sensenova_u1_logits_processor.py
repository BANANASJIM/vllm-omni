# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test: SenseNovaU1ForCausalLM instantiation under the diffusion
config shim must not crash on missing ``head_dtype`` and the resulting
LogitsProcessor must have ``head_dtype is None`` (= use model dtype)."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1ForCausalLM,
)
from vllm_omni.diffusion.vllm_config import _DiffusionVllmModelConfig
from vllm_omni.transformers_utils.configs.sensenova_u1 import (
    SenseNovaU1Config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@contextmanager
def _single_rank_distributed(tmp_path: Path) -> Iterator[None]:
    """Initialize the single-rank model-parallel group required by the model."""
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=f"file://{tmp_path / 'torch_dist_init'}",
        )
        initialize_model_parallel()
        yield
    finally:
        cleanup_dist_env_and_memory()


def _build_tiny_sensenova_u1(quant_config=None, prefix="model"):
    """Build a minimal SenseNovaU1ForCausalLM for unit tests."""
    config = SenseNovaU1Config(
        llm_config={
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "max_position_embeddings_hw": 128,
        },
    )
    return SenseNovaU1ForCausalLM(config.llm_config, quant_config=quant_config, prefix=prefix)


def test_logits_processor_head_dtype_under_diffusion_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role",
        lambda **_: (SDPABackend, None),
    )
    with _single_rank_distributed(tmp_path):
        # Create a vLLM config with a wrapped diffusion config
        fake_diff_config = _DiffusionVllmModelConfig(
            model="sensenova-u1-test",
            dtype=torch.bfloat16,
            max_model_len=8192,
            original_max_model_len=8192,
        )
        vllm_config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
        )
        vllm_config.modelconfig = fake_diff_config  # type: ignore[assignment]

        # Build a tiny version of the Causal LM component
        with set_current_vllm_config(vllm_config):
            model = _build_tiny_sensenova_u1()

        # Ensure that we have a set head_dtype attribute. Currently, the head_dtype
        # is set to None
        head_dtype = model.logits_processor.head_dtype
        assert head_dtype is None


def test_distributed_cleanup_runs_when_model_parallel_setup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = sys.modules[__name__]
    cleanup_calls: list[None] = []

    def fail_model_parallel_setup() -> None:
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(module, "init_distributed_environment", lambda **_: None)
    monkeypatch.setattr(module, "initialize_model_parallel", fail_model_parallel_setup)
    monkeypatch.setattr(module, "cleanup_dist_env_and_memory", lambda: cleanup_calls.append(None))

    with pytest.raises(RuntimeError, match="injected setup failure"), _single_rank_distributed(tmp_path):
        pass

    assert cleanup_calls == [None]


def test_distributed_cleanup_runs_when_assertion_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = sys.modules[__name__]
    cleanup_calls: list[None] = []
    monkeypatch.setattr(module, "init_distributed_environment", lambda **_: None)
    monkeypatch.setattr(module, "initialize_model_parallel", lambda: None)
    monkeypatch.setattr(module, "cleanup_dist_env_and_memory", lambda: cleanup_calls.append(None))

    with pytest.raises(AssertionError), _single_rank_distributed(tmp_path):
        assert False

    assert cleanup_calls == [None]
