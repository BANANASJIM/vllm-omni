# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
from collections.abc import Iterator
from typing import Literal

import pytest

from vllm_omni.engine.messages import (
    AbortResultMessage,
    CollectiveRPCResultMessage,
    EngineQueueMessage,
    ErrorMessage,
)
from vllm_omni.engine.rpc_result_router import RpcResultRouter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class GenericCorrelatedResultMessage(EngineQueueMessage, kw_only=True):  # type: ignore[call-arg]
    type: Literal["generic_correlated_result"] = "generic_correlated_result"
    namespace: str
    correlation_id: str

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return (self.namespace, self.correlation_id)


def _generic_result(correlation_id: str) -> GenericCorrelatedResultMessage:
    return GenericCorrelatedResultMessage(
        namespace="plugin",
        correlation_id=correlation_id,
    )


@pytest.fixture
def router_and_source() -> Iterator[tuple[RpcResultRouter, queue.Queue[EngineQueueMessage]]]:
    source: queue.Queue[EngineQueueMessage] = queue.Queue()
    router = RpcResultRouter(source)
    try:
        yield router, source
    finally:
        router.close()


def test_rpc_result_router_routes_out_of_order_results_by_correlation_id(router_and_source):
    router, source = router_and_source
    first = router.register(("plugin", "first"))
    second = router.register(("plugin", "second"))

    source.put(_generic_result("second"))
    source.put(_generic_result("first"))

    assert first.get(timeout=1).correlation_id == "first"
    assert second.get(timeout=1).correlation_id == "second"


def test_rpc_result_router_drops_only_the_late_result_after_unregister(router_and_source):
    router, source = router_and_source
    expired = router.register(("plugin", "expired"))
    active = router.register(("collective", "active"))
    router.unregister(("plugin", "expired"), expired)

    source.put(_generic_result("expired"))
    source.put(
        CollectiveRPCResultMessage(
            rpc_id="active",
            method="health",
            stage_ids=[0],
            results=["ok"],
        )
    )

    assert active.get(timeout=1).rpc_id == "active"
    assert expired.empty()


def test_rpc_result_router_broadcasts_fatal_errors_to_pending_waiters(router_and_source):
    router, source = router_and_source
    plugin = router.register(("plugin", "one"))
    collective = router.register(("collective", "two"))

    source.put(ErrorMessage(error="orchestrator failed", fatal=True))

    assert plugin.get(timeout=1).error == "orchestrator failed"
    assert collective.get(timeout=1).error == "orchestrator failed"
    with pytest.raises(RuntimeError, match="orchestrator failed"):
        router.register(("plugin", "after-failure"))


def test_rpc_result_router_does_not_broadcast_uncorrelated_nonfatal_errors(router_and_source):
    router, source = router_and_source
    waiter = router.register(("plugin", "active"))

    source.put(ErrorMessage(error="request failed", fatal=False, request_id="other"))
    source.put(_generic_result("active"))

    assert waiter.get(timeout=1).correlation_id == "active"


def test_rpc_result_router_routes_abort_results_by_correlation_id(router_and_source):
    router, source = router_and_source
    waiter = router.register(("abort", "abort-42"))

    source.put(AbortResultMessage(rpc_id="abort-42", success=True))

    result = waiter.get(timeout=1)
    assert isinstance(result, AbortResultMessage)
    assert result.rpc_id == "abort-42"


def test_rpc_result_router_close_unblocks_waiters_and_stops_consumer(router_and_source):
    router, _ = router_and_source
    waiter = router.register(("plugin", "pending"))

    router.close()

    result = waiter.get(timeout=1)
    assert isinstance(result, ErrorMessage)
    assert result.fatal is True
    assert result.error == "RPC result router closed"
    assert not router._thread.is_alive()
    with pytest.raises(RuntimeError, match="router is closed"):
        router.register(("plugin", "after-close"))
