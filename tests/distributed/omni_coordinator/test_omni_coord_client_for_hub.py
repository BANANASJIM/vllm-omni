# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import zmq

from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForHub,
    ReplicaList,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@contextmanager
def _hub_client_resources() -> Iterator[tuple[zmq.Socket, OmniCoordClientForHub]]:
    ctx = zmq.Context()
    pub = None
    client = None
    try:
        pub = ctx.socket(zmq.PUB)
        pub.bind("tcp://127.0.0.1:*")
        endpoint = pub.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
        client = OmniCoordClientForHub(endpoint)
        yield pub, client
    finally:
        try:
            if client is not None and not client._closed:
                client.close()
        finally:
            try:
                if pub is not None:
                    pub.close(0)
            finally:
                ctx.term()


@pytest.fixture
def hub_client_resources() -> Iterator[tuple[zmq.Socket, OmniCoordClientForHub]]:
    with _hub_client_resources() as resources:
        yield resources


def _publish_until(
    pub,
    payload: object,
    cond: Callable[[], bool],
    timeout: float = 2.0,
    interval: float = 0.01,
) -> bool:
    """Retry a PUB message until the subscriber observes it or time runs out."""
    message = json.dumps(payload).encode("utf-8")
    deadline = time.monotonic() + timeout
    while True:
        pub.send(message)
        if cond():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return cond()
        time.sleep(min(interval, remaining))


def test_hub_client_caches_replica_list_from_pub(hub_client_resources):
    """Verify OmniCoordClientForHub receives replica list updates from OmniCoordinator and caches for get_replica_list()."""
    pub, client = hub_client_resources

    now = time.time()
    replicas_payload = [
        {
            "input_addr": "tcp://stage:10001",
            "output_addr": "tcp://stage:10001-out",
            "stage_id": 0,
            "status": "up",
            "queue_length": 0,
            "last_heartbeat": now,
            "registered_at": now,
        },
        {
            "input_addr": "tcp://stage:10002",
            "output_addr": "tcp://stage:10002-out",
            "stage_id": 0,
            "status": "up",
            "queue_length": 1,
            "last_heartbeat": now,
            "registered_at": now,
        },
        {
            "input_addr": "tcp://stage:10003",
            "output_addr": "tcp://stage:10003-out",
            "stage_id": 1,
            "status": "error",
            "queue_length": 5,
            "last_heartbeat": now,
            "registered_at": now,
        },
    ]

    payload = {"replicas": replicas_payload, "timestamp": now}
    assert _publish_until(pub, payload, lambda: len(client.get_replica_list().replicas) == 3)

    rep_list = client.get_replica_list()
    assert isinstance(rep_list, ReplicaList)
    assert len(rep_list.replicas) == 3

    for src, rep in zip(replicas_payload, rep_list.replicas, strict=True):
        assert rep.input_addr == src["input_addr"]
        assert rep.output_addr == src["output_addr"]
        assert rep.stage_id == src["stage_id"]
        assert rep.status.value == src["status"]

    stage0 = client.get_replicas_for_stage(0)
    stage1 = client.get_replicas_for_stage(1)

    assert all(rep.stage_id == 0 for rep in stage0.replicas)
    assert all(rep.stage_id == 1 for rep in stage1.replicas)

    # Send an updated list with fewer replicas and verify cache refresh.
    updated_payload = {
        "replicas": replicas_payload[:2],
        "timestamp": now + 1.0,
    }
    assert _publish_until(pub, updated_payload, lambda: len(client.get_replica_list().replicas) == 2)
    updated_list = client.get_replica_list()
    assert len(updated_list.replicas) == 2


def test_hub_client_close_closes_sub_socket(hub_client_resources):
    """Verify OmniCoordClientForHub.close() marks client as closed; second close raises."""
    _, client = hub_client_resources
    client.close()

    with pytest.raises(RuntimeError, match="already closed"):
        client.close()


def test_hub_resources_close_when_assertion_fails():
    pub: zmq.Socket
    client: OmniCoordClientForHub
    singleton_ctx = zmq.Context.instance()

    with pytest.raises(AssertionError), _hub_client_resources() as (pub, client):
        assert False

    assert pub.closed
    assert client._ctx.closed
    assert not client._thread.is_alive()
    assert zmq.Context.instance() is singleton_ctx
    assert not singleton_ctx.closed

    sentinel = singleton_ctx.socket(zmq.PAIR)
    try:
        assert sentinel.getsockopt(zmq.TYPE) == zmq.PAIR
    finally:
        sentinel.close(0)
