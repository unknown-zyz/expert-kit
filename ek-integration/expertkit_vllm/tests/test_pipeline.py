"""Core completion-driven uBatch scheduling tests for the versioned patch."""

import concurrent.futures
import threading
import time

import pytest
from vllm.v1.worker.ubatching import (
    EXPERTKIT_PIPELINE_PATCH_VERSION,
    UBatchAborted,
    UBatchCoordinator,
)


def test_installed_patch_has_four_ubatch_workspace_fix() -> None:
    assert EXPERTKIT_PIPELINE_PATCH_VERSION >= 2


def test_scheduler_resumes_ubatches_in_future_completion_order() -> None:
    coordinator = UBatchCoordinator(4)
    futures = [concurrent.futures.Future() for _ in range(4)]
    barrier = threading.Barrier(5)
    observed: list[tuple[int, str]] = []

    def run(ubatch_id: int) -> None:
        barrier.wait()
        coordinator.enter(ubatch_id)
        observed.append((ubatch_id, "A"))
        coordinator.wait_for_future(ubatch_id, futures[ubatch_id])
        observed.append((ubatch_id, "E2A"))
        coordinator.finish(ubatch_id)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    time.sleep(0.02)
    for ubatch_id in (2, 0, 3, 1):
        futures[ubatch_id].set_result(ubatch_id)
    for thread in threads:
        thread.join(timeout=2)

    assert observed[:4] == [(0, "A"), (1, "A"), (2, "A"), (3, "A")]
    assert [item[0] for item in observed[4:]] == [2, 0, 3, 1]
    assert not any(thread.is_alive() for thread in threads)


def test_scheduler_broadcasts_the_first_failure_to_waiters() -> None:
    coordinator = UBatchCoordinator(2)
    pending = concurrent.futures.Future()
    failure = RuntimeError("remote expert failed")
    coordinator.enter(0)
    coordinator.abort(failure)

    with pytest.raises(UBatchAborted) as raised:
        coordinator.wait_for_future(0, pending)

    assert raised.value.__cause__ is failure
