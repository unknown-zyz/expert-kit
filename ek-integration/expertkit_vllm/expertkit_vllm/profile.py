import os
from contextlib import contextmanager

import nvtx


def enabled() -> bool:
    return os.getenv("EK_NSYS_PROFILE", "0") == "1"


def label(
    phase: str,
    request_id: str = "none",
    microbatch_id: int = 0,
    layer_id: int = 0,
) -> str:
    return (
        f"EKC:{phase}:req={request_id}:u={microbatch_id}:l={layer_id}"
    )


@contextmanager
def range(
    phase: str,
    request_id: str = "none",
    microbatch_id: int = 0,
    layer_id: int = 0,
):
    if not enabled():
        yield
        return
    nvtx.push_range(label(phase, request_id, microbatch_id, layer_id))
    try:
        yield
    finally:
        nvtx.pop_range()
