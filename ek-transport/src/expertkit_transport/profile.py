"""Opt-in NVTX correlation for cross-process Expert Kit profiling."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

import torch

_PROFILE_ENV = "EK_NSYS_PROFILE"
_METADATA_CALL = "x-expertkit-profile-call"
_METADATA_UBATCH = "x-expertkit-profile-ubatch"
_METADATA_LAYER = "x-expertkit-profile-layer"
_CURRENT: ContextVar[ProfileContext | None] = ContextVar(
    "expertkit_profile_context",
    default=None,
)


@dataclass(frozen=True, slots=True)
class ProfileContext:
    """Identify one logical routed-layer call across Frontend and Worker."""

    call_id: str
    microbatch_id: int
    layer_id: int
    token_count: int | None = None

    def __post_init__(self) -> None:
        if not self.call_id or ":" in self.call_id:
            raise ValueError("profile call_id must be nonempty and contain no colon")
        for name in ("microbatch_id", "layer_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"profile {name} must be a nonnegative integer")
        if self.token_count is not None and (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise ValueError("profile token_count must be a positive integer")


def enabled() -> bool:
    """Return whether detailed nsys instrumentation is explicitly enabled."""

    return os.getenv(_PROFILE_ENV, "0") == "1"


def current_context() -> ProfileContext | None:
    """Return the call bound to the current asynchronous execution context."""

    return _CURRENT.get()


def bind_context(context: ProfileContext | None) -> Token[ProfileContext | None] | None:
    """Bind a call to this context when profiling is enabled."""

    if not enabled() or context is None:
        return None
    return _CURRENT.set(context)


def reset_context(token: Token[ProfileContext | None] | None) -> None:
    """Restore the context that preceded :func:`bind_context`."""

    if token is not None:
        _CURRENT.reset(token)


def label(phase: str, context: ProfileContext | None = None) -> str:
    """Build the stable label consumed by the nsys analyzer."""

    resolved = context or current_context()
    if resolved is None:
        return f"EKP:{phase}:call=none:u=0:l=0"
    result = f"EKP:{phase}:call={resolved.call_id}:u={resolved.microbatch_id}:l={resolved.layer_id}"
    if resolved.token_count is not None:
        result += f":n={resolved.token_count}"
    return result


@contextmanager
def nvtx_range(
    phase: str,
    context: ProfileContext | None = None,
) -> Iterator[None]:
    """Emit one Host NVTX range without affecting the disabled hot path."""

    if not enabled():
        yield
        return
    torch.cuda.nvtx.range_push(label(phase, context))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_range_start(
    phase: str,
    context: ProfileContext | None = None,
) -> int | None:
    """Start a non-nested range that may finish in a later callback."""

    if not enabled():
        return None
    return torch.cuda.nvtx.range_start(label(phase, context))


def nvtx_range_end(range_id: int | None) -> None:
    """Finish a range returned by :func:`nvtx_range_start`."""

    if range_id is not None:
        torch.cuda.nvtx.range_end(range_id)


def grpc_metadata(
    context: ProfileContext | None = None,
) -> tuple[tuple[str, str], ...] | None:
    """Encode correlation as profiling-only gRPC metadata."""

    if not enabled():
        return None
    resolved = context or current_context()
    if resolved is None:
        return None
    return (
        (_METADATA_CALL, resolved.call_id),
        (_METADATA_UBATCH, str(resolved.microbatch_id)),
        (_METADATA_LAYER, str(resolved.layer_id)),
    )


def profile_context_from_metadata(
    metadata: Sequence[tuple[str, str | bytes]],
) -> ProfileContext | None:
    """Decode correlation metadata, ignoring incomplete non-profile calls."""

    if not enabled():
        return None
    values: dict[str, str] = {}
    for key, value in metadata:
        if key not in {_METADATA_CALL, _METADATA_UBATCH, _METADATA_LAYER}:
            continue
        values[key] = value.decode() if isinstance(value, bytes) else value
    if not values:
        return None
    required = {_METADATA_CALL, _METADATA_UBATCH, _METADATA_LAYER}
    if values.keys() != required:
        raise ValueError("incomplete Expert Kit profiling metadata")
    try:
        return ProfileContext(
            call_id=values[_METADATA_CALL],
            microbatch_id=int(values[_METADATA_UBATCH]),
            layer_id=int(values[_METADATA_LAYER]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid Expert Kit profiling metadata") from error
