"""Tests for profiling-only correlation and NVTX helpers."""

import torch

from expertkit_transport.profile import (
    ProfileContext,
    bind_context,
    current_context,
    grpc_metadata,
    nvtx_range,
    profile_context_from_metadata,
    reset_context,
)


def test_profile_context_round_trips_through_grpc_metadata(monkeypatch) -> None:
    monkeypatch.setenv("EK_NSYS_PROFILE", "1")
    expected = ProfileContext("123-4", 2, 17)

    metadata = grpc_metadata(expected)

    assert metadata is not None
    assert profile_context_from_metadata(metadata) == expected


def test_profile_context_binding_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("EK_NSYS_PROFILE", raising=False)
    expected = ProfileContext("123-4", 2, 17)

    token = bind_context(expected)

    assert token is None
    assert current_context() is None
    assert grpc_metadata(expected) is None


def test_nvtx_range_uses_bound_context(monkeypatch) -> None:
    monkeypatch.setenv("EK_NSYS_PROFILE", "1")
    observed: list[str] = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", observed.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: observed.append("pop"))
    expected = ProfileContext("123-4", 2, 17)
    token = bind_context(expected)
    try:
        with nvtx_range("E"):
            assert current_context() == expected
    finally:
        reset_context(token)

    assert observed == ["EKP:E:call=123-4:u=2:l=17", "pop"]
    assert current_context() is None
