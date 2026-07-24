"""Tests for profile-only vLLM decoder scopes."""

import torch

from expertkit_vllm.profile import _instrument_decoder_class


def test_decoder_profile_marks_layer_and_ubatch_once(monkeypatch) -> None:
    monkeypatch.setenv("EK_NSYS_PROFILE", "1")
    observed: list[str] = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", observed.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: observed.append("pop"))

    class Decoder:
        layer_idx = 9

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
        ) -> torch.Tensor:
            del positions, residual
            return hidden_states + 1

    _instrument_decoder_class(Decoder, lambda: 3)
    _instrument_decoder_class(Decoder, lambda: 7)

    result = Decoder().forward(torch.tensor([0]), torch.tensor([[4]]), None)

    assert result.tolist() == [[5]]
    assert observed == ["EKP:A_SCOPE:call=scope:u=3:l=9:n=1", "pop"]
