"""Profile-only vLLM model instrumentation."""

from __future__ import annotations

from functools import wraps
from typing import Any

from expertkit_transport.profile import ProfileContext, enabled, nvtx_range


def _instrument_decoder_class(decoder_class: type[Any], current_ubatch: Any) -> None:
    """Wrap one decoder class once, retaining its exact public signature metadata."""

    if hasattr(decoder_class, "_expertkit_profile_original_forward"):
        return
    original = decoder_class.forward

    @wraps(original)
    def profiled(self: Any, *args: Any, **kwargs: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 1:
            hidden_states = args[1]
        token_count = int(hidden_states.shape[0])
        layer_id = getattr(self, "layer_idx", None)
        if layer_id is None:
            experts = getattr(getattr(self, "mlp", None), "experts", None)
            layer_id = getattr(experts, "layer_id", None)
        if layer_id is None:
            raise RuntimeError(
                f"cannot resolve Expert Kit layer id for {type(self).__name__}"
            )
        context = ProfileContext(
            call_id="scope",
            microbatch_id=current_ubatch(),
            layer_id=layer_id,
            token_count=token_count,
        )
        with nvtx_range("A_SCOPE", context):
            return original(self, *args, **kwargs)

    decoder_class._expertkit_profile_original_forward = original
    decoder_class.forward = profiled


def install_vllm_profile() -> None:
    """Install supported MoE layer scopes only for an explicit nsys run."""

    if not enabled():
        return
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    _instrument_decoder_class(DeepseekV2DecoderLayer, dbo_current_ubatch_id)
    _instrument_decoder_class(Qwen3MoeDecoderLayer, dbo_current_ubatch_id)
