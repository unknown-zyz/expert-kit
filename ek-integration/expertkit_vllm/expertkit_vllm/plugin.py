"""vLLM general plugin registration for remote Routed-MoE execution."""

from __future__ import annotations

import logging
import os
from importlib.metadata import version

logger = logging.getLogger(__name__)


def register() -> None:
    """Replace the vLLM 0.25.1 FusedMoE factory when explicitly enabled."""

    if os.getenv("EK_ENABLE") != "1":
        return

    if os.getenv("EK_PIPELINE_ENABLE", "0") == "1":
        installed_vllm = version("vllm")
        if installed_vllm != "0.25.1":
            raise RuntimeError(
                "Expert Kit pipeline requires the versioned vLLM 0.25.1 patch; "
                f"found {installed_vllm}"
            )
        from vllm.v1.worker import ubatching

        if (
            not hasattr(ubatching, "dbo_wait_for_future")
            or getattr(ubatching, "EXPERTKIT_PIPELINE_PATCH_VERSION", 0) < 2
        ):
            raise RuntimeError(
                "Expert Kit pipeline v2 patch is not applied; run "
                "scripts/apply_vllm_pipeline_patch.py --apply"
            )

    import vllm.model_executor.layers.fused_moe as fused_moe_package
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    from expertkit_vllm.experts.remote_moe import remote_fused_moe

    fused_moe_layer.FusedMoE = remote_fused_moe
    fused_moe_package.FusedMoE = remote_fused_moe
    logger.info("enabled Expert Kit remote MoE for vLLM 0.25.1")
