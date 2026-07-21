#!/usr/bin/env python3
"""Run the benchmark under nsys without py-cpuinfo's external `file` probe."""

from __future__ import annotations

import platform
import runpy
import struct


def _in_process_architecture(
    executable: str | None = None,
    bits: str = "",
    linkage: str = "",
) -> tuple[str, str]:
    """Return this interpreter's ABI without spawning a traced child process."""

    del executable
    return bits or f"{struct.calcsize('P') * 8}bit", linkage or "ELF"


platform.architecture = _in_process_architecture
runpy.run_module("expertkit_vllm.benchmark", run_name="__main__")
