"""Tests for DeepSeek nsys aggregation and SVG rendering."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load(name: str) -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(layer: int, ubatch: int) -> dict[str, object]:
    start = layer * 1_000_000 + ubatch * 100_000
    durations = {"A": 10_000, "A2E": 20_000, "E": 40_000, "E2A": 30_000}
    return {
        "call_id": f"{layer}-{ubatch}",
        "layer_id": layer,
        "microbatch_id": ubatch,
        "scope": [start, start + 120_000],
        "token_count": 16,
        "stages": {
            "A": [[start, start + 10_000]],
            "A2E": [[start + 10_000, start + 30_000]],
            "E": [[start + 30_000, start + 70_000]],
            "E2A": [[start + 70_000, start + 100_000]],
        },
        "durations_ns": durations,
        "e_thread_on_cpu_ns": 35_000,
        "subphases_ns": {},
        "memcpy_bytes": 0,
        "critical_path_ns": 120_000,
    }


def test_layer_rows_and_representative_window_cover_all_stages() -> None:
    analyzer = _load("analyze_deepseek_nsys")
    calls = [_call(layer, ubatch) for layer in range(1, 27) for ubatch in range(4)]

    rows = analyzer._layer_rows(calls)
    representative = analyzer._representative(calls)

    assert len(rows) == 26
    assert all(
        abs(sum(stage["share"] for stage in row["stages"].values()) - 1.0) < 1e-12
        for row in rows
    )
    assert len(representative["layers"]) == 2
    assert len(representative["calls"]) == 8


def test_decode_calls_remove_one_leading_prefill_per_layer() -> None:
    analyzer = _load("analyze_deepseek_nsys")
    calls = []
    for layer in range(1, 27):
        prefill = _call(layer, 0)
        prefill["call_id"] = f"prefill-{layer}"
        prefill["scope"] = [layer * 1_000_000 - 500_000, layer * 1_000_000 - 1]
        calls.append(prefill)
        calls.extend(_call(layer, ubatch) for ubatch in range(4))

    filtered = analyzer._decode_calls(calls)

    assert len(filtered) == 26 * 4
    assert not any(call["call_id"].startswith("prefill-") for call in filtered)


def test_decode_calls_use_token_count_to_remove_all_chunked_prefills() -> None:
    analyzer = _load("analyze_deepseek_nsys")
    calls = []
    for layer in range(2):
        for index in range(3):
            prefill = _call(layer, 0)
            prefill["call_id"] = f"prefill-{layer}-{index}"
            prefill["token_count"] = 1024
            calls.append(prefill)
        calls.extend(_call(layer, ubatch) for ubatch in range(4))

    filtered = analyzer._decode_calls(calls, decode_max_tokens=64)

    assert len(filtered) == 2 * 4
    assert all(call["token_count"] <= 64 for call in filtered)


def test_qwen_layer_zero_is_included_in_rows_and_representative() -> None:
    analyzer = _load("analyze_deepseek_nsys")
    calls = [_call(layer, ubatch) for layer in range(48) for ubatch in range(4)]

    rows = analyzer._layer_rows(calls, set(range(48)))
    representative = analyzer._representative(calls, set(range(48)))

    assert [row["layer_id"] for row in rows] == list(range(48))
    assert representative["layers"][0] >= 0
    assert representative["layers"][1] == representative["layers"][0] + 1


def test_representative_uses_same_wave_when_ubatch_zero_has_tail_calls() -> None:
    analyzer = _load("analyze_deepseek_nsys")
    calls = []
    for wave in range(3):
        for layer in range(2):
            for ubatch in range(4):
                call = _call(layer, ubatch)
                start = wave * 10_000_000 + layer * 1_000_000 + ubatch * 100_000
                call["scope"] = [start, start + 120_000]
                call["call_id"] = f"wave-{wave}-{layer}-{ubatch}"
                calls.append(call)
    for layer in range(2):
        tail = _call(layer, 0)
        tail["scope"] = [100_000_000 + layer * 1_000_000, 100_120_000]
        tail["call_id"] = f"tail-{layer}"
        calls.append(tail)

    representative = analyzer._representative(calls)
    starts = [call["scope"][0] for call in representative["calls"]]

    assert max(starts) - min(starts) < 2_000_000
    assert not any(call["call_id"].startswith("tail-") for call in representative["calls"])


def test_renderers_write_heatmap_and_bordered_timeline(tmp_path: Path) -> None:
    analyzer = _load("analyze_deepseek_nsys")
    renderer = _load("render_deepseek_pipeline")
    calls = [_call(layer, ubatch) for layer in range(1, 27) for ubatch in range(4)]
    payload = {
        "pipeline": {
            "layers": analyzer._layer_rows(calls),
            "representative": analyzer._representative(calls),
            "communication_hiding": analyzer._hide_metrics(calls),
        }
    }
    heatmap = tmp_path / "heatmap.svg"
    timeline = tmp_path / "timeline.svg"

    renderer.render_heatmap(payload, heatmap)
    renderer.render_timeline(payload, timeline)

    assert "Layer 26" in heatmap.read_text(encoding="utf-8")
    timeline_text = timeline.read_text(encoding="utf-8")
    assert 'stroke-width="1.4"' in timeline_text
    assert "uBatch 3" in timeline_text
