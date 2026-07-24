#!/usr/bin/env python3
"""Render per-layer stage shares and a representative four-uBatch timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STAGES = ("A", "A2E", "E", "E2A")
STAGE_COLORS = {
    "A": "#4C78A8",
    "A2E": "#F59E0B",
    "E": "#59A14F",
    "E2A": "#E15759",
}
UBATCH_COLORS = ("#4C78A8", "#F28E5B", "#59A14F", "#B279A2")


def _write(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_heatmap(payload: dict[str, Any], output: Path) -> None:
    rows = payload["pipeline"]["layers"]
    model_name = payload.get("definitions", {}).get("model", "MoE model")
    width = 1320
    left = 150
    top = 92
    cell_width = 260
    row_height = 30
    height = top + len(rows) * row_height + 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:21px;font-weight:700}.head{font-size:14px;font-weight:700}.cell{font-size:11px}.note{font-size:11px;fill:#4b5563}</style>",
        f'<text x="28" y="32" class="title">{model_name} pipeline: per-layer four-stage work share</text>',
        '<text x="28" y="55" class="note">batch 64 · four uBatches · 31 profiled decode steps · cell = mean latency / share of A+A2E+E+E2A work</text>',
    ]
    for column, stage in enumerate(STAGES):
        x = left + column * cell_width
        parts.append(
            f'<rect x="{x}" y="67" width="{cell_width - 8}" height="24" rx="3" fill="{STAGE_COLORS[stage]}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{x + (cell_width - 8) / 2}" y="84" text-anchor="middle" class="head" style="fill:#ffffff">{stage}</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * row_height
        parts.append(
            f'<text x="105" y="{y + 20}" class="head">Layer {row["layer_id"]}</text>'
        )
        for column, stage in enumerate(STAGES):
            data = row["stages"][stage]
            share = data["share"]
            x = left + column * cell_width
            opacity = 0.16 + 0.78 * min(share / 0.75, 1.0)
            parts.append(
                f'<rect x="{x}" y="{y + 2}" width="{cell_width - 8}" height="{row_height - 4}" rx="2" fill="{STAGE_COLORS[stage]}" opacity="{opacity:.3f}" stroke="#ffffff"/>'
            )
            text_color = "#ffffff" if opacity > 0.56 else "#172033"
            parts.append(
                f'<text x="{x + (cell_width - 8) / 2}" y="{y + 20}" text-anchor="middle" class="cell" style="fill:{text_color}">{data["mean_ms"]:.3f} ms · {share * 100:.1f}%</text>'
            )
    note_y = top + len(rows) * row_height + 34
    parts.append(
        f'<text x="{left}" y="{note_y}" class="note">Percentages describe accumulated stage work; because micro-batches overlap, they are not end-to-end wall-time percentages.</text>'
    )
    parts.append("</svg>")
    _write(output, parts)


def render_timeline(payload: dict[str, Any], output: Path) -> None:
    representative = payload["pipeline"]["representative"]
    model_name = payload.get("definitions", {}).get("model", "MoE model")
    calls = representative["calls"]
    layers = representative["layers"]
    origin = min(
        interval[0]
        for call in calls
        for stage in STAGES
        for interval in call["stages"][stage]
    )
    finish = max(
        interval[1]
        for call in calls
        for stage in STAGES
        for interval in call["stages"][stage]
    )
    span = max(finish - origin, 1)
    width = 1480
    chart_left = 145
    chart_width = 1285
    top = 112
    row_height = 86
    chart_bottom = top + row_height * len(STAGES)
    height = chart_bottom + 170

    def x(timestamp: int) -> float:
        return chart_left + (timestamp - origin) / span * chart_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#b23b2b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:21px;font-weight:700}.row{font-size:15px;font-weight:700}.small{font-size:10px}.note{font-size:12px;fill:#4b5563}</style>",
        f'<text x="28" y="32" class="title">{model_name} A → A2E → E → E2A pipeline overlap</text>',
        f'<text x="28" y="56" class="note">representative layers {layers[0]}–{layers[1]}; four micro-batches; all blocks use measured nsys timestamps and explicit borders</text>',
    ]
    for tick in range(7):
        timestamp = origin + span * tick / 6
        xpos = x(timestamp)
        parts.append(
            f'<line x1="{xpos:.2f}" y1="80" x2="{xpos:.2f}" y2="{chart_bottom}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{xpos:.2f}" y="74" text-anchor="middle" class="small">{(timestamp - origin) / 1e6:.1f} ms</text>'
        )
    ordered = sorted(calls, key=lambda call: (call["layer_id"], call["microbatch_id"]))
    envelopes: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    for row_index, stage in enumerate(STAGES):
        y = top + row_index * row_height
        parts.append(f'<text x="28" y="{y + 34}" class="row">{stage}</text>')
        for call in ordered:
            intervals = call["stages"][stage]
            start = min(interval[0] for interval in intervals)
            end = max(interval[1] for interval in intervals)
            xpos = x(start)
            duration_ns = call["durations_ns"][stage]
            block_width = max(
                (duration_ns / span * chart_width)
                if stage == "A"
                else x(end) - xpos,
                2.0,
            )
            ubatch = call["microbatch_id"]
            color = UBATCH_COLORS[ubatch % len(UBATCH_COLORS)]
            duration_ms = duration_ns / 1e6
            envelopes[(stage, call["layer_id"], ubatch)] = (xpos, block_width, y)
            parts.append(
                f'<rect x="{xpos:.2f}" y="{y + 7}" width="{block_width:.2f}" height="38" rx="3" fill="{color}" opacity="0.79" stroke="#172033" stroke-width="1.4" vector-effect="non-scaling-stroke"><title>uBatch {ubatch}, layer {call["layer_id"]}, {stage}: {duration_ms:.3f} ms</title></rect>'
            )
            if block_width >= 55:
                center = xpos + block_width / 2
                parts.append(
                    f'<text x="{center:.2f}" y="{y + 22}" text-anchor="middle" class="small" style="fill:#ffffff;paint-order:stroke;stroke:#172033;stroke-width:1.4px"><tspan x="{center:.2f}">u{ubatch} · L{call["layer_id"]}</tspan><tspan x="{center:.2f}" dy="12">{duration_ms:.2f} ms</tspan></text>'
                )
    for ubatch in representative["microbatches"]:
        left = envelopes.get(("E2A", layers[0], ubatch))
        right = envelopes.get(("A", layers[1], ubatch))
        if left is None or right is None:
            continue
        start_x = left[0] + left[1]
        end_x = right[0]
        parts.append(
            f'<path d="M {start_x:.2f} {top + 3 * row_height + 45:.2f} C {start_x + 12:.2f} {top + 3 * row_height + 62:.2f}, {end_x - 12:.2f} {top - 10:.2f}, {end_x:.2f} {top + 7:.2f}" fill="none" stroke="#b23b2b" stroke-width="1.2" marker-end="url(#arrow)" opacity="0.75"/>'
        )
    legend_y = chart_bottom + 34
    for ubatch in representative["microbatches"]:
        xpos = chart_left + ubatch * 145
        color = UBATCH_COLORS[ubatch % len(UBATCH_COLORS)]
        parts.append(
            f'<rect x="{xpos}" y="{legend_y}" width="18" height="18" rx="2" fill="{color}" stroke="#172033"/>'
        )
        parts.append(
            f'<text x="{xpos + 26}" y="{legend_y + 14}" class="note">uBatch {ubatch}</text>'
        )
    hiding = payload["pipeline"]["communication_hiding"]
    parts.append(
        f'<text x="{chart_left}" y="{legend_y + 54}" class="note">A2E hidden {hiding["A2E"]["hide_ratio"] * 100:.1f}% · E2A hidden {hiding["E2A"]["hide_ratio"] * 100:.1f}% · total exposed communication {hiding["total"]["exposed_ms"]:.1f} ms</text>'
    )
    parts.append(
        f'<text x="{chart_left}" y="{legend_y + 78}" class="note">A uses active CUDA-kernel duration at its first measured kernel timestamp; A2E/E/E2A use their measured wall intervals.</text>'
    )
    parts.append("</svg>")
    _write(output, parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--heatmap", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    render_heatmap(payload, args.heatmap)
    render_timeline(payload, args.timeline)
    print(
        json.dumps(
            {"heatmap": str(args.heatmap), "timeline": str(args.timeline)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
