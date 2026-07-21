#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


STAGES = ("A", "A2E", "E", "E2A")
STAGE_LABELS = {
    "A": "Attention",
    "A2E": "A2E",
    "E": "Expert",
    "E2A": "E2A",
}
LAYER_COLORS = ("#4C78A8", "#F28E5B")


def render(payload: dict, output: Path) -> dict:
    calls = payload["calls"]
    layers = payload["selected_layers"]
    if len(layers) != 2:
        raise ValueError("the nsys diagram requires exactly two adjacent layers")
    origin = min(
        call.get("A_attributed", [call["A_envelope"]])[0][0]
        for call in calls
    )
    finish = max(call["E2A"][0][1] for call in calls)
    span = max(finish - origin, 1)
    chart_left = 150
    chart_width = 1120
    chart_top = 116
    row_height = 82
    chart_bottom = chart_top + len(STAGES) * row_height
    svg_height = chart_bottom + 210

    def x(timestamp: int) -> float:
        return chart_left + (timestamp - origin) / span * chart_width

    def layer_color(layer_id: int) -> str:
        return LAYER_COLORS[layers.index(layer_id)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="{svg_height}" viewBox="0 0 1320 {svg_height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c2412d"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:21px;font-weight:700}.label{font-size:15px;font-weight:600}.small{font-size:11px}.metric{font-size:13px}</style>',
        '<text x="34" y="32" class="title">Expert-Kit A → A2E → E → E2A — Nsight Systems hardware timeline</text>',
        f'<text x="34" y="56" class="small">layers {layers[0]}–{layers[1]}; four micro-batches; A includes attributed local Other Compute; bars are time-scaled</text>',
    ]
    for tick in range(7):
        timestamp = origin + span * tick / 6
        xpos = x(timestamp)
        parts.append(
            f'<line x1="{xpos:.2f}" y1="82" x2="{xpos:.2f}" y2="{chart_bottom}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{xpos:.2f}" y="76" text-anchor="middle" class="small">{(timestamp - origin) / 1e6:.2f} ms</text>'
        )

    ordered = sorted(
        calls, key=lambda call: (call["layer_id"], call["microbatch_id"])
    )
    for row, stage in enumerate(STAGES):
        y = chart_top + row * row_height
        parts.append(
            f'<text x="26" y="{y + 35}" class="label">{STAGE_LABELS[stage]}</text>'
        )
        parts.append(
            f'<line x1="{chart_left}" y1="{y + 42}" x2="{chart_left + chart_width}" y2="{y + 42}" stroke="#cbd5e1"/>'
        )
        for call in ordered:
            color = layer_color(call["layer_id"])
            intervals = (
                call.get("A_attributed", [call["A_envelope"]])
                if stage == "A"
                else call[stage]
            )
            duration_ms = sum(end - start for start, end in intervals) / 1e6
            if stage == "A":
                envelope = intervals[0]
            elif stage == "E":
                envelope = call["E_envelope"]
            else:
                envelope = intervals[0]
            envelope_x = x(envelope[0])
            envelope_width = max(x(envelope[1]) - envelope_x, 1.5)
            if stage in ("A", "E"):
                duration_kind = (
                    "attributed local stage span"
                    if stage == "A"
                    else "hardware-active"
                )
                parts.append(
                    f'<rect x="{envelope_x:.2f}" y="{y + 13}" width="{envelope_width:.2f}" height="31" rx="3" fill="{color}" opacity="0.78" stroke="#172033" stroke-width="1.2" vector-effect="non-scaling-stroke"><title>uBatch {call["microbatch_id"]}, layer {call["layer_id"]}, {STAGE_LABELS[stage]}: {duration_ms:.3f} ms {duration_kind}; {(envelope[1] - envelope[0]) / 1e6:.3f} ms wall envelope</title></rect>'
                )
            else:
                for start, end in intervals:
                    xpos = x(start)
                    width = max(x(end) - xpos, 1.2)
                    parts.append(
                        f'<rect x="{xpos:.2f}" y="{y + 13}" width="{width:.2f}" height="31" rx="2" fill="{color}" opacity="0.78" stroke="#172033" stroke-width="1.0" vector-effect="non-scaling-stroke"><title>uBatch {call["microbatch_id"]}, layer {call["layer_id"]}, {STAGE_LABELS[stage]}: {duration_ms:.3f} ms</title></rect>'
                    )
            if envelope_width >= 40:
                parts.append(
                    f'<text x="{envelope_x + envelope_width / 2:.2f}" y="{y + 27}" text-anchor="middle" class="small" style="fill:#ffffff;font-size:9px;paint-order:stroke;stroke:#172033;stroke-width:1.5px"><tspan x="{envelope_x + envelope_width / 2:.2f}">u{call["microbatch_id"]} · L{call["layer_id"]}</tspan><tspan x="{envelope_x + envelope_width / 2:.2f}" dy="11">{duration_ms:.2f} ms</tspan></text>'
                )

    first_layer = {
        call["microbatch_id"]: call
        for call in calls
        if call["layer_id"] == layers[0]
    }
    second_layer = {
        call["microbatch_id"]: call
        for call in calls
        if call["layer_id"] == layers[1]
    }
    for ubatch in range(4):
        left = first_layer[ubatch]
        right = second_layer[ubatch]
        start_x = x(left["E2A"][0][1])
        end_x = x(right.get("A_attributed", [right["A_envelope"]])[0][0])
        start_y = chart_top + 3 * row_height + 13
        end_y = chart_top + 44
        control_x = max(start_x + 12, (start_x + end_x) / 2)
        parts.append(
            f'<path d="M {start_x:.2f} {start_y:.2f} C {control_x:.2f} {start_y - 18:.2f}, {control_x:.2f} {end_y + 18:.2f}, {end_x:.2f} {end_y:.2f}" fill="none" stroke="#c2412d" stroke-width="1.3" marker-end="url(#arrow)" opacity="0.8"/>'
        )

    metrics = payload["metrics"]
    visible_metrics = payload.get("visible_window_metrics", metrics)
    legend_y = chart_bottom + 32
    for index, layer_id in enumerate(layers):
        xpos = chart_left + index * 190
        parts.append(
            f'<rect x="{xpos}" y="{legend_y}" width="18" height="18" rx="2" fill="{layer_color(layer_id)}"/>'
        )
        parts.append(
            f'<text x="{xpos + 27}" y="{legend_y + 14}" class="metric">layer {layer_id}</text>'
        )
    parts.append(
        f'<text x="{chart_left + 405}" y="{legend_y + 14}" class="small">A: attributed local stage · E: Worker CPU active · A2E/E2A: transport interval</text>'
    )
    metric_y = legend_y + 48
    parts.append(
        f'<text x="{chart_left}" y="{metric_y}" class="metric">A2E hidden {metrics["A2E"]["hide_ratio"] * 100:.1f}% · exposed {metrics["A2E"]["exposed_ns"] / 1e6:.3f} ms</text>'
    )
    parts.append(
        f'<text x="{chart_left + 365}" y="{metric_y}" class="metric">E2A hidden {metrics["E2A"]["hide_ratio"] * 100:.1f}% · exposed {metrics["E2A"]["exposed_ns"] / 1e6:.3f} ms</text>'
    )
    parts.append(
        f'<text x="{chart_left + 740}" y="{metric_y}" class="metric">full capture hidden {metrics["total"]["hide_ratio"] * 100:.1f}% ({metrics["total"]["classification"]})</text>'
    )
    parts.append(
        f'<text x="{chart_left}" y="{metric_y + 28}" class="small">A bar also includes weighted combine/norm/router/top-k; hiding metric uses measured A CUDA + E on-CPU unions ({visible_metrics["total"]["hide_ratio"] * 100:.1f}% in view)</text>'
    )
    means = {
        stage: sum(
            sum(
                end - start
                for start, end in (
                    call.get("A_attributed", [call["A_envelope"]])
                    if stage == "A"
                    else call[stage]
                )
            )
            for call in calls
        )
        / len(calls)
        / 1e6
        for stage in STAGES
    }
    parts.append(
        f'<text x="{chart_left}" y="{metric_y + 55}" class="metric">selected-window mean: A {means["A"]:.3f} ms · A2E {means["A2E"]:.3f} ms · E {means["E"]:.3f} ms · E2A {means["E2A"]:.3f} ms</text>'
    )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))
    return {
        "layers": layers,
        "microbatches": sorted({call["microbatch_id"] for call in calls}),
        "window_ns": span,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = render(json.loads(args.analysis.read_text()), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
