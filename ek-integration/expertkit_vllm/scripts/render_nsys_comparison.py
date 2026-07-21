#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


COMPONENTS = (
    ("client_to_controller_ingress", "Client→Controller", "#4C78A8"),
    ("controller_before_worker", "Controller prepare", "#F2CF5B"),
    ("worker_rpc_span", "Worker RPC span", "#54A24B"),
    ("controller_after_worker", "Controller aggregate", "#B279A2"),
    ("controller_to_client_return", "Controller→Client", "#E45756"),
)


def render(payload: dict, output: Path) -> None:
    modes = payload["modes"]
    mode_order = ["sync", "pipeline-serial", "pipeline-parallel"]
    labels = {
        "sync": "sync / Worker×1",
        "pipeline-serial": "pipeline / Worker×1",
        "pipeline-parallel": "pipeline / Worker×4",
    }
    width = 1320
    chart_left = 245
    chart_width = 1015
    row_height = 108
    top = 116
    totals = {
        mode: sum(
            data["derived"].get(name, {}).get("mean_ms", 0.0)
            for name, _label, _color in COMPONENTS
        )
        for mode, data in modes.items()
    }
    maximum = max(totals.values(), default=1.0)
    height = top + len(mode_order) * row_height + 190
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:21px;font-weight:700}.label{font-size:14px;font-weight:600}.small{font-size:11px}.metric{font-size:12px}</style>',
        '<text x="34" y="34" class="title">Expert-Kit sync vs pipeline — Nsight Systems critical-path breakdown</text>',
        '<text x="34" y="58" class="small">mean per client Expert request; bars use cross-process request IDs and global nsys timestamps</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        x = chart_left + chart_width * tick / 5
        parts.append(
            f'<line x1="{x:.2f}" y1="82" x2="{x:.2f}" y2="{top + len(mode_order) * row_height - 24}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="76" text-anchor="middle" class="small">{value:.2f} ms</text>'
        )
    for index, mode in enumerate(mode_order):
        data = modes[mode]
        y = top + index * row_height
        parts.append(f'<text x="34" y="{y + 22}" class="label">{labels[mode]}</text>')
        x = chart_left
        for name, component_label, color in COMPONENTS:
            value = data["derived"].get(name, {}).get("mean_ms", 0.0)
            component_width = value / maximum * chart_width
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{max(component_width, 1.0):.2f}" height="32" fill="{color}" stroke="#172033" stroke-width="0.8"><title>{component_label}: {value:.3f} ms</title></rect>'
            )
            if component_width >= 72:
                parts.append(
                    f'<text x="{x + component_width / 2:.2f}" y="{y + 20}" text-anchor="middle" class="small" style="fill:#fff">{value:.2f} ms</text>'
                )
            x += component_width
        counts = data["counts"]
        queue = data["phases"].get("WORKER_QUEUE", {}).get("mean_ms", 0.0)
        expert = data["derived"].get("expert_active_union", {}).get("mean_ms", 0.0)
        d2h_value = data["gpu_copies"].get("D2H_ENQUEUE", {}).get("mean_ms")
        h2d_values = [
            data["gpu_copies"].get(phase, {}).get("mean_ms")
            for phase in ("H2D_ENQUEUE", "H2D_COPY")
        ]
        h2d_values = [value for value in h2d_values if value is not None]
        d2h_text = f"{d2h_value:.4f}" if d2h_value is not None else "n/a"
        h2d_text = f"{max(h2d_values):.4f}" if h2d_values else "n/a"
        benchmark = payload.get("benchmarks", {}).get("modes", {}).get(mode)
        benchmark_text = ""
        if benchmark:
            throughput = benchmark["summary"]["tokens_per_second_median"]
            change = benchmark["throughput_change_vs_sync_percent"]
            benchmark_text = (
                f" · throughput {throughput:.3f} tok/s ({change:+.2f}%)"
            )
        parts.append(
            f'<text x="{chart_left}" y="{y + 53}" class="metric">total {totals[mode]:.3f} ms · Worker RPC/client {counts["worker_rpcs_per_client_request"]:.2f} · Expert concurrency {counts["expert_max_concurrency"]} · queue/RPC {queue:.3f} ms</text>'
        )
        parts.append(
            f'<text x="{chart_left}" y="{y + 71}" class="metric">E active/request {expert:.3f} ms · GPU D2H/H2D {d2h_text}/{h2d_text} ms{benchmark_text}</text>'
        )
    legend_y = top + len(mode_order) * row_height + 8
    x = 110
    for _name, component_label, color in COMPONENTS:
        parts.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" fill="{color}"/>')
        parts.append(f'<text x="{x + 23}" y="{legend_y + 13}" class="small">{component_label}</text>')
        x += 205
    parts.append(
        f'<text x="34" y="{legend_y + 54}" class="small">Worker RPC span contains Controller→Worker gRPC, Worker queue, Expert compute, serialization and responses; nested phase tables in the JSON provide the detailed split.</text>'
    )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.output)


if __name__ == "__main__":
    main()
