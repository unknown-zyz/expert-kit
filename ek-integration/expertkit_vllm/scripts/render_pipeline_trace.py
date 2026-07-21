#!/usr/bin/env python3
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


STAGES = ("A", "A2E", "E", "E2A")
COLORS = {
    "A": "#4C78A8",
    "A2E": "#F58518",
    "E": "#54A24B",
    "E2A": "#E45756",
}


def interval(event: dict) -> tuple[float, float]:
    start = float(event["ts"])
    return start, start + float(event["dur"])


def overlaps(left: dict, right: dict) -> bool:
    left_start, left_end = interval(left)
    right_start, right_end = interval(right)
    return max(left_start, right_start) < min(left_end, right_end)


def complete_calls(events: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], dict] = defaultdict(dict)
    for event in events:
        if event.get("cat") != "expertkit_pipeline" or event.get("name") not in STAGES:
            continue
        args = event.get("args", {})
        key = (
            str(args.get("request_id", "")),
            int(args.get("microbatch_id", -1)),
            int(args.get("layer_id", -1)),
        )
        grouped[key][event["name"]] = event
    calls = []
    for (request_id, ubatch_id, layer_id), stages in grouped.items():
        if all(stage in stages for stage in STAGES) and ubatch_id in range(4):
            calls.append(
                {
                    "request_id": request_id,
                    "microbatch_id": ubatch_id,
                    "layer_id": layer_id,
                    "stages": stages,
                    "start": interval(stages["A"])[0],
                }
            )
    return calls


def select_window(calls: list[dict]) -> list[dict]:
    by_layer: dict[int, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for call in calls:
        by_layer[call["layer_id"]][call["microbatch_id"]].append(call)
    best = None
    for layer_id, by_ubatch in by_layer.items():
        if any(not by_ubatch[ubatch_id] for ubatch_id in range(4)):
            continue
        for entries in by_ubatch.values():
            entries.sort(key=lambda call: call["start"])
        anchors = [call for entries in by_ubatch.values() for call in entries]
        for anchor in anchors:
            selected = [
                min(
                    by_ubatch[ubatch_id],
                    key=lambda call: abs(call["start"] - anchor["start"]),
                )
                for ubatch_id in range(4)
            ]
            a_events = [call["stages"]["A"] for call in selected]
            e_events = [call["stages"]["E"] for call in selected]
            compute_overlap = any(
                left_index != right_index and overlaps(a_event, e_event)
                for left_index, a_event in enumerate(a_events)
                for right_index, e_event in enumerate(e_events)
            )
            expert_overlap = any(
                overlaps(left, right)
                for index, left in enumerate(e_events)
                for right in e_events[index + 1 :]
            )
            if not compute_overlap or not expert_overlap:
                continue
            spread = max(call["start"] for call in selected) - min(
                call["start"] for call in selected
            )
            score = (spread, layer_id)
            if best is None or score < best[0]:
                best = (score, selected)
    if best is None:
        raise ValueError("no four-uBatch window with A/E and concurrent-E overlap")
    return sorted(best[1], key=lambda call: call["microbatch_id"])


def render(events: list[dict], output: Path) -> dict:
    selected = select_window(complete_calls(events))
    stage_events = [
        call["stages"][stage] for call in selected for stage in STAGES
    ]
    origin = min(interval(event)[0] for event in stage_events)
    finish = max(interval(event)[1] for event in stage_events)
    span = max(finish - origin, 1.0)
    chart_left = 120
    chart_width = 1040
    row_height = 70
    chart_top = 86
    svg_height = chart_top + row_height * 4 + 90

    def x(timestamp: float) -> float:
        return chart_left + (timestamp - origin) / span * chart_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="{svg_height}" viewBox="0 0 1240 {svg_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.label{font-size:14px}.small{font-size:11px}.title{font-size:20px;font-weight:700}</style>',
        '<text x="40" y="34" class="title">Expert-Kit four-stage pipeline — real trace window</text>',
        f'<text x="40" y="58" class="small">layer {selected[0]["layer_id"]}; window {span / 1000:.3f} ms; bars are time-scaled</text>',
    ]
    for tick in range(6):
        timestamp = origin + span * tick / 5
        xpos = x(timestamp)
        parts.append(
            f'<line x1="{xpos:.2f}" y1="70" x2="{xpos:.2f}" y2="{chart_top + row_height * 4}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{xpos:.2f}" y="78" text-anchor="middle" class="small">{(timestamp - origin) / 1000:.2f} ms</text>'
        )
    for row, call in enumerate(selected):
        ypos = chart_top + row * row_height
        parts.append(
            f'<text x="24" y="{ypos + 29}" class="label">uBatch {call["microbatch_id"]}</text>'
        )
        parts.append(
            f'<line x1="{chart_left}" y1="{ypos + 34}" x2="{chart_left + chart_width}" y2="{ypos + 34}" stroke="#d1d5db"/>'
        )
        for stage in STAGES:
            event = call["stages"][stage]
            start, end = interval(event)
            xpos = x(start)
            width = max(x(end) - xpos, 2.0)
            label = html.escape(stage)
            parts.append(
                f'<rect x="{xpos:.2f}" y="{ypos + 10}" width="{width:.2f}" height="32" rx="4" fill="{COLORS[stage]}" opacity="0.9"><title>{label}: {(end - start) / 1000:.3f} ms</title></rect>'
            )
            if width >= 28:
                parts.append(
                    f'<text x="{xpos + width / 2:.2f}" y="{ypos + 31}" text-anchor="middle" class="small" fill="#ffffff" style="fill:#ffffff">{label}</text>'
                )
    legend_y = chart_top + row_height * 4 + 34
    for index, stage in enumerate(STAGES):
        xpos = 310 + index * 150
        parts.append(
            f'<rect x="{xpos}" y="{legend_y}" width="18" height="18" rx="3" fill="{COLORS[stage]}"/>'
        )
        parts.append(
            f'<text x="{xpos + 26}" y="{legend_y + 14}" class="label">{stage}</text>'
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))

    a_events = [call["stages"]["A"] for call in selected]
    e_events = [call["stages"]["E"] for call in selected]
    return {
        "layer_id": selected[0]["layer_id"],
        "request_ids": [call["request_id"] for call in selected],
        "window_us": span,
        "compute_expert_overlap": any(
            left_index != right_index and overlaps(a_event, e_event)
            for left_index, a_event in enumerate(a_events)
            for right_index, e_event in enumerate(e_events)
        ),
        "concurrent_experts": any(
            overlaps(left, right)
            for index, left in enumerate(e_events)
            for right in e_events[index + 1 :]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text())
    summary = render(payload.get("traceEvents", []), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
