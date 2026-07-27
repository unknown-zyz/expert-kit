#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path


A_RE = re.compile(r"^EK:A:u=(?P<u>\d+):l=(?P<l>\d+)$")
STAGE_RE = re.compile(
    r"^EK:(?P<stage>A2E|E2A):req=(?P<req>[^:]+):u=(?P<u>\d+):l=(?P<l>\d+)$"
)
E_RE = re.compile(
    r"^EK:E:req=(?P<req>[^:]+):u=(?P<u>\d+):l=(?P<l>\d+):expert=(?P<expert>.+)$"
)


def merge_intervals(
    intervals: list[tuple[int, int]], gap_ns: int = 0
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap_ns:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersect_intervals(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    left = merge_intervals(left)
    right = merge_intervals(right)
    intersections = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            intersections.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return intersections


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def nvtx_ranges(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        """
        SELECT start, end, globalTid, COALESCE(text, strings.value) AS label
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.textId
        WHERE end IS NOT NULL
          AND (
            COALESCE(text, strings.value) = 'EK_PROFILE_WINDOW'
            OR COALESCE(text, strings.value) LIKE 'EK:%'
            OR COALESCE(text, strings.value) LIKE 'EKC:%'
            OR COALESCE(text, strings.value) LIKE 'EKR:%'
          )
        ORDER BY start
        """
    )
    return [
        {"start": row[0], "end": row[1], "global_tid": row[2], "label": row[3]}
        for row in rows
    ]


def gpu_activities(connection: sqlite3.Connection) -> dict[int, list[dict]]:
    activities: dict[int, list[dict]] = defaultdict(list)
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        rows = connection.execute(
            """
            SELECT correlationId, start, end, globalPid, strings.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
            LEFT JOIN StringIds AS strings ON strings.id = kernels.demangledName
            """
        )
        for correlation, start, end, pid, name in rows:
            activities[correlation].append(
                {
                    "kind": "kernel",
                    "start": start,
                    "end": end,
                    "global_pid": pid,
                    "name": name or "kernel",
                }
            )
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        rows = connection.execute(
            """
            SELECT correlationId, start, end, globalPid, copyKind, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        )
        for correlation, start, end, pid, copy_kind, size in rows:
            activities[correlation].append(
                {
                    "kind": "memcpy",
                    "start": start,
                    "end": end,
                    "global_pid": pid,
                    "copy_kind": copy_kind,
                    "bytes": size,
                }
            )
    return activities


def project_gpu_ranges(
    connection: sqlite3.Connection, ranges: list[dict]
) -> None:
    activities = gpu_activities(connection)
    runtime_by_tid: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for start, end, tid, correlation in connection.execute(
            """
            SELECT start, end, globalTid, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE correlationId IS NOT NULL
            ORDER BY start
            """
        ):
            runtime_by_tid[tid].append((start, end, correlation))
    for event in ranges:
        projected = []
        for start, _end, correlation in runtime_by_tid[event["global_tid"]]:
            if event["start"] <= start <= event["end"]:
                projected.extend(activities.get(correlation, ()))
        event["gpu"] = projected


def on_cpu_intervals(
    connection: sqlite3.Connection, global_tid: int, start: int, end: int
) -> list[tuple[int, int]]:
    if not table_exists(connection, "SCHED_EVENTS"):
        return [(start, end)]
    previous = connection.execute(
        """
        SELECT isSchedIn FROM SCHED_EVENTS
        WHERE globalTid = ? AND start <= ? ORDER BY start DESC LIMIT 1
        """,
        (global_tid, start),
    ).fetchone()
    events = list(
        connection.execute(
            """
            SELECT start, isSchedIn FROM SCHED_EVENTS
            WHERE globalTid = ? AND start > ? AND start < ? ORDER BY start
            """,
            (global_tid, start, end),
        )
    )
    if previous is None and not events:
        return [(start, end)]
    active = previous is not None and bool(previous[0])
    active_start = start if active else None
    intervals = []
    for timestamp, scheduled_in in events:
        if scheduled_in and not active:
            active = True
            active_start = timestamp
        elif not scheduled_in and active:
            intervals.append((active_start, timestamp))
            active = False
            active_start = None
    if active:
        intervals.append((active_start, end))
    return intervals


def build_calls(connection: sqlite3.Connection) -> list[dict]:
    ranges = nvtx_ranges(connection)
    profile_windows = [
        (event["start"], event["end"])
        for event in ranges
        if event["label"] == "EK_PROFILE_WINDOW"
    ]
    if profile_windows:
        ranges = [
            event
            for event in ranges
            if event["label"] != "EK_PROFILE_WINDOW"
            and any(
                max(event["start"], start) < min(event["end"], end)
                for start, end in profile_windows
            )
        ]
    project_gpu_ranges(connection, ranges)
    attention_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    stages: dict[str, dict[str, dict]] = defaultdict(dict)
    experts: dict[str, list[dict]] = defaultdict(list)
    for event in ranges:
        if match := A_RE.match(event["label"]):
            event["microbatch_id"] = int(match["u"])
            event["layer_id"] = int(match["l"])
            attention_by_key[(event["microbatch_id"], event["layer_id"])].append(event)
        elif match := STAGE_RE.match(event["label"]):
            event["request_id"] = match["req"]
            event["microbatch_id"] = int(match["u"])
            event["layer_id"] = int(match["l"])
            stages[event["request_id"]][match["stage"]] = event
        elif match := E_RE.match(event["label"]):
            event["request_id"] = match["req"]
            event["microbatch_id"] = int(match["u"])
            event["layer_id"] = int(match["l"])
            event["expert_id"] = match["expert"]
            experts[event["request_id"]].append(event)

    calls = []
    used_attention: set[int] = set()
    for request_id, request_stages in stages.items():
        if "A2E" not in request_stages or "E2A" not in request_stages:
            continue
        a2e_marker = request_stages["A2E"]
        candidates = [
            event
            for event in attention_by_key[
                (a2e_marker["microbatch_id"], a2e_marker["layer_id"])
            ]
            if event["start"] <= a2e_marker["start"] and id(event) not in used_attention
        ]
        if not candidates or not experts[request_id]:
            continue
        attention = max(candidates, key=lambda event: event["start"])
        used_attention.add(id(attention))
        a_intervals = merge_intervals(
            [
                (activity["start"], activity["end"])
                for activity in attention["gpu"]
                if activity["kind"] == "kernel"
            ]
        )
        h2d = [
            activity
            for activity in request_stages["E2A"]["gpu"]
            if activity["kind"] == "memcpy" and activity["copy_kind"] == 1
        ]
        d2h = [
            activity
            for activity in a2e_marker["gpu"]
            if activity["kind"] == "memcpy" and activity["copy_kind"] == 2
        ]
        if not a_intervals or not h2d:
            continue
        expert_ranges = experts[request_id]
        e_wall = [(event["start"], event["end"]) for event in expert_ranges]
        e_active = merge_intervals(
            [
                interval
                for event in expert_ranges
                for interval in on_cpu_intervals(
                    connection,
                    event["global_tid"],
                    event["start"],
                    event["end"],
                )
            ]
        )
        # The A2E NVTX range starts after local post-attention work has
        # prepared the routed request.  Use that boundary for transport so
        # router/top-k work is attributed to local accelerator compute rather
        # than communication.
        a2e_start = a2e_marker["start"]
        a2e_end = min(start for start, _end in e_wall)
        e2a_start = max(end for _start, end in e_wall)
        e2a_end = max(activity["end"] for activity in h2d)
        if a2e_end < a2e_start or e2a_end < e2a_start:
            continue
        calls.append(
            {
                "request_id": request_id,
                "microbatch_id": a2e_marker["microbatch_id"],
                "layer_id": a2e_marker["layer_id"],
                "A": a_intervals,
                "A_envelope": [a_intervals[0][0], a_intervals[-1][1]],
                "A_dispatch_start": a2e_start,
                "A2E": [[a2e_start, a2e_end]],
                "E": e_active,
                "E_envelope": [min(start for start, _end in e_wall), e2a_start],
                "E_wall_ns": interval_duration(e_wall),
                "E_active_ns": interval_duration(e_active),
                "E2A": [[e2a_start, e2a_end]],
                "d2h": [
                    [activity["start"], activity["end"], activity["bytes"]]
                    for activity in d2h
                ],
                "h2d": [
                    [activity["start"], activity["end"], activity["bytes"]]
                    for activity in h2d
                ],
                "expert_count": len(expert_ranges),
            }
        )

    # Attribute all local work between the preceding layer's returned tensor
    # and this layer's dispatch to A.  This includes the previous layer's
    # weighted combine plus this layer's norms, Attention, router and top-k.
    # The interval is a wall-clock stage span; A remains the measured CUDA
    # kernel union used by the hardware-overlap calculation.
    for call in calls:
        previous = [
            candidate
            for candidate in calls
            if candidate["microbatch_id"] == call["microbatch_id"]
            and candidate["layer_id"] == call["layer_id"] - 1
            and candidate["E2A"][0][1] <= call["A_envelope"][0]
        ]
        stage_start = (
            max(candidate["E2A"][0][1] for candidate in previous)
            if previous
            else call["A_envelope"][0]
        )
        stage_end = max(call["A_dispatch_start"], call["A_envelope"][1])
        call["A_attributed"] = [[stage_start, stage_end]]
    return calls


def select_two_layers(calls: list[dict]) -> list[dict]:
    by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for call in calls:
        by_key[(call["layer_id"], call["microbatch_id"])].append(call)
    for entries in by_key.values():
        entries.sort(key=lambda call: call["A_envelope"][0])
    candidates = []
    layers = sorted({call["layer_id"] for call in calls})
    ubatches = sorted({call["microbatch_id"] for call in calls})
    for layer in layers:
        if layer + 1 not in layers:
            continue
        anchors = by_key.get((layer, 0), [])
        for anchor in anchors:
            first = []
            for ubatch in ubatches:
                choices = by_key.get((layer, ubatch), [])
                if not choices:
                    break
                first.append(
                    min(
                        choices,
                        key=lambda call: abs(
                            call["A_envelope"][0] - anchor["A_envelope"][0]
                        ),
                    )
                )
            if len(first) != len(ubatches):
                continue
            second = []
            for call in first:
                choices = [
                    candidate
                    for candidate in by_key.get(
                        (layer + 1, call["microbatch_id"]), []
                    )
                    if candidate["A_envelope"][0] >= call["E2A"][0][1]
                ]
                if not choices:
                    break
                second.append(min(choices, key=lambda item: item["A_envelope"][0]))
            if len(second) != len(ubatches) or len({item["request_id"] for item in second}) != len(ubatches):
                continue
            selected = first + second
            span = max(item["E2A"][0][1] for item in selected) - min(
                item["A_envelope"][0] for item in selected
            )
            candidates.append((span, selected))
    if not candidates:
        raise ValueError("no complete adjacent-layer window for four micro-batches")
    return min(candidates, key=lambda item: item[0])[1]


def hiding_metrics(
    calls: list[dict], compute_calls: list[dict] | None = None
) -> dict:
    result = {}
    all_compute = {
        call["request_id"]: merge_intervals(
            [tuple(interval) for interval in call["A"] + call["E"]]
        )
        for call in (compute_calls or calls)
    }
    totals = {"communication_ns": 0, "hidden_ns": 0}
    for stage in ("A2E", "E2A"):
        communication_total = hidden_total = 0
        per_call = []
        for call in calls:
            communication = [tuple(interval) for interval in call[stage]]
            other_compute = merge_intervals(
                [
                    interval
                    for request_id, intervals in all_compute.items()
                    if request_id != call["request_id"]
                    for interval in intervals
                ]
            )
            duration = interval_duration(communication)
            hidden = interval_duration(intersect_intervals(communication, other_compute))
            communication_total += duration
            hidden_total += hidden
            per_call.append(
                {
                    "request_id": call["request_id"],
                    "layer_id": call["layer_id"],
                    "microbatch_id": call["microbatch_id"],
                    "communication_ns": duration,
                    "hidden_ns": hidden,
                    "exposed_ns": duration - hidden,
                    "hide_ratio": hidden / duration if duration else 1.0,
                }
            )
        result[stage] = {
            "communication_ns": communication_total,
            "hidden_ns": hidden_total,
            "exposed_ns": communication_total - hidden_total,
            "hide_ratio": hidden_total / communication_total
            if communication_total
            else 1.0,
            "per_call": per_call,
        }
        totals["communication_ns"] += communication_total
        totals["hidden_ns"] += hidden_total
    totals["exposed_ns"] = totals["communication_ns"] - totals["hidden_ns"]
    totals["hide_ratio"] = (
        totals["hidden_ns"] / totals["communication_ns"]
        if totals["communication_ns"]
        else 1.0
    )
    totals["classification"] = (
        "mostly-hidden" if totals["hide_ratio"] >= 0.95 else
        "partially-hidden" if totals["hide_ratio"] > 0 else
        "not-hidden"
    )
    result["total"] = totals
    return result


def analyze(sqlite_path: Path) -> dict:
    with sqlite3.connect(sqlite_path) as connection:
        calls = build_calls(connection)
        selected = select_two_layers(calls)

    def duration_ms(call: dict, stage: str) -> float:
        intervals = call.get(
            "A_attributed", [call["A_envelope"]]
        ) if stage == "A" else call[stage]
        return sum(end - start for start, end in intervals) / 1e6

    all_layer_metrics = {}
    for layer_id in sorted({call["layer_id"] for call in calls}):
        layer_calls = [call for call in calls if call["layer_id"] == layer_id]
        all_layer_metrics[str(layer_id)] = {
            stage: {
                "count": len(layer_calls),
                "mean_ms": (
                    sum(duration_ms(call, stage) for call in layer_calls)
                    / len(layer_calls)
                    if layer_calls
                    else 0.0
                ),
                "total_ms": sum(duration_ms(call, stage) for call in layer_calls),
            }
            for stage in ("A", "A2E", "E", "E2A")
        }
    return {
        "source": str(sqlite_path),
        "call_count": len(calls),
        "microbatches": sorted({call["microbatch_id"] for call in calls}),
        "all_layer_metrics": all_layer_metrics,
        "selected_layers": sorted({call["layer_id"] for call in selected}),
        "calls": selected,
        "metrics": hiding_metrics(selected, calls),
        "visible_window_metrics": hiding_metrics(selected, selected),
        "definitions": {
            "A": "union of CUDA kernels launched by full Qwen3MoE Attention",
            "A_attributed": (
                "previous layer E2A completion to dispatch; includes local weighted "
                "combine, norms, Attention, router and top-k"
            ),
            "A2E": "dispatch NVTX start to first Worker Expert compute",
            "E": "Worker exp.forward on-CPU intervals",
            "E2A": "last Worker Expert completion to final correlated H2D completion",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="nsys .sqlite or .nsys-rep")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    input_path = args.input.resolve()
    if input_path.suffix == ".nsys-rep":
        sqlite_path = input_path.with_suffix(".sqlite")
        subprocess.run(
            [
                "nsys",
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                "--quiet=true",
                f"--output={sqlite_path}",
                str(input_path),
            ],
            check=True,
        )
    else:
        sqlite_path = input_path
    result = analyze(sqlite_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
