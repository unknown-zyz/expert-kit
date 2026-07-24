#!/usr/bin/env python3
"""Correlate MoE A/A2E/E/E2A ranges across Frontend and Worker."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

PROFILE_RE = re.compile(
    r"^EKP:(?P<phase>[^:]+):call=(?P<call>[^:]+):"
    r"u=(?P<ubatch>\d+):l=(?P<layer>\d+)"
    r"(?::n=(?P<tokens>\d+))?$"
)
STAGES = ("A", "A2E", "E", "E2A")


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values_ns: list[int]) -> dict[str, float | int]:
    values_ms = [value / 1e6 for value in values_ns]
    return {
        "count": len(values_ms),
        "total_ms": sum(values_ms),
        "mean_ms": statistics.mean(values_ms) if values_ms else 0.0,
        "p50_ms": statistics.median(values_ms) if values_ms else 0.0,
        "p95_ms": _percentile(values_ms, 0.95),
    }


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _ranges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT start, end, globalTid, COALESCE(events.text, strings.value)
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.textId
        WHERE end IS NOT NULL
          AND COALESCE(events.text, strings.value) LIKE 'EKP:%'
        ORDER BY start
        """
    )
    result = []
    for start, end, global_tid, label in rows:
        match = PROFILE_RE.match(label)
        if match is None:
            continue
        result.append(
            {
                "start": start,
                "end": end,
                "global_tid": global_tid,
                "label": label,
                "phase": match["phase"],
                "call_id": match["call"],
                "microbatch_id": int(match["ubatch"]),
                "layer_id": int(match["layer"]),
                "token_count": (
                    int(match["tokens"]) if match["tokens"] is not None else None
                ),
                "gpu": [],
            }
        )
    return result


def _project_gpu(connection: sqlite3.Connection, ranges: list[dict[str, Any]]) -> None:
    activities: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for correlation, start, end, name in connection.execute(
            """
            SELECT correlationId, start, end, strings.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
            LEFT JOIN StringIds AS strings ON strings.id = kernels.demangledName
            """
        ):
            activities[correlation].append(
                {"kind": "kernel", "start": start, "end": end, "name": name}
            )
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for correlation, start, end, copy_kind, size in connection.execute(
            """
            SELECT correlationId, start, end, copyKind, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ):
            activities[correlation].append(
                {
                    "kind": "memcpy",
                    "start": start,
                    "end": end,
                    "copy_kind": copy_kind,
                    "bytes": size,
                }
            )
    runtime_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for start, tid, correlation in connection.execute(
            """
            SELECT start, globalTid, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE correlationId IS NOT NULL
            ORDER BY start
            """
        ):
            runtime_rows[tid].append((start, correlation))
    runtime = {
        tid: (
            [item[0] for item in items],
            [item[1] for item in items],
        )
        for tid, items in runtime_rows.items()
    }
    for event in ranges:
        projected: list[dict[str, Any]] = []
        timestamps, correlations = runtime.get(event["global_tid"], ([], []))
        first = bisect.bisect_left(timestamps, event["start"])
        stop = bisect.bisect_right(timestamps, event["end"])
        for correlation in correlations[first:stop]:
            projected.extend(activities.get(correlation, ()))
        event["gpu"] = projected


def _scheduler_events(
    connection: sqlite3.Connection, ranges: list[dict[str, Any]]
) -> dict[int, tuple[list[int], list[bool]]]:
    """Load scheduler transitions once instead of querying per Expert call."""
    if not _table_exists(connection, "SCHED_EVENTS"):
        return {}
    tids = sorted({event["global_tid"] for event in ranges if event["phase"] == "E"})
    if not tids:
        return {}
    placeholders = ",".join("?" for _ in tids)
    grouped: dict[int, tuple[list[int], list[bool]]] = {}
    for tid, timestamp, scheduled_in in connection.execute(
        f"""
        SELECT globalTid, start, isSchedIn FROM SCHED_EVENTS
        WHERE globalTid IN ({placeholders}) ORDER BY globalTid, start
        """,  # noqa: S608 - placeholders are generated, not user supplied
        tids,
    ):
        timestamps, states = grouped.setdefault(tid, ([], []))
        timestamps.append(timestamp)
        states.append(bool(scheduled_in))
    return grouped


def _on_cpu(
    scheduler: dict[int, tuple[list[int], list[bool]]], event: dict[str, Any]
) -> list[tuple[int, int]]:
    timeline = scheduler.get(event["global_tid"])
    if timeline is None:
        return []
    timestamps, states = timeline
    start = event["start"]
    end = event["end"]
    previous = bisect.bisect_right(timestamps, start) - 1
    active = previous >= 0 and states[previous]
    active_start = start if active else None
    intervals: list[tuple[int, int]] = []
    index = previous + 1
    while index < len(timestamps) and timestamps[index] < end:
        timestamp = timestamps[index]
        scheduled_in = states[index]
        if scheduled_in and not active:
            active = True
            active_start = timestamp
        elif not scheduled_in and active:
            assert active_start is not None
            intervals.append((active_start, timestamp))
            active = False
            active_start = None
        index += 1
    if active:
        assert active_start is not None
        intervals.append((active_start, end))
    return intervals


def _pick_scope(
    scopes: list[dict[str, Any]], call: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = [
        scope
        for scope in scopes
        if scope["microbatch_id"] == call["microbatch_id"]
        and scope["layer_id"] == call["layer_id"]
        and scope["start"] <= call["start"] <= scope["end"]
    ]
    return min(
        candidates, key=lambda event: event["end"] - event["start"], default=None
    )


def _build_calls(
    ranges: list[dict[str, Any]],
    scheduler: dict[int, tuple[list[int], list[bool]]],
) -> list[dict[str, Any]]:
    by_call: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scopes = []
    for event in ranges:
        if event["phase"] == "A_SCOPE":
            scopes.append(event)
        elif event["call_id"] != "none":
            by_call[event["call_id"]][event["phase"]].append(event)
    calls = []
    for call_id, phases in by_call.items():
        client = min(
            phases.get("A2E_CALL", ()), key=lambda event: event["start"], default=None
        )
        experts = phases.get("E", [])
        if client is None or not experts:
            continue
        scope = _pick_scope(scopes, client)
        if scope is None:
            continue
        e_intervals = _merge([(event["start"], event["end"]) for event in experts])
        e_start = e_intervals[0][0]
        e_end = e_intervals[-1][1]
        gpu = [
            activity
            for event in phases.values()
            for item in event
            for activity in item["gpu"]
        ]
        call_end = max([client["end"], *(activity["end"] for activity in gpu)])
        a_gpu = [activity for activity in scope["gpu"] if activity["kind"] == "kernel"]
        a_intervals = _merge([(item["start"], item["end"]) for item in a_gpu])
        if not a_intervals or e_start < client["start"] or call_end < e_end:
            continue
        stage_intervals = {
            "A": a_intervals,
            "A2E": [(client["start"], e_start)],
            "E": e_intervals,
            "E2A": [(e_end, call_end)],
        }
        subphases = {
            phase: _duration([(event["start"], event["end"]) for event in events])
            for phase, events in phases.items()
            if phase not in {"A2E_CALL", "E"}
        }
        copies = [activity for activity in gpu if activity["kind"] == "memcpy"]
        calls.append(
            {
                "call_id": call_id,
                "microbatch_id": client["microbatch_id"],
                "layer_id": client["layer_id"],
                "scope": [scope["start"], scope["end"]],
                "token_count": scope["token_count"],
                "stages": {
                    name: [list(item) for item in value]
                    for name, value in stage_intervals.items()
                },
                "durations_ns": {
                    name: _duration(value) for name, value in stage_intervals.items()
                },
                "e_thread_on_cpu_ns": sum(
                    _duration(_on_cpu(scheduler, event)) for event in experts
                ),
                "subphases_ns": subphases,
                "memcpy_bytes": sum(item["bytes"] for item in copies),
                "critical_path_ns": scope["end"] - scope["start"],
            }
        )
    return sorted(calls, key=lambda call: call["scope"][0])


def _decode_calls(
    calls: list[dict[str, Any]], decode_max_tokens: int | None = None
) -> list[dict[str, Any]]:
    """Keep decode scopes, using profiled token counts when available."""
    if decode_max_tokens is not None:
        with_counts = [call for call in calls if call.get("token_count") is not None]
        if len(with_counts) == len(calls):
            return [call for call in calls if call["token_count"] <= decode_max_tokens]
    layer_ids = sorted({call["layer_id"] for call in calls})
    prefill_ids = {
        min(
            (
                call
                for call in calls
                if call["layer_id"] == layer_id and call["microbatch_id"] == 0
            ),
            key=lambda call: call["scope"][0],
        )["call_id"]
        for layer_id in layer_ids
    }
    return [call for call in calls if call["call_id"] not in prefill_ids]


def _hide_metrics(calls: list[dict[str, Any]]) -> dict[str, Any]:
    # A2E ends where this call's E begins and E2A starts where it ends, while
    # this call's A precedes A2E. Its own compute therefore has zero-width
    # intersection with its communication intervals. A single global compute
    # union is equivalent to rebuilding "all calls except self" for every call.
    all_compute = _merge(
        [
            tuple(item)
            for call in calls
            for stage in ("A", "E")
            for item in call["stages"][stage]
        ]
    )
    compute_starts = [interval[0] for interval in all_compute]
    compute_ends = [interval[1] for interval in all_compute]

    def hidden_duration(communication: list[tuple[int, int]]) -> int:
        hidden = 0
        for start, end in communication:
            index = bisect.bisect_right(compute_ends, start)
            while index < len(all_compute) and compute_starts[index] < end:
                hidden += max(
                    0,
                    min(end, compute_ends[index]) - max(start, compute_starts[index]),
                )
                index += 1
        return hidden

    result: dict[str, Any] = {}
    for stage in ("A2E", "E2A"):
        total = hidden = 0
        for call in calls:
            communication = [tuple(item) for item in call["stages"][stage]]
            total += _duration(communication)
            hidden += hidden_duration(communication)
        result[stage] = {
            "total_ms": total / 1e6,
            "hidden_ms": hidden / 1e6,
            "exposed_ms": (total - hidden) / 1e6,
            "hide_ratio": hidden / total if total else 1.0,
        }
    total = sum(result[stage]["total_ms"] for stage in ("A2E", "E2A"))
    hidden = sum(result[stage]["hidden_ms"] for stage in ("A2E", "E2A"))
    result["total"] = {
        "total_ms": total,
        "hidden_ms": hidden,
        "exposed_ms": total - hidden,
        "hide_ratio": hidden / total if total else 1.0,
    }
    return result


def _layer_rows(
    calls: list[dict[str, Any]], layer_ids: set[int] | None = None
) -> list[dict[str, Any]]:
    rows = []
    observed = {call["layer_id"] for call in calls}
    for layer_id in sorted(observed if layer_ids is None else observed & layer_ids):
        selected = [call for call in calls if call["layer_id"] == layer_id]
        summaries = {
            stage: _summary([call["durations_ns"][stage] for call in selected])
            for stage in STAGES
        }
        total = sum(summary["total_ms"] for summary in summaries.values())
        rows.append(
            {
                "layer_id": layer_id,
                "call_count": len(selected),
                "stages": {
                    stage: {
                        **summary,
                        "share": summary["total_ms"] / total if total else 0.0,
                    }
                    for stage, summary in summaries.items()
                },
                "critical_path": _summary(
                    [call["critical_path_ns"] for call in selected]
                ),
                "e_thread_on_cpu": _summary(
                    [call["e_thread_on_cpu_ns"] for call in selected]
                ),
            }
        )
    return rows


def _representative(
    calls: list[dict[str, Any]], layer_ids: set[int] | None = None
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        if layer_ids is None or call["layer_id"] in layer_ids:
            grouped[(call["layer_id"], call["microbatch_id"])].append(call)
    for values in grouped.values():
        values.sort(key=lambda call: call["scope"][0])
    ubatches = sorted({call["microbatch_id"] for call in calls})
    if len(ubatches) < 2:
        ubatches = [0]
    candidates = []
    available_layers = sorted({layer for layer, _ubatch in grouped})
    for layer_id in available_layers:
        if layer_id + 1 not in available_layers:
            continue
        keys = [
            (layer, ubatch) for layer in (layer_id, layer_id + 1) for ubatch in ubatches
        ]
        if any(not grouped[key] for key in keys):
            continue
        layer_candidates = []
        # uBatch 0 can contain extra unbatched tail calls after the active
        # request count falls below the DBO threshold. Use every uBatch-1 call
        # as an anchor and pick the nearest scope from all other lanes. This
        # aligns one real scheduler wave instead of matching list ordinals.
        anchor_ubatch = ubatches[1] if len(ubatches) > 1 else ubatches[0]
        for anchor in grouped[(layer_id, anchor_ubatch)]:
            anchor_start = anchor["scope"][0]
            selected = [
                min(
                    grouped[key],
                    key=lambda call: abs(call["scope"][0] - anchor_start),
                )
                for key in keys
            ]
            starts = [call["scope"][0] for call in selected]
            span = max(starts) - min(starts)
            mean_total = statistics.mean(
                sum(call["durations_ns"].values()) for call in selected
            )
            layer_candidates.append((span, mean_total, layer_id, selected))
        if layer_candidates:
            candidates.append(min(layer_candidates, key=lambda item: item[0]))
    if not candidates:
        raise ValueError("no complete adjacent-layer window is available")
    median_total = statistics.median(item[1] for item in candidates)
    _, _, layer_id, selected = min(
        candidates, key=lambda item: (abs(item[1] - median_total), item[0])
    )
    return {
        "layers": [layer_id, layer_id + 1],
        "microbatches": ubatches,
        "calls": selected,
    }


def _analyze(
    sqlite_path: Path,
    mode: str,
    *,
    first_moe_layer: int = 1,
    num_moe_layers: int = 26,
    decode_max_tokens: int | None = None,
) -> dict[str, Any]:
    uri = f"file:{sqlite_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        ranges = _ranges(connection)
        _project_gpu(connection, ranges)
        scheduler = _scheduler_events(connection, ranges)
        calls = _decode_calls(
            _build_calls(ranges, scheduler),
            decode_max_tokens=decode_max_tokens,
        )
    expected_layers = set(range(first_moe_layer, first_moe_layer + num_moe_layers))
    observed_layers = {call["layer_id"] for call in calls}
    missing = sorted(expected_layers - observed_layers)
    if missing:
        raise ValueError(f"{mode} profile is missing MoE layers: {missing}")
    stages = {
        stage: _summary([call["durations_ns"][stage] for call in calls])
        for stage in STAGES
    }
    stage_total = sum(item["total_ms"] for item in stages.values())
    subphases: dict[str, list[int]] = defaultdict(list)
    for call in calls:
        for phase, duration in call["subphases_ns"].items():
            subphases[phase].append(duration)
    return {
        "mode": mode,
        "source": str(sqlite_path),
        "call_count": len(calls),
        "stages": {
            stage: {
                **item,
                "share": item["total_ms"] / stage_total if stage_total else 0.0,
            }
            for stage, item in stages.items()
        },
        "subphases": {
            phase: _summary(values) for phase, values in sorted(subphases.items())
        },
        "communication_hiding": _hide_metrics(calls),
        "layers": _layer_rows(calls, expected_layers),
        "representative": _representative(calls, expected_layers),
    }


def _sqlite(report: Path) -> Path:
    if report.suffix == ".sqlite":
        return report.resolve()
    output = report.with_suffix(".detailed.sqlite").resolve()
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--quiet=true",
            f"--output={output}",
            str(report.resolve()),
        ],
        check=True,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="DeepSeek-V2-Lite")
    parser.add_argument("--first-moe-layer", type=int, default=1)
    parser.add_argument("--num-moe-layers", type=int, default=26)
    parser.add_argument(
        "--decode-max-tokens",
        type=int,
        help="maximum decoder-scope token count; excludes chunked prefill",
    )
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "definitions": {
            "A": "CUDA kernel union inside the decoder layer scope",
            "A2E": "Frontend routed dispatch start to first Worker backend compute",
            "E": "Worker Torch backend compute wall intervals",
            "E2A": "last Worker compute completion to Frontend result readiness",
            "share": "stage total divided by A+A2E+E+E2A totals; overlapping calls are work share, not wall share",
            "scope": "decode calls only; token-count filtering excludes chunked prefill when --decode-max-tokens is set",
            "model": args.model_name,
            "first_moe_layer": args.first_moe_layer,
            "num_moe_layers": args.num_moe_layers,
        },
        "sync": _analyze(
            _sqlite(args.sync),
            "sync",
            first_moe_layer=args.first_moe_layer,
            num_moe_layers=args.num_moe_layers,
            decode_max_tokens=args.decode_max_tokens,
        ),
        "pipeline": _analyze(
            _sqlite(args.pipeline),
            "pipeline",
            first_moe_layer=args.first_moe_layer,
            num_moe_layers=args.num_moe_layers,
            decode_max_tokens=args.decode_max_tokens,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {mode: payload[mode]["stages"] for mode in ("sync", "pipeline")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
