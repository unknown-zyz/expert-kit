#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


STAGES = ("A", "A2E", "E", "E2A")


def interval(event: dict) -> tuple[float, float]:
    start = float(event["ts"])
    return start, start + float(event["dur"])


def overlaps(left: dict, right: dict) -> bool:
    left_start, left_end = interval(left)
    right_start, right_end = interval(right)
    return max(left_start, right_start) < min(left_end, right_end)


def validate(events: list[dict]) -> dict:
    pipeline_events = [
        event
        for event in events
        if event.get("cat") == "expertkit_pipeline" and event.get("name") in STAGES
    ]
    calls: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    for event in pipeline_events:
        args = event.get("args", {})
        key = (
            str(args.get("request_id", "")),
            int(args.get("microbatch_id", -1)),
            int(args.get("layer_id", -1)),
        )
        calls[key][event["name"]] = event

    if not calls:
        raise ValueError("trace contains no Expert-Kit pipeline events")

    complete_calls = {}
    incomplete_tail_calls = []
    trace_end = max(interval(event)[1] for event in pipeline_events)
    for key, stages in calls.items():
        missing = [stage for stage in STAGES if stage not in stages]
        if missing:
            latest = max(interval(event)[1] for event in stages.values())
            # Periodic snapshots can capture calls that are still in flight at
            # the file boundary. Only tolerate incomplete calls at that tail.
            if trace_end - latest <= 1_000_000:
                incomplete_tail_calls.append(key)
                continue
            raise ValueError(f"{key} is missing stages away from trace tail: {missing}")
        complete_calls[key] = stages
        previous_end = None
        for stage in STAGES:
            start, end = interval(stages[stage])
            if previous_end is not None and start + 1e-3 < previous_end:
                raise ValueError(f"{key} has out-of-order stage {stage}")
            previous_end = end

    calls = complete_calls
    ubatches = {key[1] for key in calls}
    if len(ubatches) < 4:
        raise ValueError(f"expected four uBatches, observed {sorted(ubatches)}")

    a_events = [event for event in pipeline_events if event["name"] == "A"]
    e_events = [event for event in pipeline_events if event["name"] == "E"]
    compute_comm_overlap = any(
        a["args"]["microbatch_id"] != e["args"]["microbatch_id"] and overlaps(a, e)
        for a in a_events
        for e in e_events
    )
    concurrent_experts = any(
        left["args"]["microbatch_id"] != right["args"]["microbatch_id"]
        and overlaps(left, right)
        for index, left in enumerate(e_events)
        for right in e_events[index + 1 :]
    )
    if not compute_comm_overlap:
        raise ValueError("no E(uBatch i) / A(uBatch j) overlap was observed")
    if not concurrent_experts:
        raise ValueError("no concurrently in-flight E stages were observed")

    return {
        "calls": len(calls),
        "incomplete_tail_calls": len(incomplete_tail_calls),
        "microbatches": sorted(ubatches),
        "compute_comm_overlap": compute_comm_overlap,
        "concurrent_experts": concurrent_experts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text())
    result = validate(payload.get("traceEvents", []))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
