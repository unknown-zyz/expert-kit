#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from analyze_nsys_pipeline import merge_intervals, nvtx_ranges, project_gpu_ranges


PHASE_RE = re.compile(
    r"^(?P<domain>EKC|EKR):(?P<phase>[^:]+)"
    r":req=(?P<req>[^:]+):u=(?P<u>\d+):l=(?P<l>\d+)$"
)
EXPERT_RE = re.compile(
    r"^EK:E:req=(?P<req>[^:]+):u=(?P<u>\d+):l=(?P<l>\d+)"
    r":expert=(?P<expert>.+)$"
)
ATTENTION_RE = re.compile(r"^EK:A:u=(?P<u>\d+):l=(?P<l>\d+)$")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary_ms(values_ns: list[int]) -> dict:
    values_ms = [value / 1e6 for value in values_ns]
    return {
        "count": len(values_ms),
        "total_ms": sum(values_ms),
        "mean_ms": statistics.mean(values_ms) if values_ms else 0.0,
        "median_ms": statistics.median(values_ms) if values_ms else 0.0,
        "p95_ms": percentile(values_ms, 0.95),
        "min_ms": min(values_ms, default=0.0),
        "max_ms": max(values_ms, default=0.0),
    }


def max_concurrency(intervals: list[tuple[int, int]]) -> int:
    points = []
    for start, end in intervals:
        points.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _time, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def first(events: list[dict]) -> dict | None:
    return min(events, key=lambda event: event["start"]) if events else None


def analyze_sqlite(sqlite_path: Path, mode: str) -> dict:
    with sqlite3.connect(sqlite_path) as connection:
        all_ranges = nvtx_ranges(connection)
        windows = [
            event for event in all_ranges if event["label"] == "EK_PROFILE_WINDOW"
        ]
        if not windows:
            raise ValueError(f"{sqlite_path}: EK_PROFILE_WINDOW is missing")
        window = max(windows, key=lambda event: event["end"] - event["start"])
        ranges = [
            event
            for event in all_ranges
            if event["label"] != "EK_PROFILE_WINDOW"
            and max(event["start"], window["start"])
            < min(event["end"], window["end"])
        ]
        project_gpu_ranges(connection, ranges)

    startup_phases: dict[str, list[dict]] = defaultdict(list)
    for event in all_ranges:
        if match := PHASE_RE.match(event["label"]):
            phase = match["phase"]
            if phase in {"GRPC_CHANNEL_CREATE", "GRPC_CHANNEL_READY"}:
                startup_phases[phase].append(event)

    phases: dict[str, list[dict]] = defaultdict(list)
    by_request: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    experts: dict[str, list[dict]] = defaultdict(list)
    generic: dict[str, list[dict]] = defaultdict(list)
    attention_ranges: list[dict] = []
    for event in ranges:
        if match := ATTENTION_RE.match(event["label"]):
            event["microbatch_id"] = int(match["u"])
            event["layer_id"] = int(match["l"])
            attention_ranges.append(event)
        elif match := PHASE_RE.match(event["label"]):
            event["request_id"] = match["req"]
            event["microbatch_id"] = int(match["u"])
            event["layer_id"] = int(match["l"])
            phase = match["phase"]
            phases[phase].append(event)
            by_request[match["req"]][phase].append(event)
        elif match := EXPERT_RE.match(event["label"]):
            event["request_id"] = match["req"]
            event["expert_id"] = match["expert"]
            experts[match["req"]].append(event)
        elif event["label"].startswith("EKR:"):
            generic[event["label"]].append(event)

    phase_summaries = {
        name: summary_ms([event["end"] - event["start"] for event in events])
        for name, events in sorted(phases.items())
    }
    generic_summaries = {
        name.removeprefix("EKR:"): summary_ms(
            [event["end"] - event["start"] for event in events]
        )
        for name, events in sorted(generic.items())
    }
    gpu_compute_events = {"ATTENTION": attention_ranges}
    for phase in ("ROUTER_TOPK", "WEIGHTED_COMBINE"):
        gpu_compute_events[phase] = phases.get(phase, [])
    gpu_compute = {}
    for name, events in gpu_compute_events.items():
        durations = []
        for event in events:
            intervals = merge_intervals(
                [
                    (activity["start"], activity["end"])
                    for activity in event.get("gpu", [])
                    if activity["kind"] == "kernel"
                ]
            )
            if intervals:
                durations.append(sum(end - start for start, end in intervals))
        gpu_compute[name] = {
            **summary_ms(durations),
            "nvtx_range_count": len(events),
            "ranges_with_gpu_activity": len(durations),
        }

    copies: dict[str, list[tuple[int, int]]] = defaultdict(list)
    copy_bytes: dict[str, list[int]] = defaultdict(list)
    for phase in ("D2H_ENQUEUE", "H2D_ENQUEUE", "H2D_COPY"):
        for event in phases.get(phase, []):
            for activity in event.get("gpu", []):
                if activity["kind"] != "memcpy":
                    continue
                copies[phase].append((activity["start"], activity["end"]))
                copy_bytes[phase].append(activity["bytes"])

    derived: dict[str, list[int]] = defaultdict(list)
    complete_requests = 0
    worker_rpc_count = 0
    expert_count = 0
    per_request = []
    for request_id, request_phases in by_request.items():
        client_grpc = first(request_phases.get("CLIENT_CONTROLLER_GRPC", []))
        controller = first(request_phases.get("CONTROLLER_REQUEST", []))
        worker_grpc = request_phases.get("CONTROLLER_WORKER_GRPC", [])
        queues = request_phases.get("WORKER_QUEUE", [])
        request_experts = experts.get(request_id, [])
        if not (client_grpc and controller and worker_grpc and request_experts):
            continue
        complete_requests += 1
        worker_rpc_count += len(worker_grpc)
        expert_count += len(request_experts)
        first_worker = min(event["start"] for event in worker_grpc)
        last_worker = max(event["end"] for event in worker_grpc)
        first_expert = min(event["start"] for event in request_experts)
        last_expert = max(event["end"] for event in request_experts)
        derived["client_to_controller_ingress"].append(
            max(0, controller["start"] - client_grpc["start"])
        )
        derived["controller_before_worker"].append(
            max(0, first_worker - controller["start"])
        )
        derived["worker_rpc_span"].append(last_worker - first_worker)
        derived["expert_wall_span"].append(last_expert - first_expert)
        derived["expert_active_union"].append(
            sum(
                end - start
                for start, end in merge_intervals(
                    [(event["start"], event["end"]) for event in request_experts]
                )
            )
        )
        if queues:
            derived["worker_queue_total"].append(
                sum(event["end"] - event["start"] for event in queues)
            )
        derived["controller_after_worker"].append(
            max(0, controller["end"] - last_worker)
        )
        derived["controller_to_client_return"].append(
            max(0, client_grpc["end"] - controller["end"])
        )
        derived["client_controller_grpc"].append(
            client_grpc["end"] - client_grpc["start"]
        )
        per_request.append(
            {
                "request_id": request_id,
                "layer_id": controller["layer_id"],
                "microbatch_id": controller["microbatch_id"],
                "worker_rpcs": len(worker_grpc),
                "experts": len(request_experts),
            }
        )

    expert_intervals = [
        (event["start"], event["end"])
        for request_events in experts.values()
        for event in request_events
    ]
    layers = {item["layer_id"] for item in per_request}
    client_requests = len(phases.get("CLIENT_CONTROLLER_GRPC", []))
    return {
        "mode": mode,
        "source": str(sqlite_path),
        "profile_window_ms": (window["end"] - window["start"]) / 1e6,
        "counts": {
            "client_requests": client_requests,
            "complete_requests": complete_requests,
            "worker_rpcs": worker_rpc_count,
            "expert_invocations": expert_count,
            "layers": len(layers),
            "worker_rpcs_per_client_request": (
                worker_rpc_count / complete_requests if complete_requests else 0.0
            ),
            "expert_max_concurrency": max_concurrency(expert_intervals),
        },
        "phases": phase_summaries,
        "startup_phases": {
            name: summary_ms(
                [event["end"] - event["start"] for event in events]
            )
            for name, events in sorted(startup_phases.items())
        },
        "worker_nested_phases": generic_summaries,
        "gpu_compute": gpu_compute,
        "derived": {
            name: summary_ms(values) for name, values in sorted(derived.items())
        },
        "gpu_copies": {
            phase: {
                **summary_ms([end - start for start, end in intervals]),
                "total_bytes": sum(copy_bytes[phase]),
                "mean_bytes": (
                    statistics.mean(copy_bytes[phase])
                    if copy_bytes[phase]
                    else 0
                ),
            }
            for phase, intervals in sorted(copies.items())
        },
        "requests": per_request,
    }


def ensure_sqlite(path: Path) -> Path:
    if path.suffix != ".nsys-rep":
        return path
    sqlite_path = path.with_suffix(".sqlite")
    subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "--output", str(sqlite_path), str(path)],
        check=True,
    )
    return sqlite_path


def first_divergence(expected: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(
        zip(expected, actual, strict=False)
    ):
        if left != right:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def analyze_benchmarks(paths: dict[str, Path]) -> dict:
    results = {
        mode: json.loads(path.resolve(strict=True).read_text())
        for mode, path in paths.items()
    }
    reference = results["sync"]["runs"][0]["outputs"]
    sync_throughput = results["sync"]["summary"][
        "tokens_per_second_median"
    ]
    sync_latency = results["sync"]["summary"][
        "latency_seconds_median"
    ]
    modes = {}
    for mode, result in results.items():
        comparisons = []
        strict_equivalence = True
        within_mode_reference = result["runs"][0]["outputs"]
        within_mode_divergences = []
        within_mode_stable = True
        for run in result["runs"]:
            for prompt_index, (expected, actual) in enumerate(
                zip(reference, run["outputs"], strict=True)
            ):
                divergence = first_divergence(
                    expected["token_ids"], actual["token_ids"]
                )
                strict_equivalence = (
                    strict_equivalence and divergence is None
                )
                if divergence is not None:
                    comparisons.append(
                        {
                            "repetition": run["repetition"],
                            "prompt_index": prompt_index,
                            "first_divergence": divergence,
                        }
                    )
            for prompt_index, (expected, actual) in enumerate(
                zip(within_mode_reference, run["outputs"], strict=True)
            ):
                divergence = first_divergence(
                    expected["token_ids"], actual["token_ids"]
                )
                within_mode_stable = (
                    within_mode_stable and divergence is None
                )
                if divergence is not None:
                    within_mode_divergences.append(
                        {
                            "repetition": run["repetition"],
                            "prompt_index": prompt_index,
                            "first_divergence": divergence,
                        }
                    )
        throughput = result["summary"]["tokens_per_second_median"]
        latency = result["summary"]["latency_seconds_median"]
        modes[mode] = {
            "source": str(paths[mode]),
            "summary": result["summary"],
            "throughput_change_vs_sync_percent": (
                throughput / sync_throughput - 1
            )
            * 100,
            "latency_change_vs_sync_percent": (
                latency / sync_latency - 1
            )
            * 100,
            "strict_token_equivalence_to_sync": strict_equivalence,
            "divergences": comparisons,
            "within_mode_stable": within_mode_stable,
            "within_mode_divergences": within_mode_divergences,
        }
    return {"modes": modes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", type=Path, required=True)
    parser.add_argument("--pipeline-serial", type=Path, required=True)
    parser.add_argument("--pipeline-parallel", type=Path, required=True)
    parser.add_argument("--benchmark-sync", type=Path)
    parser.add_argument("--benchmark-pipeline-serial", type=Path)
    parser.add_argument("--benchmark-pipeline-parallel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "sync": args.sync,
        "pipeline-serial": args.pipeline_serial,
        "pipeline-parallel": args.pipeline_parallel,
    }
    payload = {
        "modes": {
            mode: analyze_sqlite(ensure_sqlite(path.resolve(strict=True)), mode)
            for mode, path in inputs.items()
        }
    }
    benchmark_inputs = {
        "sync": args.benchmark_sync,
        "pipeline-serial": args.benchmark_pipeline_serial,
        "pipeline-parallel": args.benchmark_pipeline_parallel,
    }
    if any(benchmark_inputs.values()):
        if not all(benchmark_inputs.values()):
            parser.error("all three --benchmark-* paths are required together")
        payload["benchmarks"] = analyze_benchmarks(benchmark_inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        mode: {
            "window_ms": result["profile_window_ms"],
            "counts": result["counts"],
            "derived": result["derived"],
            "gpu_copies": result["gpu_copies"],
        }
        for mode, result in payload["modes"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
