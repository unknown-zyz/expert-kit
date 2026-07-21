#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROMPTS = [
    "Hello, my name is",
    "The president of the United",
    "In a distant future, humanity",
    "The key idea behind a pipeline is",
]


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(runs: list[dict]) -> dict:
    latencies = [float(run["latency_seconds"]) for run in runs]
    throughputs = [float(run["tokens_per_second"]) for run in runs]
    return {
        "repetitions": len(runs),
        "latency_seconds_median": statistics.median(latencies),
        "latency_seconds_p95": percentile95(latencies),
        "tokens_per_second_median": statistics.median(throughputs),
        "tokens_per_second_p95": percentile95(throughputs),
    }


def output_payload(outputs) -> list[dict]:
    return [
        {
            "prompt": output.prompt,
            "token_ids": list(output.outputs[0].token_ids),
            "text": output.outputs[0].text,
        }
        for output in outputs
    ]


def run_child(args: argparse.Namespace) -> None:
    pipeline_enabled = args.child_mode == "pipeline"
    os.environ["EK_PIPELINE_ENABLE"] = "1" if pipeline_enabled else "0"
    if pipeline_enabled and args.trace is not None:
        os.environ["EK_PIPELINE_TRACE"] = str(args.trace)
    else:
        os.environ.pop("EK_PIPELINE_TRACE", None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_MLA_DISABLE", "1")
    os.environ.setdefault("EK_ENABLE", "1")
    os.environ.setdefault("EK_MODEL_NAME", "qwen3-30b-a3b")
    os.environ.setdefault("EK_MODE", "expert_mode")
    os.environ.setdefault("EK_ADDR", "localhost:5002")
    os.environ.setdefault("EK_CLIENT_TIMEOUT", "120")

    model_value = os.environ.get("QWEN3_30B_A3B_ROOT")
    if not model_value:
        raise RuntimeError("QWEN3_30B_A3B_ROOT is required")
    model_root = Path(model_value).expanduser().resolve(strict=True)

    if args.nsys_capture:
        # Nsight Systems 2026.1 process-tree tracing can leave the short-lived
        # `file` subprocess used by platform.architecture() as a zombie while
        # py-cpuinfo waits on its stdout pipe. The result is invariant for this
        # x86_64 profiling environment, so avoid that unrelated helper process.
        import platform

        def nsys_architecture(*_args, **_kwargs):
            return ("64bit", "ELF")

        platform.architecture = nsys_architecture

    import importlib.metadata

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_root),
        trust_remote_code=True,
        max_model_len=256,
        enforce_eager=True,
        cpu_offload_gb=64,
        max_num_batched_tokens=1024,
        ubatch_size=4 if pipeline_enabled else 0,
        dbo_decode_token_threshold=0,
        dbo_prefill_token_threshold=0,
        seed=0,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=0)
    for _ in range(args.warmup):
        llm.generate(PROMPTS, sampling, use_tqdm=False)

    runs = []
    for repetition in range(args.repetitions):
        capture = args.nsys_capture and repetition == 0
        if capture:
            torch.cuda.nvtx.range_push("EK_PROFILE_WINDOW")
        started = time.perf_counter()
        try:
            outputs = llm.generate(PROMPTS, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
        finally:
            if capture:
                torch.cuda.nvtx.range_pop()
        generated = output_payload(outputs)
        total_tokens = sum(len(item["token_ids"]) for item in generated)
        runs.append(
            {
                "repetition": repetition,
                "latency_seconds": elapsed,
                "total_output_tokens": total_tokens,
                "tokens_per_second": total_tokens / elapsed,
                "outputs": generated,
            }
        )
    result = {
        "mode": args.child_mode,
        "pipeline_enabled": pipeline_enabled,
        "model": str(model_root),
        "environment": {
            "vllm": importlib.metadata.version("vllm"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "settings": {
            "prompts": PROMPTS,
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "ubatch_size": 4 if pipeline_enabled else 0,
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    args.child_output.write_text(json.dumps(result, indent=2))


def first_divergence(expected: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def compare_tokens(sync: dict, pipeline: dict) -> dict:
    expected = sync["runs"][0]["outputs"]
    comparisons = []
    strict = True
    for run in pipeline["runs"]:
        for prompt_index, (left, right) in enumerate(
            zip(expected, run["outputs"], strict=True)
        ):
            divergence = first_divergence(left["token_ids"], right["token_ids"])
            strict = strict and divergence is None
            comparisons.append(
                {
                    "pipeline_repetition": run["repetition"],
                    "prompt_index": prompt_index,
                    "prompt": left["prompt"],
                    "equal": divergence is None,
                    "first_divergence": divergence,
                    "sync_text": left["text"],
                    "pipeline_text": right["text"],
                }
            )
    return {"strict_token_equivalence": strict, "comparisons": comparisons}


def mode_stability(result: dict) -> dict:
    reference = result["runs"][0]["outputs"]
    comparisons = []
    stable = True
    for run in result["runs"][1:]:
        divergences = []
        for left, right in zip(reference, run["outputs"], strict=True):
            divergence = first_divergence(left["token_ids"], right["token_ids"])
            divergences.append(divergence)
            stable = stable and divergence is None
        comparisons.append(
            {
                "repetition": run["repetition"],
                "first_divergence_by_prompt": divergences,
            }
        )
    return {"stable": stable, "comparisons": comparisons}


def run_parent(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="expertkit-pipeline-benchmark-") as temp:
        temp_root = Path(temp)
        results = {}
        for mode in ("sync", "pipeline"):
            child_output = temp_root / f"{mode}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-mode",
                mode,
                "--child-output",
                str(child_output),
                "--warmup",
                str(args.warmup),
                "--repetitions",
                str(args.repetitions),
                "--max-tokens",
                str(args.max_tokens),
            ]
            if args.trace is not None:
                command.extend(("--trace", str(args.trace)))
            if args.nsys_capture:
                command.append("--nsys-capture")
            subprocess.run(command, check=True)
            results[mode] = json.loads(child_output.read_text())

    sync_throughput = results["sync"]["summary"]["tokens_per_second_median"]
    pipeline_throughput = results["pipeline"]["summary"][
        "tokens_per_second_median"
    ]
    sync_latency = results["sync"]["summary"]["latency_seconds_median"]
    pipeline_latency = results["pipeline"]["summary"]["latency_seconds_median"]
    report = {
        "sync": results["sync"],
        "pipeline": results["pipeline"],
        "comparison": {
            "throughput_change_percent": (
                pipeline_throughput / sync_throughput - 1
            )
            * 100,
            "latency_change_percent": (pipeline_latency / sync_latency - 1) * 100,
            "sync_within_mode_stability": mode_stability(results["sync"]),
            "pipeline_within_mode_stability": mode_stability(results["pipeline"]),
            **compare_tokens(results["sync"], results["pipeline"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "sync": results["sync"]["summary"],
        "pipeline": results["pipeline"]["summary"],
        "comparison": {
            "throughput_change_percent": report["comparison"]["throughput_change_percent"],
            "latency_change_percent": report["comparison"]["latency_change_percent"],
            "strict_token_equivalence": report["comparison"]["strict_token_equivalence"],
        },
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/expertkit-pipeline-benchmark.json"))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="wrap the first measured generation in EK_PROFILE_WINDOW NVTX",
    )
    parser.add_argument("--child-mode", choices=("sync", "pipeline"))
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child_mode and args.child_output is None:
        parser.error("--child-output is required with --child-mode")
    return args


def main() -> None:
    args = parse_args()
    if args.child_mode:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
