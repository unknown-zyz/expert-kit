"""Deterministic offline vLLM benchmark for the Expert Kit pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ShareGPTSample:
    """One reproducibly selected ShareGPT prompt."""

    dataset_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]

    def manifest_entry(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "prompt_tokens": len(self.prompt_token_ids),
            "prompt_token_sha256": _hash_json(self.prompt_token_ids),
        }


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_sharegpt_samples(
    dataset_path: Path,
    tokenizer: Any,
    *,
    count: int,
    seed: int,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    fixed_prompt_tokens: int | None = None,
) -> tuple[ShareGPTSample, ...]:
    """Select first-turn ShareGPT prompts using a local deterministic shuffle."""

    with dataset_path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("ShareGPT dataset must contain a JSON list")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    selected: list[ShareGPTSample] = []
    for index in order:
        record = records[index]
        if not isinstance(record, dict):
            continue
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        first, second = conversations[0], conversations[1]
        if not isinstance(first, dict) or not isinstance(second, dict):
            continue
        if first.get("from") != "human" or second.get("from") != "gpt":
            continue
        prompt = first.get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        token_ids = tuple(tokenizer(prompt, add_special_tokens=True).input_ids)
        if not min_prompt_tokens <= len(token_ids) <= max_prompt_tokens:
            continue
        if fixed_prompt_tokens is not None:
            if len(token_ids) < fixed_prompt_tokens:
                continue
            token_ids = token_ids[:fixed_prompt_tokens]
        selected.append(
            ShareGPTSample(
                dataset_id=str(record.get("id", index)),
                prompt=prompt,
                prompt_token_ids=token_ids,
            )
        )
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(
        f"dataset provided only {len(selected)} eligible prompts; need {count}"
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(runs: list[dict[str, object]]) -> dict[str, float | None]:
    fields = (
        "elapsed_seconds",
        "requests_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "ttft_p50_seconds",
        "ttft_p95_seconds",
        "tpot_p50_seconds",
        "tpot_p95_seconds",
    )
    summary: dict[str, float | None] = {}
    for field in fields:
        available = [run[field] for run in runs if run[field] is not None]
        if len(available) != len(runs):
            summary[f"{field}_median"] = None
            summary[f"{field}_p95"] = None
            continue
        values = [float(value) for value in available]
        summary[f"{field}_median"] = statistics.median(values)
        summary[f"{field}_p95"] = _percentile(values, 0.95)
    return summary


def _measure(
    llm: Any,
    inputs: list[dict[str, list[int]]],
    sampling_params: Any,
    *,
    cuda_profiler_range: bool,
) -> tuple[dict[str, object], list[list[int]], list[str]]:
    import torch

    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    started = time.perf_counter()
    try:
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
    elapsed = time.perf_counter() - started
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    texts = [output.outputs[0].text for output in outputs]
    prompt_tokens = sum(len(output.prompt_token_ids or ()) for output in outputs)
    output_tokens = sum(len(tokens) for tokens in token_ids)
    request_metrics = [output.metrics for output in outputs]
    metrics_available = all(value is not None for value in request_metrics)
    if metrics_available:
        ttft = [float(value.first_token_latency) for value in request_metrics]
        tpot = [
            (value.last_token_ts - value.first_token_ts) / (len(tokens) - 1)
            for value, tokens in zip(request_metrics, token_ids, strict=True)
            if len(tokens) > 1
        ]
        corrupted_requests: int | None = sum(
            bool(value.is_corrupted) for value in request_metrics
        )
    else:
        ttft = []
        tpot = []
        corrupted_requests = None
    metrics = {
        "elapsed_seconds": elapsed,
        "request_count": len(outputs),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "requests_per_second": len(outputs) / elapsed,
        "input_tokens_per_second": prompt_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
        "total_tokens_per_second": (prompt_tokens + output_tokens) / elapsed,
        "request_metrics_available": metrics_available,
        "ttft_p50_seconds": statistics.median(ttft) if ttft else None,
        "ttft_p95_seconds": _percentile(ttft, 0.95) if ttft else None,
        "tpot_p50_seconds": statistics.median(tpot) if tpot else None,
        "tpot_p95_seconds": _percentile(tpot, 0.95) if tpot else None,
        "corrupted_requests": corrupted_requests,
        "token_sha256": _hash_json(token_ids),
    }
    return metrics, token_ids, texts


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    """Load one vLLM engine and execute all configured batch sizes."""

    os.environ["EK_ENABLE"] = "1"
    os.environ["EK_PIPELINE_ENABLE"] = "1" if arguments.mode == "pipeline" else "0"
    # Keep sync and pipeline measurements on the same runner. The reviewed
    # four-stage vLLM patch targets the legacy GPU model runner, while vLLM
    # 0.25.1 may otherwise auto-select its unpatched V2 runner.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(arguments.model, trust_remote_code=True)
    samples = load_sharegpt_samples(
        arguments.dataset,
        tokenizer,
        count=max(arguments.batch_sizes),
        seed=arguments.dataset_seed,
        min_prompt_tokens=arguments.min_prompt_tokens,
        max_prompt_tokens=arguments.max_prompt_tokens,
        fixed_prompt_tokens=arguments.fixed_prompt_tokens,
    )
    manifest = [sample.manifest_entry() for sample in samples]
    llm = LLM(
        model=str(arguments.model),
        tokenizer=str(arguments.model),
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        max_model_len=arguments.max_model_len,
        max_num_seqs=max(arguments.batch_sizes),
        max_num_batched_tokens=arguments.max_num_batched_tokens
        or max(arguments.batch_sizes) * arguments.max_prompt_tokens,
        gpu_memory_utilization=arguments.gpu_memory_utilization,
        ubatch_size=4 if arguments.mode == "pipeline" else 0,
        dbo_decode_token_threshold=0,
        dbo_prefill_token_threshold=0,
        seed=arguments.generation_seed,
    )
    sampling = SamplingParams(
        temperature=0,
        seed=arguments.generation_seed,
        ignore_eos=True,
        min_tokens=arguments.output_tokens,
        max_tokens=arguments.output_tokens,
    )
    batches = []
    deterministic = True
    for batch_size in arguments.batch_sizes:
        selected = samples[:batch_size]
        inputs = [
            {"prompt_token_ids": list(sample.prompt_token_ids)} for sample in selected
        ]
        for _ in range(arguments.warmup_runs):
            _measure(llm, inputs, sampling, cuda_profiler_range=False)
        runs = []
        reference_tokens: list[list[int]] | None = None
        reference_texts: list[str] | None = None
        for run_index in range(arguments.runs):
            metrics, token_ids, texts = _measure(
                llm,
                inputs,
                sampling,
                cuda_profiler_range=arguments.cuda_profiler_range,
            )
            if any(len(tokens) != arguments.output_tokens for tokens in token_ids):
                raise RuntimeError(
                    "one or more requests did not generate the fixed token count"
                )
            if metrics["corrupted_requests"] not in (None, 0):
                raise RuntimeError("vLLM reported corrupted output logits")
            if reference_tokens is None:
                reference_tokens = token_ids
                reference_texts = texts
            elif token_ids != reference_tokens:
                deterministic = False
            runs.append({"run_index": run_index, **metrics})
        assert reference_tokens is not None and reference_texts is not None
        batches.append(
            {
                "batch_size": batch_size,
                "sample_ids": [sample.dataset_id for sample in selected],
                "runs": runs,
                "summary": _summary(runs),
                "reference_token_ids": reference_tokens,
                "reference_texts": reference_texts,
                "deterministic": all(
                    run["token_sha256"] == runs[0]["token_sha256"] for run in runs
                ),
            }
        )
    report = {
        "schema_version": 1,
        "mode": arguments.mode,
        "model": str(arguments.model.resolve()),
        "dataset": str(arguments.dataset.resolve()),
        "dataset_seed": arguments.dataset_seed,
        "generation_seed": arguments.generation_seed,
        "min_prompt_tokens": arguments.min_prompt_tokens,
        "max_prompt_tokens": arguments.max_prompt_tokens,
        "fixed_prompt_tokens": arguments.fixed_prompt_tokens,
        "max_num_batched_tokens": arguments.max_num_batched_tokens
        or max(arguments.batch_sizes) * arguments.max_prompt_tokens,
        "output_tokens": arguments.output_tokens,
        "warmup_runs": arguments.warmup_runs,
        "measured_runs": arguments.runs,
        "sample_manifest_sha256": _hash_json(manifest),
        "sample_manifest": manifest,
        "batches": batches,
        "deterministic": deterministic
        and all(batch["deterministic"] for batch in batches),
    }
    # vLLM does not expose a stable public engine shutdown hook for this
    # offline path. Close Expert Kit while the asyncio executor is still live;
    # the registered atexit callback then becomes an idempotent no-op.
    from expertkit_vllm.experts.remote_moe import close_clients

    close_clients()
    return report


def compare_reports(
    sync: dict[str, object], pipeline: dict[str, object]
) -> dict[str, object]:
    """Compare two benchmark reports request-by-request and token-by-token."""

    if sync.get("mode") != "sync" or pipeline.get("mode") != "pipeline":
        raise ValueError("comparison requires sync and pipeline reports")
    if sync.get("sample_manifest_sha256") != pipeline.get("sample_manifest_sha256"):
        raise ValueError("reports used different ShareGPT sample manifests")
    sync_batches = {batch["batch_size"]: batch for batch in sync["batches"]}
    pipeline_batches = {batch["batch_size"]: batch for batch in pipeline["batches"]}
    if sync_batches.keys() != pipeline_batches.keys():
        raise ValueError("reports contain different batch sizes")
    comparisons = []
    first_mismatch: dict[str, object] | None = None
    for batch_size in sorted(sync_batches):
        expected = sync_batches[batch_size]["reference_token_ids"]
        actual = pipeline_batches[batch_size]["reference_token_ids"]
        equal = expected == actual
        if not equal and first_mismatch is None:
            for request_index, (left, right) in enumerate(
                zip(expected, actual, strict=True)
            ):
                if left == right:
                    continue
                token_index = next(
                    (
                        index
                        for index, pair in enumerate(zip(left, right, strict=False))
                        if pair[0] != pair[1]
                    ),
                    min(len(left), len(right)),
                )
                first_mismatch = {
                    "batch_size": batch_size,
                    "request_index": request_index,
                    "dataset_id": sync_batches[batch_size]["sample_ids"][request_index],
                    "token_index": token_index,
                    "sync_token": left[token_index]
                    if token_index < len(left)
                    else None,
                    "pipeline_token": right[token_index]
                    if token_index < len(right)
                    else None,
                }
                break
        sync_output = sync_batches[batch_size]["summary"][
            "output_tokens_per_second_median"
        ]
        pipeline_output = pipeline_batches[batch_size]["summary"][
            "output_tokens_per_second_median"
        ]
        comparisons.append(
            {
                "batch_size": batch_size,
                "token_ids_equal": equal,
                "sync_output_tokens_per_second": sync_output,
                "pipeline_output_tokens_per_second": pipeline_output,
                "output_throughput_speedup": pipeline_output / sync_output,
            }
        )
    passed = (
        bool(sync.get("deterministic"))
        and bool(pipeline.get("deterministic"))
        and first_mismatch is None
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "sync_deterministic": sync.get("deterministic"),
        "pipeline_deterministic": pipeline.get("deterministic"),
        "first_mismatch": first_mismatch,
        "batches": comparisons,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m expertkit_vllm.benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--mode", choices=("sync", "pipeline"), required=True)
    run.add_argument(
        "--batch-sizes", nargs="+", type=_positive_int, default=[4, 8, 16, 32, 64]
    )
    run.add_argument("--dataset-seed", type=int, default=42)
    run.add_argument("--generation-seed", type=int, default=0)
    run.add_argument("--min-prompt-tokens", type=_positive_int, default=4)
    run.add_argument("--max-prompt-tokens", type=_positive_int, default=512)
    run.add_argument(
        "--fixed-prompt-tokens",
        type=_positive_int,
        help="truncate every selected prompt to this exact token count",
    )
    run.add_argument("--output-tokens", type=_positive_int, default=32)
    run.add_argument("--max-model-len", type=_positive_int, default=1024)
    run.add_argument(
        "--max-num-batched-tokens",
        type=_positive_int,
        help="override vLLM's scheduler token budget",
    )
    run.add_argument("--warmup-runs", type=int, default=1)
    run.add_argument("--runs", type=_positive_int, default=5)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    run.add_argument("--cuda-profiler-range", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--sync", type=Path, required=True)
    compare.add_argument("--pipeline", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        if arguments.warmup_runs < 0:
            raise ValueError("warmup-runs must be nonnegative")
        if not 0 < arguments.gpu_memory_utilization <= 1:
            raise ValueError("gpu-memory-utilization must be in (0, 1]")
        if (
            arguments.fixed_prompt_tokens is not None
            and arguments.fixed_prompt_tokens > arguments.max_prompt_tokens
        ):
            raise ValueError("fixed-prompt-tokens cannot exceed max-prompt-tokens")
        report = run_benchmark(arguments)
        exit_code = 0 if report["deterministic"] else 2
    else:
        with arguments.sync.open(encoding="utf-8") as source:
            sync = json.load(source)
        with arguments.pipeline.open(encoding="utf-8") as source:
            pipeline = json.load(source)
        report = compare_reports(sync, pipeline)
        exit_code = 0 if report["passed"] else 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
