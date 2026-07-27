#!/usr/bin/env python3
"""Run the Qwen3 sync/pipeline benchmark and produce one self-contained report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = Path("/data/models/huggingface/Qwen/Qwen3-30B-A3B")
DEFAULT_DATASET = Path("/data/datasets/ShareGPT52K/sg_52k.json")


def prompt_from_record(record: dict) -> str | None:
    turns = record.get("conversations") or record.get("messages") or []
    for turn in turns:
        role = str(turn.get("from", turn.get("role", ""))).lower()
        if role in {"human", "user"}:
            value = turn.get("value", turn.get("content", ""))
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def create_manifest(
    dataset: Path,
    output: Path,
    batch_size: int,
    seed: int,
    model: Path,
    max_prompt_tokens: int,
    input_tokens: int,
) -> None:
    payload = json.loads(dataset.resolve(strict=True).read_text())
    records = payload if isinstance(payload, list) else payload.get("data", [])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=True
    )
    candidates = []
    for index, record in enumerate(records):
        prompt = prompt_from_record(record)
        if prompt:
            token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            if input_tokens <= token_count <= max_prompt_tokens:
                token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                candidates.append(
                    {
                        "id": str(record.get("id", index)),
                        "prompt": prompt,
                        "prompt_tokens": token_count,
                        "prompt_token_ids": token_ids[:input_tokens],
                    }
                )
    if len(candidates) < batch_size:
        raise RuntimeError(
            f"dataset contains only {len(candidates)} usable prompts; "
            f"batch size is {batch_size}"
        )
    selected = random.Random(seed).sample(candidates, batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "seed": seed,
                "batch_size": batch_size,
                "prompts": selected,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.resolve(strict=True).read_bytes()).hexdigest()


def run_mode(
    *,
    root: Path,
    args: argparse.Namespace,
    mode: str,
    profile: bool,
    manifest: Path,
    output_dir: Path,
) -> Path:
    launcher = root / "ek-integration/expertkit_vllm/scripts/run_nsys_pipeline_profile.py"
    python = root / ".venv/bin/python"
    benchmark_output = output_dir / f"{mode}-benchmark.json"
    log_dir = output_dir / "logs" / f"{mode}-{'nsys' if profile else 'perf'}"
    env = dict(os.environ)
    env["QWEN3_30B_A3B_ROOT"] = str(args.model.resolve(strict=True))
    env["EK_CONFIG"] = str(args.config.resolve(strict=True))

    launcher_args = [
        str(python),
        str(launcher),
        "--config",
        str(args.config.resolve(strict=True)),
        "--model",
        str(args.model.resolve(strict=True)),
        "--mode",
        mode,
        "--worker-threads",
        str(args.worker_threads),
        "--warmup",
        str(args.warmup if not profile else 1),
        "--repetitions",
        str(args.repetitions if not profile else 1),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--client-timeout",
        str(args.client_timeout),
        "--prompts-file",
        str(manifest),
        "--log-dir",
        str(log_dir),
        "--benchmark-output",
        str(benchmark_output),
    ]
    if not profile:
        launcher_args.append("--no-profile-window")
        run_command(launcher_args, env)
        return benchmark_output

    report_base = output_dir / mode
    nsys_command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--trace-fork-before-exec=true",
        "--sample=none",
        "--cpuctxsw=process-tree",
        "--cuda-event-trace=false",
        "--resolve-symbols=false",
        "--wait=primary",
        "--force-overwrite=true",
        f"--output={report_base}",
        *launcher_args,
    ]
    run_command(nsys_command, env)
    return report_base.with_suffix(".nsys-rep")


def run_analyzer(root: Path, report: Path, output: Path) -> None:
    analyzer = root / "ek-integration/expertkit_vllm/scripts/analyze_nsys_pipeline.py"
    run_command([sys.executable, str(analyzer), str(report), str(output)], dict(os.environ))


def summarize_stage(analysis: dict) -> dict:
    calls = analysis.get("calls", [])
    result = {}
    for stage in ("A", "A2E", "E", "E2A"):
        durations = []
        for call in calls:
            intervals = call.get("A_attributed", [call["A_envelope"]]) if stage == "A" else call[stage]
            durations.append(sum(end - start for start, end in intervals) / 1e6)
        result[stage] = {
            "count": len(durations),
            "mean_ms": sum(durations) / len(durations) if durations else 0.0,
            "total_ms": sum(durations),
        }
    return result


def strict_token_equivalence(sync: dict, pipeline: dict) -> bool:
    reference = sync["runs"][0]["outputs"]
    return all(
        expected["token_ids"] == actual["token_ids"]
        for run in pipeline["runs"]
        for expected, actual in zip(reference, run["outputs"], strict=True)
    )


def write_report(args: argparse.Namespace, output: Path, manifest: Path, results: dict, analyses: dict) -> None:
    sync_summary = results["sync"]["summary"]
    pipeline_summary = results["pipeline"]["summary"]

    def metrics(mode: str, summary: dict) -> dict:
        latency = float(summary["latency_seconds_median"])
        tokens = int(results[mode]["runs"][0]["total_output_tokens"])
        return {
            "latency_median_s": latency,
            "throughput_median_tok_s": float(summary["tokens_per_second_median"]),
            "batch_average_tpot_ms": latency * 1000.0 / max(tokens, 1),
        }

    payload = {
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "manifest": str(manifest),
        "settings": {
            "model": str(args.model),
            "config": str(args.config),
            "dataset": str(args.dataset),
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "input_tokens": args.input_tokens,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "worker_threads": args.worker_threads,
        },
        "results": results,
        "metrics": {
            "sync": metrics("sync", sync_summary),
            "pipeline": metrics("pipeline", pipeline_summary),
        },
        "stages": {mode: summarize_stage(analysis) for mode, analysis in analyses.items()},
        "comparison": {
            "strict_token_equivalence": strict_token_equivalence(
                results["sync"], results["pipeline"]
            )
        },
        "nsys": analyses,
    }
    json_output = output / "qwen3-four-stage-report.json"
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        "# Qwen3-30B-A3B 四阶段流水线性能报告",
        "",
        f"模型：`{payload['model']}`",
        f"数据集：`{payload['dataset']}`，batch={args.batch_size}",
        "",
        "> TPOT 定义为 batch 测量窗口总耗时 / 总生成 token 数，包含 prefill；不是纯 decode TPOT。",
        "",
        "## 性能指标",
        "",
        "| 模式 | latency median (s) | throughput (tok/s) | batch-average TPOT (ms/token) |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("sync", "pipeline"):
        item = payload["metrics"][mode]
        lines.append(f"| {mode} | {item['latency_median_s']:.3f} | {item['throughput_median_tok_s']:.3f} | {item['batch_average_tpot_ms']:.3f} |")
    lines.extend(["", "## 阶段时间", "", "| 模式 | Attention | A2E | Expert | E2A |", "|---|---:|---:|---:|---:|"])
    for mode in ("sync", "pipeline"):
        stages = payload["stages"][mode]
        lines.append("| " + mode + " | " + " | ".join(f"{stages[s]['mean_ms']:.3f} ms" for s in ("A", "A2E", "E", "E2A")) + " |")
    lines.extend([
        "",
        "## 正确性",
        "",
        f"严格 token 等价：`{payload['comparison']['strict_token_equivalence']}`。如果不等价，不能将流水线结果标记为功能正确。",
        "",
        "## 各层阶段时间",
        "",
        "| 模式 | Layer | Attention | A2E | Expert | E2A |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in ("sync", "pipeline"):
        for layer, values in analyses.get(mode, {}).get("all_layer_metrics", {}).items():
            lines.append(
                f"| {mode} | {layer} | "
                + " | ".join(
                    f"{values[stage]['mean_ms']:.3f} ms"
                    for stage in ("A", "A2E", "E", "E2A")
                )
                + " |"
            )
    lines.extend(["", "## 复现命令", "", "```bash", " ".join(sys.argv), "```", ""])
    (output / "qwen3-four-stage-report.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(json_output), "metrics": payload["metrics"], "stages": payload["stages"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--client-timeout", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen3-30b-a3b-four-stage"))
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_tokens < 1 or args.repetitions < 1:
        raise SystemExit("batch-size, max-tokens and repetitions must be positive")
    args.model = args.model.resolve(strict=True)
    args.config = args.config.resolve(strict=True)
    args.dataset = args.dataset.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "prompt-manifest.json"
    create_manifest(
        args.dataset,
        manifest,
        args.batch_size,
        seed=0,
        model=args.model,
        max_prompt_tokens=args.max_model_len - args.max_tokens,
        input_tokens=args.input_tokens,
    )

    root = Path(__file__).resolve().parents[3]
    results = {}
    for mode in ("sync", "pipeline"):
        result_path = args.output_dir / f"{mode}-benchmark.json"
        reusable = False
        if args.resume and result_path.is_file():
            previous = json.loads(result_path.read_text())
            reusable = (
                previous.get("settings", {}).get("prompt_manifest_sha256")
                == manifest_digest(manifest)
            )
        if not reusable:
            result_path = run_mode(root=root, args=args, mode=mode, profile=False, manifest=manifest, output_dir=args.output_dir)
        results[mode] = json.loads(result_path.read_text())

    analyses = {}
    if not args.skip_nsys:
        for mode in ("sync", "pipeline"):
            report = run_mode(root=root, args=args, mode=mode, profile=True, manifest=manifest, output_dir=args.output_dir)
            analysis_path = args.output_dir / f"{mode}-nsys-analysis.json"
            run_analyzer(root, report, analysis_path)
            analyses[mode] = json.loads(analysis_path.read_text())
            if mode == "pipeline":
                renderer = root / "ek-integration/expertkit_vllm/scripts/render_nsys_pipeline.py"
                run_command(
                    [
                        sys.executable,
                        str(renderer),
                        str(analysis_path),
                        str(args.output_dir / "qwen3-four-stage-overlap.svg"),
                    ],
                    dict(os.environ),
                )
    else:
        analyses = {"sync": {}, "pipeline": {}}
    write_report(args, args.output_dir, manifest, results, analyses)


if __name__ == "__main__":
    main()
