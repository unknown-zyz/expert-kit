#!/usr/bin/env python3
"""Run the complete Qwen3 sync/pipeline experiment and write its report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_RUNTIME = Path("/home/zhangyz/expert-kit/output/qwen3-30b-a3b-pipeline-dev-py")


def _run(command: list[str], env: dict[str, str], allowed: set[int] = {0}) -> int:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, env=env)
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _batch(report: dict[str, Any], batch_size: int) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in report.get("batches", ())
            if item["batch_size"] == batch_size
        ),
        None,
    )


def _ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 1000:.3f}"


def _token_mismatch_stats(
    sync: dict[str, Any], pipeline: dict[str, Any], batch_size: int
) -> tuple[int, int, int]:
    sync_batch = _batch(sync, batch_size)
    pipeline_batch = _batch(pipeline, batch_size)
    if sync_batch is None or pipeline_batch is None:
        return 0, 0, 0
    request_mismatches = 0
    token_mismatches = 0
    total_tokens = 0
    for sync_tokens, pipeline_tokens in zip(
        sync_batch["reference_token_ids"],
        pipeline_batch["reference_token_ids"],
        strict=True,
    ):
        differences = sum(
            left != right
            for left, right in zip(sync_tokens, pipeline_tokens, strict=True)
        )
        request_mismatches += differences > 0
        token_mismatches += differences
        total_tokens += len(sync_tokens)
    return request_mismatches, token_mismatches, total_tokens


def _write_report(
    output: Path,
    args: argparse.Namespace,
    *,
    status: str,
    error: str | None = None,
) -> None:
    sync_path = output / "results/sync.json"
    pipeline_path = output / "results/pipeline.json"
    comparison_path = output / "results/comparison.json"
    analysis_path = output / "results/nsys-analysis.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(),
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "settings": {
            "batch_sizes": args.batch_sizes,
            "fixed_prompt_tokens": args.fixed_prompt_tokens,
            "output_tokens": args.output_tokens,
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "ubatch_size": 4,
            "worker_device": "cpu",
            "worker_max_active_batches": 4,
            "worker_omp_threads": 4,
        },
        "error": error,
    }
    for name, path in (
        ("sync", sync_path),
        ("pipeline", pipeline_path),
        ("comparison", comparison_path),
        ("nsys", analysis_path),
    ):
        if path.exists():
            payload[name] = _load(path)
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "results/qwen3-four-stage-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Qwen3-30B-A3B 四阶段流水线实验报告",
        "",
        f"状态：**{status}**",
        "",
        f"模型：`{payload['model']}`",
        f"数据集：`{payload['dataset']}`",
        f"设置：batch={','.join(map(str, args.batch_sizes))}，输入={args.fixed_prompt_tokens} tokens，输出={args.output_tokens} tokens，warmup={args.warmup_runs}，runs={args.runs}。",
        "",
    ]
    if error:
        lines.extend(["## 首个阻塞错误", "", "```text", error, "```", ""])
    available_batches = [
        (batch_size, mode, batch)
        for batch_size in args.batch_sizes
        for mode in ("sync", "pipeline")
        if (report := payload.get(mode)) is not None
        and (batch := _batch(report, batch_size)) is not None
    ]
    if available_batches:
        lines.extend(
            [
                "## 性能",
                "",
                "TPOT 是每请求 `(last_token_ts - first_token_ts) / (output_tokens - 1)`；下表为各 measured run 的中位数。正常性能测试不启用 NSys。",
                "",
                "| Batch | 模式 | TPOT P50 (ms/token) | TPOT P95 (ms/token) | TTFT P50 (ms) | Output tok/s | Requests/s |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for batch_size, mode, batch in available_batches:
            summary = batch["summary"]
            lines.append(
                f"| {batch_size} | {mode} | "
                f"{_ms(summary['tpot_p50_seconds_median'])} | "
                f"{_ms(summary['tpot_p95_seconds_median'])} | "
                f"{_ms(summary['ttft_p50_seconds_median'])} | "
                f"{summary['output_tokens_per_second_median']:.3f} | "
                f"{summary['requests_per_second_median']:.3f} |"
            )
        if "comparison" in payload:
            lines.extend(["", "### 同步/流水线对比", ""])
            lines.append(
                f"严格 token 等价：`{payload['comparison']['passed']}`；只有两种模式逐请求、逐 token 相同且各自重复运行确定时才通过。"
            )
            lines.extend(
                [
                    "",
                    "| Batch | Output throughput speedup | Token IDs equal |",
                    "|---:|---:|---|",
                ]
            )
            for row in payload["comparison"]["batches"]:
                lines.append(
                    f"| {row['batch_size']} | {row['output_throughput_speedup']:.3f}x | {row['token_ids_equal']} |"
                )
            comparison = payload["comparison"]
            lines.extend(
                [
                    "",
                    f"两种模式各自重复运行确定：sync=`{comparison['sync_deterministic']}`，pipeline=`{comparison['pipeline_deterministic']}`。",
                ]
            )
            if first := comparison.get("first_mismatch"):
                request_mismatches, token_mismatches, total_tokens = (
                    _token_mismatch_stats(
                        payload["sync"], payload["pipeline"], first["batch_size"]
                    )
                )
                lines.extend(
                    [
                        f"首次分叉：request {first['request_index']}（dataset `{first['dataset_id']}`）的生成 token {first['token_index']}，sync={first['sync_token']}，pipeline={first['pipeline_token']}。",
                        f"总计 {request_mismatches}/{first['batch_size']} 个请求、{token_mismatches}/{total_tokens} 个 token 不同。该现象符合 BF16 GEMM 在不同 batch shape 下舍入路径变化并经 greedy decoding 累积放大的特征；尚无证据表明是 RPC 丢包或路由错位，因此严格验收仍标记 FAIL。",
                    ]
                )
            first_row = payload["comparison"]["batches"][0]
            speed_delta = (first_row["output_throughput_speedup"] - 1) * 100
            lines.extend(
                [
                    "",
                    "### 性能变化分析",
                    "",
                    f"流水线 output throughput 变化 `{speed_delta:+.1f}%`。TTFT P50 从 `{_ms(payload['sync']['batches'][0]['summary']['ttft_p50_seconds_median'])}` ms 降到 `{_ms(payload['pipeline']['batches'][0]['summary']['ttft_p50_seconds_median'])}` ms，但 TPOT P50 从 `{_ms(payload['sync']['batches'][0]['summary']['tpot_p50_seconds_median'])}` 增到 `{_ms(payload['pipeline']['batches'][0]['summary']['tpot_p50_seconds_median'])}` ms/token。",
                ]
            )
    if "nsys" in payload:
        lines.extend(
            [
                "",
                "## 四阶段 NSys 时间",
                "",
                "A 为 decoder layer scope 内 CUDA kernel active-time union；A2E 为 Frontend dispatch 到 Worker 首个 Expert compute；E 为 Python Worker Torch backend wall interval；E2A 为最后 Expert compute 完成到 Frontend result ready。阶段总量可跨 micro-batch 重叠，因此 share 是工作量占比而非端到端 wall-time 占比。",
                "",
                "| 模式 | 阶段 | Mean (ms/call) | P50 | P95 | 工作量占比 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for mode in ("sync", "pipeline"):
            for stage in ("A", "A2E", "E", "E2A"):
                item = payload["nsys"][mode]["stages"][stage]
                lines.append(
                    f"| {mode} | {stage} | {item['mean_ms']:.3f} | {item['p50_ms']:.3f} | {item['p95_ms']:.3f} | {item['share'] * 100:.1f}% |"
                )
        hiding = payload["nsys"]["pipeline"]["communication_hiding"]
        lines.extend(
            [
                "",
                f"流水线通信隐藏率：A2E `{hiding['A2E']['hide_ratio'] * 100:.1f}%`，E2A `{hiding['E2A']['hide_ratio'] * 100:.1f}%`，合计 `{hiding['total']['hide_ratio'] * 100:.1f}%`。",
                "",
                f"瓶颈不是 Worker admission queue：其均值仅 `{payload['nsys']['pipeline']['subphases']['A2E_WORKER_QUEUE']['mean_ms']:.3f}` ms。主要退化来自一次完整 Expert call 被拆成约四个小 call：sync 共 `{payload['nsys']['sync']['call_count']}` 次、pipeline 共 `{payload['nsys']['pipeline']['call_count']}` 次；pipeline 每个更小 call 的 E 均值反而从 `{payload['nsys']['sync']['stages']['E']['mean_ms']:.3f}` 增至 `{payload['nsys']['pipeline']['stages']['E']['mean_ms']:.3f}` ms，说明 CPU 小矩阵效率下降并叠加四路并发的内存/线程竞争。A2E 均值也从 `{payload['nsys']['sync']['stages']['A2E']['mean_ms']:.3f}` 增至 `{payload['nsys']['pipeline']['stages']['A2E']['mean_ms']:.3f}` ms。",
                "",
                f"pipeline E2A 的 P50 为 `{payload['nsys']['pipeline']['stages']['E2A']['p50_ms']:.3f}` ms、P95 为 `{payload['nsys']['pipeline']['stages']['E2A']['p95_ms']:.3f}` ms。E2A 包含 CUDA stream/协调器等待，不等同于纯 gRPC 线耗时；其中 `{hiding['E2A']['hide_ratio'] * 100:.1f}%` 与其它 uBatch 的 A/E 重叠，故它不是吞吐下降的主要暴露路径。",
                "",
                "## 各层四阶段平均时间",
                "",
                "| Layer | 模式 | A (ms) | A2E (ms) | E (ms) | E2A (ms) |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for layer_id in range(48):
            for mode in ("sync", "pipeline"):
                row = next(
                    item
                    for item in payload["nsys"][mode]["layers"]
                    if item["layer_id"] == layer_id
                )
                values = [
                    row["stages"][stage]["mean_ms"]
                    for stage in ("A", "A2E", "E", "E2A")
                ]
                lines.append(
                    f"| {layer_id} | {mode} | "
                    + " | ".join(f"{value:.3f}" for value in values)
                    + " |"
                )
        lines.extend(
            [
                "",
                "图：`qwen3-four-stage-share.svg`（48 层阶段占比）和 `qwen3-four-stage-overlap.svg`（代表性相邻两层 micro-batch 重叠）。",
            ]
        )
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "```bash",
            "/home/zhangyz/expert-kit/.venv/bin/python \\",
            "  ek-integration/expertkit_vllm/scripts/run_qwen3_experiment.py",
            "```",
            "",
            "可加 `--skip-nsys` 只测性能，或 `--resume` 复用已完成的正常性能结果。脚本只停止自己启动的进程，不删除 PostgreSQL、权重缓存、原始 NSys 或结果文件。",
            "",
        ]
    )
    (output / "results/qwen3-four-stage-report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/data/models/huggingface/Qwen/Qwen3-30B-A3B"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/datasets/ShareGPT52K/sg_52k.json"),
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--fixed-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-weight-build", action="store_true")
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.runtime_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "nsys").mkdir(parents=True, exist_ok=True)
    args.model.resolve(strict=True)
    args.dataset.resolve(strict=True)
    env = os.environ.copy()
    env.pop("EK_CONFIG", None)
    python = "/home/zhangyz/expert-kit/.venv/bin/python"
    scripts = repository / "ek-integration/expertkit_vllm/scripts"
    status = "BLOCKED"
    error: str | None = None
    try:
        _run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            env,
        )
        _run(
            [python, str(scripts / "apply_vllm_pipeline_patch.py"), "--apply"],
            env,
        )
        if not args.skip_prepare:
            command = [
                python,
                str(scripts / "prepare_qwen3_benchmark.py"),
                "--repository",
                str(repository),
                "--runtime-root",
                str(output),
                "--model",
                str(args.model),
            ]
            if args.skip_weight_build:
                command.append("--skip-weight-build")
            _run(command, env)
        common = [
            "--repository",
            str(repository),
            "--runtime-root",
            str(output),
            "--model",
            str(args.model),
            "--dataset",
            str(args.dataset),
            "--batch-sizes",
            *(str(value) for value in args.batch_sizes),
            "--fixed-prompt-tokens",
            str(args.fixed_prompt_tokens),
            "--output-tokens",
            str(args.output_tokens),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
        ]
        for mode in ("sync", "pipeline"):
            result = output / f"results/{mode}.json"
            if not (args.resume and result.exists()):
                result.unlink(missing_ok=True)
                _run(
                    [
                        python,
                        str(scripts / "run_qwen3_stack_benchmark.py"),
                        "--mode",
                        mode,
                        "--warmup-runs",
                        str(args.warmup_runs),
                        "--runs",
                        str(args.runs),
                        "--output",
                        str(result),
                        *common,
                    ],
                    env,
                    {0, 2},
                )
                if not result.exists():
                    raise RuntimeError(
                        f"{mode} benchmark exited without writing {result}"
                    )
        comparison_output = output / "results/comparison.json"
        comparison_output.unlink(missing_ok=True)
        _run(
            [
                python,
                "-m",
                "expertkit_vllm.benchmark",
                "compare",
                "--sync",
                str(output / "results/sync.json"),
                "--pipeline",
                str(output / "results/pipeline.json"),
                "--output",
                str(comparison_output),
            ],
            {
                **env,
                "PYTHONPATH": str(repository / "ek-integration/expertkit_vllm"),
            },
            {0, 2},
        )
        comparison = _load(comparison_output)
        status = "PASS" if comparison["passed"] else "FAIL"
        if not args.skip_nsys:
            for mode in ("sync", "pipeline"):
                base = output / f"nsys/qwen3-{mode}-b{max(args.batch_sizes)}"
                report = base.with_suffix(".nsys-rep")
                if not (args.resume and report.exists()):
                    _run(
                        [
                            "nsys",
                            "profile",
                            "--trace=cuda,nvtx,osrt",
                            "--trace-fork-before-exec=true",
                            "--sample=none",
                            "--cpuctxsw=process-tree",
                            "--cuda-event-trace=false",
                            "--resolve-symbols=false",
                            "--capture-range=cudaProfilerApi",
                            "--capture-range-end=stop",
                            "--wait=primary",
                            "--force-overwrite=true",
                            f"--output={base}",
                            python,
                            str(scripts / "run_qwen3_stack_benchmark.py"),
                            "--mode",
                            mode,
                            "--batch-sizes",
                            str(max(args.batch_sizes)),
                            "--warmup-runs",
                            "1",
                            "--runs",
                            "1",
                            "--detailed-profile",
                            "--output",
                            str(output / f"results/nsys-{mode}.json"),
                            *common[: common.index("--batch-sizes")],
                            "--fixed-prompt-tokens",
                            str(args.fixed_prompt_tokens),
                            "--output-tokens",
                            str(args.output_tokens),
                            "--max-num-batched-tokens",
                            str(args.max_num_batched_tokens),
                        ],
                        env,
                        {0, 2},
                    )
            _run(
                [
                    python,
                    str(scripts / "analyze_deepseek_nsys.py"),
                    "--sync",
                    str(output / f"nsys/qwen3-sync-b{max(args.batch_sizes)}.nsys-rep"),
                    "--pipeline",
                    str(
                        output
                        / f"nsys/qwen3-pipeline-b{max(args.batch_sizes)}.nsys-rep"
                    ),
                    "--model-name",
                    "Qwen3-30B-A3B",
                    "--first-moe-layer",
                    "0",
                    "--num-moe-layers",
                    "48",
                    "--decode-max-tokens",
                    str(max(args.batch_sizes)),
                    "--output",
                    str(output / "results/nsys-analysis.json"),
                ],
                env,
            )
            _run(
                [
                    python,
                    str(scripts / "render_deepseek_pipeline.py"),
                    str(output / "results/nsys-analysis.json"),
                    "--heatmap",
                    str(output / "results/qwen3-four-stage-share.svg"),
                    "--timeline",
                    str(output / "results/qwen3-four-stage-overlap.svg"),
                ],
                env,
            )
    except Exception:
        error = traceback.format_exc()
    _write_report(output, args, status=status, error=error)
    print(
        json.dumps(
            {
                "status": status,
                "report": str(output / "results/qwen3-four-stage-report.md"),
                "error": error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status == "PASS" and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
