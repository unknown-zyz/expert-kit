#!/usr/bin/env python3
"""Run one Qwen3 benchmark mode with an isolated Python Worker service chain."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

PORTS = (55432, 6543, 5001, 5002, 51061, 51062)


def _port_open(port: int) -> bool:
    with socket.socket() as candidate:
        candidate.settimeout(0.2)
        return candidate.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(
    port: int, process: subprocess.Popen[bytes] | None, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"service exited before port {port} became ready")
        if _port_open(port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"port {port} did not become ready within {timeout} seconds")


def _start(
    name: str, command: list[str], env: dict[str, str], logs: Path
) -> tuple[subprocess.Popen[bytes], IO[bytes]]:
    log = (logs / f"{name}.log").open("wb")
    process = subprocess.Popen(
        command,
        cwd=env["EK_REPOSITORY"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sync", "pipeline"), required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--fixed-prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--detailed-profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(
            "/home/zhangyz/expert-kit/output/qwen3-30b-a3b-pipeline-dev-py"
        ),
    )
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
    args = parser.parse_args()
    repository = args.repository.resolve()
    runtime = args.runtime_root.resolve()
    config = repository / "ek-integration/expertkit_vllm/benchmarks/qwen3-30b-a3b"
    python = Path("/home/zhangyz/expert-kit/.venv/bin/python")
    worker = repository / "ek-worker/.venv/bin/ek-worker"
    binary = repository / "target/release/ek-cli"
    for required in (
        args.model,
        args.dataset,
        config / "controller.yaml",
        config / "worker.yaml",
        runtime / "postgres/PG_VERSION",
        binary,
        worker,
        python,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    conflicts = [port for port in PORTS if _port_open(port)]
    if conflicts:
        raise RuntimeError(f"benchmark requires free isolated ports: {conflicts}")

    kind = "nsys" if args.detailed_profile else "perf"
    logs = runtime / "logs" / f"{args.mode}-{kind}"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("EK_CONFIG", None)
    env.update(
        {
            "EK_REPOSITORY": str(repository),
            "EK_NSYS_PROFILE": "1" if args.detailed_profile else "0",
            "EK_ENABLE": "1",
            "EK_PIPELINE_ENABLE": "1" if args.mode == "pipeline" else "0",
            "EK_ADDR": "127.0.0.1:5002",
            "EK_INSTANCE_ID": "1",
            "EK_CLIENT_TIMEOUT": "900",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "PYTHONPATH": os.pathsep.join(
                str(repository / relative)
                for relative in (
                    "ek-proto/src",
                    "ek-transport/src",
                    "ek-integration/expertkit_vllm",
                    "ek-worker/src",
                )
            ),
        }
    )
    processes: list[tuple[str, subprocess.Popen[bytes], IO[bytes]]] = []
    postgres_started = False
    try:
        subprocess.run(
            [
                "pg_ctl",
                "-D",
                str(runtime / "postgres"),
                "-l",
                str(logs / "postgres.log"),
                "-o",
                f"-p 55432 -h 127.0.0.1 -k {runtime / 'postgres'}",
                "start",
            ],
            check=True,
            env=env,
        )
        postgres_started = True
        _wait_port(55432, None, 30)
        process, log = _start(
            "weight-server",
            [
                str(binary),
                "--config",
                str(config / "controller.yaml"),
                "weight-server",
                "--model",
                str(args.model.resolve()),
            ],
            env,
            logs,
        )
        processes.append(("weight-server", process, log))
        _wait_port(6543, process, 120)
        process, log = _start(
            "controller",
            [str(binary), "--config", str(config / "controller.yaml"), "controller"],
            env,
            logs,
        )
        processes.append(("controller", process, log))
        _wait_port(5001, process, 60)
        _wait_port(5002, process, 60)
        process, log = _start(
            "worker", [str(worker), "--config", str(config / "worker.yaml")], env, logs
        )
        processes.append(("worker", process, log))
        _wait_port(51061, process, 3600)
        subprocess.run(
            [
                str(python),
                str(repository / "ek-integration/expertkit_vllm/scripts/check_topology.py"),
                "--num-layers",
                "48",
                "--experts-per-layer",
                "128",
                "--timeout",
                "3600",
            ],
            check=True,
            env=env,
            cwd=repository,
            timeout=3700,
        )
        command = [
            str(python),
            str(repository / "ek-integration/expertkit_vllm/scripts/nsys_benchmark.py")
            if args.detailed_profile
            else "-m",
        ]
        if not args.detailed_profile:
            command.append("expertkit_vllm.benchmark")
        command.extend(
            [
                "run",
                "--model",
                str(args.model.resolve()),
                "--dataset",
                str(args.dataset.resolve()),
                "--mode",
                args.mode,
                "--batch-sizes",
                *(str(value) for value in args.batch_sizes),
                "--min-prompt-tokens",
                str(args.fixed_prompt_tokens),
                "--max-prompt-tokens",
                "1024",
                "--fixed-prompt-tokens",
                str(args.fixed_prompt_tokens),
                "--output-tokens",
                str(args.output_tokens),
                "--max-model-len",
                str(args.fixed_prompt_tokens + args.output_tokens + 64),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--warmup-runs",
                str(args.warmup_runs),
                "--runs",
                str(args.runs),
                "--output",
                str(args.output.resolve()),
            ]
        )
        if args.detailed_profile:
            command.append("--cuda-profiler-range")
        benchmark_log = logs / "benchmark.log"
        print(f"benchmark output: {benchmark_log}", flush=True)
        with benchmark_log.open("wb") as log_output:
            completed = subprocess.run(
                command,
                env=env,
                cwd=repository,
                timeout=10800,
                stdout=log_output,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode not in {0, 2}:
            raise subprocess.CalledProcessError(completed.returncode, command)
        return completed.returncode
    finally:
        for _name, process, _log in reversed(processes):
            _stop(process)
        for _name, _process, log in processes:
            log.close()
        if postgres_started:
            subprocess.run(
                ["pg_ctl", "-D", str(runtime / "postgres"), "stop", "-m", "fast"],
                check=False,
                env=env,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
