#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


SERVICES = (
    ("weight-server", 6543),
    ("controller-intra", 5001),
    ("controller-inter", 5002),
    ("worker", 51234),
)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"service exited with code {process.returncode} before port {port} opened"
            )
        if port_open(port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for port {port}")


def start_service(
    name: str,
    command: list[str],
    env: dict[str, str],
    log_dir: Path,
) -> tuple[subprocess.Popen, object]:
    log = (log_dir / f"{name}.log").open("w")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log


def stop_service(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("sync", "pipeline"), default="pipeline"
    )
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--no-profile-window",
        action="store_true",
        help="run the same isolated service setup without NVTX capture markers",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/ek-nsys-logs"))
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--client-timeout", type=int, default=600)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("/tmp/ek-nsys-workload.json"),
    )
    args = parser.parse_args()
    if args.worker_threads < 1:
        parser.error("--worker-threads must be positive")
    root = Path(__file__).resolve().parents[3]
    binary = root / "target/release/ek-cli"
    python = root / ".venv/bin/python"
    benchmark = (
        root
        / "ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py"
    )
    config = args.config.resolve(strict=True)
    model = args.model.resolve(strict=True)
    if not binary.is_file():
        raise RuntimeError(f"release binary is missing: {binary}")
    conflicts = [name for name, port in SERVICES if port_open(port)]
    if conflicts:
        raise RuntimeError(
            "profile requires exclusive service ports; already listening: "
            + ", ".join(conflicts)
        )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "EK_CONFIG": str(config),
            "QWEN3_30B_A3B_ROOT": str(model),
            "EK_NSYS_PROFILE": "0" if args.no_profile_window else "1",
            "EK_WORKER_PARALLEL": "1",
            "EK_WORKER_THREADS": str(args.worker_threads),
            "EK_ENABLE": "1",
            "EK_MODE": "expert_mode",
            "EK_ADDR": "localhost:5002",
            "EK_MODEL_NAME": "qwen3-30b-a3b",
            "EK_CLIENT_TIMEOUT": str(args.client_timeout),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    processes: list[tuple[str, subprocess.Popen, object]] = []
    try:
        process, log = start_service(
            "weight-server",
            [
                str(binary),
                "--config",
                str(config),
                "weight-server",
                "--model",
                str(model),
            ],
            env,
            args.log_dir,
        )
        processes.append(("weight-server", process, log))
        wait_for_port(6543, process, 120)

        process, log = start_service(
            "controller",
            [str(binary), "--config", str(config), "controller"],
            env,
            args.log_dir,
        )
        processes.append(("controller", process, log))
        wait_for_port(5001, process, 60)
        wait_for_port(5002, process, 60)

        process, log = start_service(
            "worker",
            [str(binary), "--config", str(config), "worker"],
            env,
            args.log_dir,
        )
        processes.append(("worker", process, log))
        wait_for_port(51234, process, 180)
        time.sleep(3)

        command = [
            str(python),
            str(benchmark),
            "--child-mode",
            args.mode,
            "--child-output",
            str(args.benchmark_output),
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
            "--max-tokens",
            str(args.max_tokens),
            "--max-model-len",
            str(args.max_model_len),
        ]
        if args.prompts_file is not None:
            command.extend(("--prompts-file", str(args.prompts_file.resolve(strict=True))))
        if not args.no_profile_window:
            command.append("--nsys-capture")
        subprocess.run(command, env=env, check=True, timeout=900)
    finally:
        for _name, process, _log in reversed(processes):
            stop_service(process)
        for _name, _process, log in processes:
            log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
