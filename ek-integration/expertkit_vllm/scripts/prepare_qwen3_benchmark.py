#!/usr/bin/env python3
"""Initialize the isolated Qwen3 benchmark database and weight index."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _port_open(port: int) -> bool:
    with socket.socket() as candidate:
        candidate.settimeout(0.2)
        return candidate.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"port {port} did not become ready within {timeout} seconds")


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--skip-weight-build", action="store_true")
    args = parser.parse_args()
    repository = args.repository.resolve()
    runtime = args.runtime_root.resolve()
    model = args.model.resolve(strict=True)
    config = (
        repository
        / "ek-integration/expertkit_vllm/benchmarks/qwen3-30b-a3b"
    )
    binary = repository / "target/release/ek-cli"
    for required in (binary, config / "controller.yaml", config / "inventory.yaml"):
        if not required.exists():
            raise FileNotFoundError(required)
    if _port_open(55432):
        raise RuntimeError("isolated PostgreSQL port 55432 is already in use")

    for name in ("postgres", "weight-cache", "worker-cache", "logs", "results", "nsys"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("EK_CONFIG", None)
    postgres = runtime / "postgres"
    if not (postgres / "PG_VERSION").exists():
        _run(["initdb", "-D", str(postgres), "-A", "trust", "-U", "dev"], env)

    started = False
    weight_server: subprocess.Popen[bytes] | None = None
    weight_log = None
    try:
        _run(
            [
                "pg_ctl",
                "-D",
                str(postgres),
                "-l",
                str(runtime / "logs/prepare-postgres.log"),
                "-o",
                f"-p 55432 -h 127.0.0.1 -k {postgres}",
                "start",
            ],
            env,
        )
        started = True
        _wait_port(55432, 30)
        exists = subprocess.run(
            [
                "psql",
                "-h",
                "127.0.0.1",
                "-p",
                "55432",
                "-U",
                "dev",
                "-d",
                "postgres",
                "-Atc",
                "SELECT 1 FROM pg_database WHERE datname='dev'",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
        if exists != "1":
            _run(["createdb", "-h", "127.0.0.1", "-p", "55432", "-U", "dev", "dev"], env)
        controller = str(config / "controller.yaml")
        _run([str(binary), "--config", controller, "db", "migrate"], env)
        if not args.skip_weight_build:
            _run(
                [
                    str(binary),
                    "--config",
                    controller,
                    "weight",
                    "build",
                    "--model",
                    str(model),
                    "--cache-dir",
                    str(runtime / "weight-cache"),
                ],
                env,
            )
        _run(
            [
                str(binary),
                "--config",
                controller,
                "model",
                "upsert",
                "--name",
                "Qwen3-30B-A3B",
            ],
            env,
        )
        weight_log = (runtime / "logs/prepare-weight-server.log").open("wb")
        weight_server = subprocess.Popen(
            [
                str(binary),
                "--config",
                controller,
                "weight-server",
                "--model",
                str(model),
            ],
            cwd=repository,
            env=env,
            stdout=weight_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_port(6543, 120)
        _run(
            [
                str(binary),
                "--config",
                controller,
                "schedule",
                "static",
                "--inventory",
                str(config / "inventory.yaml"),
            ],
            env,
        )
        instance_id = subprocess.run(
            [
                "psql",
                "-h",
                "127.0.0.1",
                "-p",
                "55432",
                "-U",
                "dev",
                "-d",
                "dev",
                "-Atc",
                "SELECT id FROM instance WHERE name='qwen3-30b-a3b-pipeline-bench'",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
        if instance_id != "1":
            raise RuntimeError(
                f"isolated benchmark expects instance ID 1, database returned {instance_id!r}"
            )
        print(f"Qwen3 benchmark runtime is ready: {runtime}")
        return 0
    finally:
        if weight_server is not None and weight_server.poll() is None:
            os.killpg(weight_server.pid, signal.SIGINT)
            try:
                weight_server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(weight_server.pid, signal.SIGTERM)
                weight_server.wait(timeout=15)
        if weight_log is not None:
            weight_log.close()
        if started:
            subprocess.run(
                ["pg_ctl", "-D", str(postgres), "stop", "-m", "fast"],
                check=False,
                env=env,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
