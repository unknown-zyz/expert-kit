#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_VERSION = "0.25.1"
FILES = {
    "vllm/model_executor/models/qwen3_moe.py": (
        "ea439daddb5e5c3c46f48f204e2f7a90612b349e33e63f2d5fc7e57916e5a415",
        "4b988573958b51591a2833f5faa101138c89d166b15d6c3696bfd063da30ac15",
    ),
    "vllm/config/vllm.py": (
        "caf6db4dbbafb3e2194022d779e4635e78ea1f51bfc5d299997640cd2806cb05",
        "a100227cbb76a9719cc01e4d4ec10cdab46fc92716734b21e511fab32b754aab",
    ),
    "vllm/v1/worker/ubatching.py": (
        "40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7",
        "1b21bca3b4723a5dca76317052f564bab0f1b746372b7c98fb7a344c26608666",
    ),
    "vllm/v1/worker/gpu_ubatch_wrapper.py": (
        "f0ef8b9315ff2af1b688a1912a7c07a91f3fffc9c6bc4befde406138e51d3b79",
        "08a4768a60320b020e43a54d62bef15ee63a045fc2c2dd589ce8c657060e6ea4",
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        "6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e",
        "dbe5972a67424628e9ba24e04c9385928bcfa5fd7247bdd49cf0b449bd0bec44",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_vllm() -> Path:
    version = importlib.metadata.version("vllm")
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"expected vLLM {EXPECTED_VERSION}, found {version}")
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("could not locate the installed vllm package")
    return Path(next(iter(spec.submodule_search_locations))).parent


def state(root: Path) -> str:
    states = []
    for relative, (pristine, patched) in FILES.items():
        actual = digest(root / relative)
        if actual == pristine:
            states.append("pristine")
        elif actual == patched:
            states.append("patched")
        else:
            raise RuntimeError(f"unexpected content hash for {root / relative}: {actual}")
    if len(set(states)) != 1:
        raise RuntimeError(f"partial patch state: {states}")
    return states[0]


def break_hardlinks(root: Path) -> None:
    for relative in FILES:
        target = root / relative
        temporary = target.with_suffix(target.suffix + ".expertkit-tmp")
        temporary.write_bytes(target.read_bytes())
        os.replace(temporary, target)


def run_patch(root: Path, reverse: bool) -> None:
    patch_file = (
        Path(__file__).resolve().parent.parent
        / "patches"
        / "vllm-0.25.1-expertkit-pipeline.patch"
    )
    command = ["patch", "--batch", "-p1", "-d", str(root)]
    if reverse:
        command.append("--reverse")
    with patch_file.open("rb") as source:
        subprocess.run(command, stdin=source, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    action.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    root = locate_vllm()
    current = state(root)
    if args.check:
        print(f"vLLM {EXPECTED_VERSION} Expert-Kit patch state: {current}")
        return 0 if current == "patched" else 1
    if args.apply and current == "patched":
        print("Expert-Kit pipeline patch is already applied")
        return 0
    if args.reverse and current == "pristine":
        print("Expert-Kit pipeline patch is already reversed")
        return 0

    break_hardlinks(root)
    run_patch(root, reverse=args.reverse)
    expected = "pristine" if args.reverse else "patched"
    actual = state(root)
    if actual != expected:
        raise RuntimeError(f"expected {expected} after patch operation, found {actual}")
    print(f"vLLM {EXPECTED_VERSION} Expert-Kit patch state: {actual}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
