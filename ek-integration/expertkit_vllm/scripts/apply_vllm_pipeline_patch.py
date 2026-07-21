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
CORE_FILES = {
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
UPGRADE_FILES = {
    "vllm/v1/worker/ubatching.py": (
        "1b21bca3b4723a5dca76317052f564bab0f1b746372b7c98fb7a344c26608666",
        "50ae00da430202e1588187dc26609bb4a76ded1d44e146e6b310ee29922e7e45",
    ),
    "vllm/v1/worker/gpu_worker.py": (
        "7e00284da7b453154af47300630483ed7ea5a5d79e724c5ee61d4a24edaf930e",
        "2c35138497a80573bdccaf07f810f386bde785eecc7ba8ffdf8b620989c33995",
    ),
}
ALL_FILES = tuple(dict.fromkeys((*CORE_FILES, *UPGRADE_FILES)))


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
    hashes = {relative: digest(root / relative) for relative in ALL_FILES}
    core_pristine = all(
        hashes[path] == values[0] for path, values in CORE_FILES.items()
    )
    legacy = all(hashes[path] == values[1] for path, values in CORE_FILES.items())
    upgraded = all(
        hashes[path] == values[1]
        for path, values in CORE_FILES.items()
        if path != "vllm/v1/worker/ubatching.py"
    ) and all(hashes[path] == values[1] for path, values in UPGRADE_FILES.items())
    gpu_pristine = (
        hashes["vllm/v1/worker/gpu_worker.py"]
        == UPGRADE_FILES["vllm/v1/worker/gpu_worker.py"][0]
    )
    if core_pristine and gpu_pristine:
        return "pristine"
    if legacy and gpu_pristine:
        return "legacy-patched"
    if upgraded:
        return "patched"
    details = ", ".join(f"{path}={value}" for path, value in hashes.items())
    raise RuntimeError(f"partial or unknown patch state: {details}")


def break_hardlinks(root: Path) -> None:
    for relative in ALL_FILES:
        target = root / relative
        temporary = target.with_suffix(target.suffix + ".expertkit-tmp")
        temporary.write_bytes(target.read_bytes())
        os.replace(temporary, target)


def run_patch(root: Path, filename: str, reverse: bool) -> None:
    patch_file = Path(__file__).resolve().parent.parent / "patches" / filename
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
        print(f"vLLM {EXPECTED_VERSION} Expert Kit patch state: {current}")
        return 0 if current == "patched" else 1
    if args.apply and current == "patched":
        print("Expert Kit pipeline patch is already applied")
        return 0
    if args.reverse and current == "pristine":
        print("Expert Kit pipeline patch is already reversed")
        return 0

    break_hardlinks(root)
    core_patch = "vllm-0.25.1-expertkit-pipeline.patch"
    upgrade_patch = "vllm-0.25.1-expertkit-pipeline-v2.patch"
    if args.apply:
        if current == "pristine":
            run_patch(root, core_patch, reverse=False)
            current = state(root)
            if current != "legacy-patched":
                raise RuntimeError(f"expected legacy-patched, found {current}")
        run_patch(root, upgrade_patch, reverse=False)
    else:
        if current == "patched":
            run_patch(root, upgrade_patch, reverse=True)
            current = state(root)
            if current != "legacy-patched":
                raise RuntimeError(f"expected legacy-patched, found {current}")
        run_patch(root, core_patch, reverse=True)
    expected = "pristine" if args.reverse else "patched"
    actual = state(root)
    if actual != expected:
        raise RuntimeError(f"expected {expected} after patch operation, found {actual}")
    print(f"vLLM {EXPECTED_VERSION} Expert Kit patch state: {actual}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
