"""Tests for deterministic ShareGPT sampling and report comparison."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from expertkit_vllm.benchmark import compare_reports, load_sharegpt_samples


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is True
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def test_sharegpt_sampling_is_seeded_and_filters_lengths(tmp_path: Path) -> None:
    records = [
        {
            "id": str(index),
            "conversations": [
                {"from": "human", "value": "x" * index},
                {"from": "gpt", "value": "answer"},
            ],
        }
        for index in range(1, 12)
    ]
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")

    first = load_sharegpt_samples(
        dataset,
        Tokenizer(),
        count=4,
        seed=42,
        min_prompt_tokens=4,
        max_prompt_tokens=8,
    )
    second = load_sharegpt_samples(
        dataset,
        Tokenizer(),
        count=4,
        seed=42,
        min_prompt_tokens=4,
        max_prompt_tokens=8,
    )

    assert first == second
    assert all(4 <= len(sample.prompt_token_ids) <= 8 for sample in first)


def _report(mode: str, token_ids: list[list[int]]) -> dict[str, object]:
    return {
        "mode": mode,
        "sample_manifest_sha256": "same",
        "deterministic": True,
        "batches": [
            {
                "batch_size": 4,
                "sample_ids": ["a", "b", "c", "d"],
                "reference_token_ids": token_ids,
                "summary": {
                    "output_tokens_per_second_median": 10 if mode == "sync" else 15
                },
            }
        ],
    }


def test_compare_reports_requires_strict_token_equality() -> None:
    expected = [[1, 2], [3], [4], [5]]
    comparison = compare_reports(
        _report("sync", expected), _report("pipeline", expected)
    )

    assert comparison["passed"] is True
    assert comparison["batches"][0]["output_throughput_speedup"] == 1.5

    actual = [[1, 9], [3], [4], [5]]
    mismatch = compare_reports(_report("sync", expected), _report("pipeline", actual))
    assert mismatch["passed"] is False
    assert mismatch["first_mismatch"] == {
        "batch_size": 4,
        "request_index": 0,
        "dataset_id": "a",
        "token_index": 1,
        "sync_token": 2,
        "pipeline_token": 9,
    }


def test_compare_reports_rejects_different_manifests() -> None:
    pipeline = _report("pipeline", [[1], [2], [3], [4]])
    pipeline["sample_manifest_sha256"] = "different"

    with pytest.raises(ValueError, match="different ShareGPT"):
        compare_reports(_report("sync", [[1], [2], [3], [4]]), pipeline)
