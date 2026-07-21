import concurrent.futures
import importlib.util
import json
import os
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
import grpc
import safetensors.torch as st
import torch

from expertkit_vllm.grpc_client import ExpertKitClient
from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2
from vllm.v1.worker.ubatching import UBatchAborted, UBatchCoordinator


class FakeStub:
    def __init__(self):
        self.requests = []

    def Forward(self, request, timeout):
        self.requests.append((request, timeout))
        hidden = st.load(request.tensor)["data"]
        output = hidden[:, None, :].repeat(1, 2, 1)
        return expert_pb2.ForwardResp(
            output_tensor=st.save({"data": output}),
            request_id=request.request_id,
            microbatch_id=request.microbatch_id,
            layer_id=request.layer_id,
        )


class MismatchedStub(FakeStub):
    def Forward(self, request, timeout):
        response = super().Forward(request, timeout)
        response.request_id = "wrong-request"
        return response


class UnavailableRpc(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class FailingStub:
    def Forward(self, request, timeout):
        raise UnavailableRpc()


def make_client() -> ExpertKitClient:
    client = ExpertKitClient.__new__(ExpertKitClient)
    client.stub = FakeStub()
    client.timeout = 3
    client._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    client._instance_id = "test-instance"
    client._request_ids = iter(range(1, 100))
    return client


def test_async_client_preserves_metadata_and_tensor():
    client = make_client()
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    call = client.submit_forward_expert(
        [["m/l1-e0", "m/l1-e1"], ["m/l1-e1", "m/l1-e0"]],
        hidden,
        request_id="request-7",
        microbatch_id=3,
        layer_id=1,
        pipeline_enabled=True,
    )
    assert isinstance(call.a2e_future.result(timeout=2), int)
    output = call.future.result(timeout=2)
    assert output.shape == (2, 2, 3)
    request, timeout = client.stub.requests[0]
    assert request.request_id == "request-7"
    assert request.microbatch_id == 3
    assert request.layer_id == 1
    assert request.pipeline_enabled
    assert timeout == 3
    client._executor.shutdown(wait=True)


def test_blocking_client_forwards_real_microbatch_and_layer_metadata():
    client = ExpertKitClient.__new__(ExpertKitClient)
    captured = {}
    result = concurrent.futures.Future()
    result.set_result(torch.ones(1, 2, 3))

    def submit(expert_ids, hidden_state, **metadata):
        captured.update(metadata)
        return SimpleNamespace(future=result)

    client.submit_forward_expert = submit
    output = client.forward_expert(
        [["m/l17-e0", "m/l17-e1"]],
        torch.ones(1, 3),
        request_id="sync-request",
        microbatch_id=3,
        layer_id=17,
    )

    assert output.shape == (1, 2, 3)
    assert captured == {
        "request_id": "sync-request",
        "microbatch_id": 3,
        "layer_id": 17,
        "pipeline_enabled": False,
    }


def test_async_client_rejects_mismatched_response_metadata():
    client = make_client()
    client.stub = MismatchedStub()
    call = client.submit_forward_expert(
        [["m/l1-e0"]],
        torch.ones(1, 3),
        request_id="request-8",
        microbatch_id=2,
        layer_id=1,
        pipeline_enabled=True,
    )
    with pytest.raises(RuntimeError, match="request_id mismatch"):
        call.future.result(timeout=2)
    client._executor.shutdown(wait=True)


def test_rpc_failure_completes_both_futures_without_hanging():
    client = make_client()
    client.stub = FailingStub()
    call = client.submit_forward_expert(
        [["m/l1-e0"]],
        torch.ones(1, 3),
        request_id="request-9",
        microbatch_id=1,
        layer_id=1,
        pipeline_enabled=True,
    )
    assert isinstance(call.a2e_future.result(timeout=2), int)
    with pytest.raises(RuntimeError, match="gRPC failed.*UNAVAILABLE"):
        call.future.result(timeout=2)
    client._executor.shutdown(wait=True)


def test_completion_driven_scheduler_uses_future_completion_order():
    coordinator = UBatchCoordinator(4)
    futures = [concurrent.futures.Future() for _ in range(4)]
    barrier = threading.Barrier(5)
    observed = []
    threads = []

    def run(ubatch_id):
        barrier.wait()
        coordinator.enter(ubatch_id)
        observed.append((ubatch_id, "A"))
        coordinator.wait_for_future(ubatch_id, futures[ubatch_id])
        observed.append((ubatch_id, "E2A"))
        coordinator.finish(ubatch_id)

    for ubatch_id in range(4):
        thread = threading.Thread(target=run, args=(ubatch_id,))
        threads.append(thread)
        thread.start()
    barrier.wait()
    time.sleep(0.02)
    for ubatch_id in (2, 0, 3, 1):
        futures[ubatch_id].set_result(ubatch_id)
    for thread in threads:
        thread.join(timeout=2)

    assert observed[:4] == [(0, "A"), (1, "A"), (2, "A"), (3, "A")]
    assert [item[0] for item in observed[4:]] == [2, 0, 3, 1]
    assert not any(thread.is_alive() for thread in threads)


def test_scheduler_propagates_failure_to_waiters():
    coordinator = UBatchCoordinator(2)
    future = concurrent.futures.Future()
    failure = RuntimeError("boom")
    coordinator.enter(0)
    coordinator.abort(failure)
    with pytest.raises(UBatchAborted) as raised:
        coordinator.wait_for_future(0, future)
    assert raised.value.__cause__ is failure


def test_trace_validator_accepts_four_stage_overlap():
    script = Path(__file__).parents[1] / "scripts" / "validate_pipeline_trace.py"
    spec = importlib.util.spec_from_file_location("trace_validator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    for ubatch_id in range(4):
        request_id = f"r{ubatch_id}"
        starts = {
            "A": ubatch_id * 20,
            "A2E": 10 + ubatch_id * 20,
            "E": 20 + ubatch_id * 20,
            "E2A": 120 + ubatch_id * 20,
        }
        durations = {"A": 10, "A2E": 10, "E": 100, "E2A": 10}
        for stage in ("A", "A2E", "E", "E2A"):
            events.append(
                {
                    "name": stage,
                    "cat": "expertkit_pipeline",
                    "ts": starts[stage],
                    "dur": durations[stage],
                    "args": {
                        "request_id": request_id,
                        "microbatch_id": ubatch_id,
                        "layer_id": 0,
                    },
                }
            )
    result = module.validate(events)
    assert result["microbatches"] == [0, 1, 2, 3]
    assert result["compute_comm_overlap"]
    assert result["concurrent_experts"]


def test_trace_renderer_emits_scaled_svg(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "render_pipeline_trace.py"
    spec = importlib.util.spec_from_file_location("trace_renderer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    for ubatch_id in range(4):
        request_id = f"render-r{ubatch_id}"
        for stage, start, duration in (
            ("A", ubatch_id * 20, 30),
            ("A2E", 30 + ubatch_id * 20, 10),
            ("E", 40 + ubatch_id * 20, 100),
            ("E2A", 140 + ubatch_id * 20, 10),
        ):
            events.append(
                {
                    "name": stage,
                    "cat": "expertkit_pipeline",
                    "ts": start,
                    "dur": duration,
                    "args": {
                        "request_id": request_id,
                        "microbatch_id": ubatch_id,
                        "layer_id": 7,
                    },
                }
            )
    output = tmp_path / "timeline.svg"
    summary = module.render(events, output)
    assert summary["layer_id"] == 7
    assert summary["compute_expert_overlap"]
    assert summary["concurrent_experts"]
    assert output.read_text().startswith("<svg")


def test_nsys_analysis_and_four_lane_renderer(tmp_path):
    scripts = Path(__file__).parents[1] / "scripts"
    analyzer_spec = importlib.util.spec_from_file_location(
        "nsys_analyzer", scripts / "analyze_nsys_pipeline.py"
    )
    analyzer = importlib.util.module_from_spec(analyzer_spec)
    analyzer_spec.loader.exec_module(analyzer)
    renderer_spec = importlib.util.spec_from_file_location(
        "nsys_renderer", scripts / "render_nsys_pipeline.py"
    )
    renderer = importlib.util.module_from_spec(renderer_spec)
    renderer_spec.loader.exec_module(renderer)

    database = tmp_path / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                text TEXT, textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, correlationId INTEGER,
                globalPid INTEGER, demangledName INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, correlationId INTEGER,
                globalPid INTEGER, copyKind INTEGER, bytes INTEGER
            );
            CREATE TABLE SCHED_EVENTS (
                start INTEGER, isSchedIn INTEGER, globalTid INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'fixture_kernel');
            """
        )
        correlation = 1
        for layer in range(2):
            for ubatch in range(4):
                base = layer * 1_000 + ubatch * 150
                request_id = f"fixture-l{layer}-u{ubatch}"
                tid = 100 + ubatch
                ranges = (
                    (base, base + 50, tid, f"EK:A:u={ubatch}:l={layer}"),
                    (
                        base + 60,
                        base + 80,
                        tid,
                        f"EK:A2E:req={request_id}:u={ubatch}:l={layer}",
                    ),
                    (
                        base + 300,
                        base + 500,
                        200 + ubatch,
                        f"EK:E:req={request_id}:u={ubatch}:l={layer}:expert=e0",
                    ),
                    (
                        base + 510,
                        base + 530,
                        tid,
                        f"EK:E2A:req={request_id}:u={ubatch}:l={layer}",
                    ),
                )
                connection.executemany(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)", ranges
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                    (base + 10, base + 11, tid, correlation),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 1, 1)",
                    (base + 100, base + 200, correlation),
                )
                correlation += 1
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                    (base + 70, base + 71, tid, correlation),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 1, 2, 4096)",
                    (base + 210, base + 230, correlation),
                )
                correlation += 1
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                    (base + 520, base + 521, tid, correlation),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 1, 1, 32768)",
                    (base + 550, base + 580, correlation),
                )
                correlation += 1

    result = analyzer.analyze(database)
    assert result["selected_layers"] == [0, 1]
    assert len(result["calls"]) == 8
    assert result["metrics"]["total"]["communication_ns"] > 0
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(result))
    output = tmp_path / "nsys-timeline.svg"
    summary = renderer.render(result, output)
    svg = output.read_text()
    assert summary["microbatches"] == [0, 1, 2, 3]
    assert svg.count(">Attention</text>") == 1
    assert svg.count(">A2E</text>") == 1
    assert svg.count(">Expert</text>") == 1
    assert svg.count(">E2A</text>") == 1


def test_nsys_comparison_correlates_cross_process_phases(tmp_path):
    scripts = Path(__file__).parents[1] / "scripts"
    pipeline_spec = importlib.util.spec_from_file_location(
        "analyze_nsys_pipeline", scripts / "analyze_nsys_pipeline.py"
    )
    pipeline_module = importlib.util.module_from_spec(pipeline_spec)
    pipeline_spec.loader.exec_module(pipeline_module)
    sys.modules["analyze_nsys_pipeline"] = pipeline_module
    spec = importlib.util.spec_from_file_location(
        "nsys_comparison", scripts / "analyze_nsys_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    database = tmp_path / "comparison.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                text TEXT, textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, correlationId INTEGER,
                globalPid INTEGER, copyKind INTEGER, bytes INTEGER
            );
            INSERT INTO NVTX_EVENTS VALUES
              (-100, -90, 1, 'EKC:GRPC_CHANNEL_CREATE:req=none:u=0:l=0', NULL),
              (-90, -40, 1, 'EKC:GRPC_CHANNEL_READY:req=none:u=0:l=0', NULL),
              (0, 1000, 1, 'EK_PROFILE_WINDOW', NULL),
              (100, 900, 1, 'EKC:CLIENT_CONTROLLER_GRPC:req=r1:u=0:l=7', NULL),
              (150, 850, 2, 'EKR:CONTROLLER_REQUEST:req=r1:u=0:l=7', NULL),
              (220, 780, 2, 'EKR:CONTROLLER_WORKER_GRPC:req=r1:u=0:l=7', NULL),
              (260, 300, 3, 'EKR:WORKER_QUEUE:req=r1:u=0:l=7', NULL),
              (300, 700, 3, 'EK:E:req=r1:u=0:l=7:expert=e1', NULL),
              (80, 120, 1, 'EKC:D2H_ENQUEUE:req=r1:u=0:l=7', NULL),
              (880, 920, 1, 'EKC:H2D_COPY:req=r1:u=0:l=7', NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
              (90, 91, 1, 1), (890, 891, 1, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
              (110, 115, 1, 1, 2, 4096),
              (900, 910, 2, 1, 1, 32768);
            """
        )
    result = module.analyze_sqlite(database, "sync")
    assert result["counts"]["complete_requests"] == 1
    assert result["counts"]["worker_rpcs_per_client_request"] == 1
    assert result["counts"]["expert_max_concurrency"] == 1
    assert result["startup_phases"]["GRPC_CHANNEL_READY"]["count"] == 1
    assert result["derived"]["client_to_controller_ingress"]["mean_ms"] > 0
    assert result["gpu_copies"]["D2H_ENQUEUE"]["total_bytes"] == 4096
    assert result["gpu_copies"]["H2D_COPY"]["total_bytes"] == 32768


def test_nsys_comparison_summarizes_three_mode_benchmarks(tmp_path):
    scripts = Path(__file__).parents[1] / "scripts"
    pipeline_spec = importlib.util.spec_from_file_location(
        "analyze_nsys_pipeline", scripts / "analyze_nsys_pipeline.py"
    )
    pipeline_module = importlib.util.module_from_spec(pipeline_spec)
    pipeline_spec.loader.exec_module(pipeline_module)
    sys.modules["analyze_nsys_pipeline"] = pipeline_module
    spec = importlib.util.spec_from_file_location(
        "nsys_comparison_benchmark",
        scripts / "analyze_nsys_comparison.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = {}
    for mode, throughput, latency, tokens in (
        ("sync", 10.0, 2.0, [1, 2]),
        ("pipeline-serial", 9.0, 2.2, [1, 3]),
        ("pipeline-parallel", 12.0, 1.6, [1, 2]),
    ):
        path = tmp_path / f"{mode}.json"
        path.write_text(
            json.dumps(
                {
                    "summary": {
                        "tokens_per_second_median": throughput,
                        "latency_seconds_median": latency,
                    },
                    "runs": [
                        {
                            "repetition": 0,
                            "outputs": [{"token_ids": tokens}],
                        }
                    ],
                }
            )
        )
        paths[mode] = path

    result = module.analyze_benchmarks(paths)["modes"]
    assert result["pipeline-serial"][
        "throughput_change_vs_sync_percent"
    ] == pytest.approx(-10.0)
    assert result["pipeline-parallel"][
        "throughput_change_vs_sync_percent"
    ] == pytest.approx(20.0)
    assert not result["pipeline-serial"][
        "strict_token_equivalence_to_sync"
    ]
    assert result["pipeline-parallel"]["within_mode_stable"]


@pytest.mark.skipif(
    os.getenv("EK_PIPELINE_LIVE_TEST") != "1",
    reason="requires a running Expert-Kit grpc service chain",
)
def test_live_pipeline_rpc_matches_sequential():
    model_root = Path(os.environ["QWEN3_30B_A3B_ROOT"]).resolve(strict=True)
    model_config = json.loads((model_root / "config.json").read_text())
    hidden_size = int(model_config["hidden_size"])
    top_k = int(model_config.get("num_experts_per_tok", 8))
    num_experts = int(model_config["num_experts"])
    model_name = os.getenv("EK_MODEL_NAME", "qwen3-30b-a3b")
    client = ExpertKitClient(
        os.getenv("EK_ADDR", "localhost:5002"),
        timeout_sec=float(os.getenv("EK_CLIENT_TIMEOUT", "120")),
    )
    generator = torch.Generator().manual_seed(20260720)
    hidden_states = [
        torch.randn(size, hidden_size, generator=generator, dtype=torch.bfloat16)
        for size in (1, 2, 3, 4)
    ]
    expert_ids = []
    for ubatch_id, hidden in enumerate(hidden_states):
        expert_ids.append(
            [
                [
                    f"{model_name}/l0-e{(ubatch_id + token + offset) % num_experts}"
                    for offset in range(top_k)
                ]
                for token in range(hidden.shape[0])
            ]
        )

    sequential = [
        client.forward_expert(ids, hidden)
        for ids, hidden in zip(expert_ids, hidden_states, strict=True)
    ]
    calls = [
        client.submit_forward_expert(
            ids,
            hidden,
            request_id=f"live-u{ubatch_id}",
            microbatch_id=ubatch_id,
            layer_id=0,
            pipeline_enabled=True,
        )
        for ubatch_id, (ids, hidden) in enumerate(
            zip(expert_ids, hidden_states, strict=True)
        )
    ]
    concurrent = [call.future.result(timeout=120) for call in calls]
    for expected, actual in zip(sequential, concurrent, strict=True):
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        assert torch.equal(expected.cpu(), actual.cpu())
    client.close()
