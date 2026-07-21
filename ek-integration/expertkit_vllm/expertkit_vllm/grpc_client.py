import itertools
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import grpc
import safetensors.torch as st
import torch

from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2, expert_pb2_grpc
from expertkit_vllm import profile

MAX_METADATA_SIZE = 20 * 1024
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
PIPELINE_WORKERS = 4


@dataclass
class ExpertCall:
    a2e_future: Future[int]
    future: Future[torch.Tensor]


class ExpertKitClient:
    def __init__(self, expertkit_addr: str = "", timeout_sec: float = 2.0):
        print(
            f"🚀 ExpertKitClient Init: ek_addr({expertkit_addr}), "
            f"timeout({timeout_sec}s)"
        )
        with profile.range("GRPC_CHANNEL_CREATE"):
            self.channel = grpc.insecure_channel(
                expertkit_addr,
                options=[
                    ("grpc.max_metadata_size", MAX_METADATA_SIZE),
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ],
            )
            self.stub = expert_pb2_grpc.ComputationServiceStub(self.channel)
        self.timeout = timeout_sec
        if profile.enabled():
            with profile.range("GRPC_CHANNEL_READY"):
                grpc.channel_ready_future(self.channel).result(
                    timeout=self.timeout
                )
        self._executor = ThreadPoolExecutor(
            max_workers=PIPELINE_WORKERS,
            thread_name_prefix="expertkit-a2e",
        )
        self._instance_id = f"vllm-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._request_ids = itertools.count(1)

    def next_request_id(self) -> str:
        return f"{self._instance_id}-{next(self._request_ids)}"

    @staticmethod
    def _stage_hidden_state(
        hidden_state: torch.Tensor,
        *,
        request_id: str,
        microbatch_id: int,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        with profile.range(
            "D2H_PREPARE", request_id, microbatch_id, layer_id
        ):
            source = hidden_state.detach().contiguous()
        if source.device.type != "cuda":
            return source.cpu(), None

        with profile.range(
            "D2H_ENQUEUE", request_id, microbatch_id, layer_id
        ):
            host = torch.empty_like(source, device="cpu", pin_memory=True)
            host.copy_(source, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
        return host, ready

    def submit_forward_expert(
        self,
        expert_ids: List[List[str]],
        hidden_state: torch.Tensor,
        *,
        request_id: str,
        microbatch_id: int,
        layer_id: int,
        pipeline_enabled: bool,
    ) -> ExpertCall:
        host_state, ready_event = self._stage_hidden_state(
            hidden_state,
            request_id=request_id,
            microbatch_id=microbatch_id,
            layer_id=layer_id,
        )
        target_is_cuda = hidden_state.device.type == "cuda"
        seq_infos = [
            expert_pb2.ForwardReq.SequenceInfo(experts=ids) for ids in expert_ids
        ]

        a2e_future: Future[int] = Future()

        def invoke() -> torch.Tensor:
            if ready_event is not None:
                with profile.range(
                    "D2H_WAIT", request_id, microbatch_id, layer_id
                ):
                    ready_event.synchronize()
            with profile.range(
                "CLIENT_INPUT_ST_SAVE", request_id, microbatch_id, layer_id
            ):
                tensor_data = st.save({"data": host_state})
            if not a2e_future.done():
                a2e_future.set_result(time.perf_counter_ns())
            with profile.range(
                "CLIENT_PROTO_BUILD", request_id, microbatch_id, layer_id
            ):
                request = expert_pb2.ForwardReq(
                    instance_id=self._instance_id,
                    sequences=seq_infos,
                    tensor=tensor_data,
                    request_id=request_id,
                    microbatch_id=microbatch_id,
                    layer_id=layer_id,
                    pipeline_enabled=pipeline_enabled,
                )
            try:
                with profile.range(
                    "CLIENT_CONTROLLER_GRPC",
                    request_id,
                    microbatch_id,
                    layer_id,
                ):
                    response: expert_pb2.ForwardResp = self.stub.Forward(
                        request, timeout=self.timeout
                    )
            except grpc.RpcError as exc:
                raise RuntimeError(
                    f"gRPC failed for {request_id}: {exc.code().name}"
                ) from exc

            if response.request_id and response.request_id != request_id:
                raise RuntimeError(
                    f"response request_id mismatch: expected {request_id}, "
                    f"got {response.request_id}"
                )
            if response.microbatch_id != microbatch_id:
                raise RuntimeError(
                    "response microbatch_id mismatch: "
                    f"expected {microbatch_id}, got {response.microbatch_id}"
                )
            if response.layer_id != layer_id:
                raise RuntimeError(
                    f"response layer_id mismatch: expected {layer_id}, "
                    f"got {response.layer_id}"
                )
            with profile.range(
                "CLIENT_OUTPUT_ST_LOAD", request_id, microbatch_id, layer_id
            ):
                output = st.load(response.output_tensor)["data"]
            if target_is_cuda:
                with profile.range(
                    "CLIENT_OUTPUT_PIN", request_id, microbatch_id, layer_id
                ):
                    output = output.pin_memory()
            return output

        result_future = self._executor.submit(invoke)

        def propagate_failure(completed: Future[torch.Tensor]) -> None:
            if completed.exception() is not None and not a2e_future.done():
                a2e_future.set_exception(completed.exception())

        result_future.add_done_callback(propagate_failure)
        return ExpertCall(a2e_future=a2e_future, future=result_future)

    def forward_expert(
        self,
        expert_ids: List[List[str]],
        hidden_state: torch.Tensor,
        *,
        request_id: str | None = None,
        microbatch_id: int = 0,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Backward-compatible blocking expert call."""
        request_id = request_id or self.next_request_id()
        call = self.submit_forward_expert(
            expert_ids,
            hidden_state,
            request_id=request_id,
            microbatch_id=microbatch_id,
            layer_id=layer_id,
            pipeline_enabled=False,
        )
        output = call.future.result()
        with profile.range(
            "H2D_COPY", request_id, microbatch_id, layer_id
        ):
            return output.to(hidden_state.device)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.channel.close()
