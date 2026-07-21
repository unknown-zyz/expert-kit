#!/usr/bin/env python3
"""Wait until every DeepSeek-V2-Lite routed expert has a ready Worker route."""

from __future__ import annotations

import argparse
import asyncio
import json

import grpc
from expertkit_proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc
from expertkit_transport.controller.messages import TopologyMessageAssembler


async def check(endpoint: str, instance_id: int, timeout: float) -> dict[str, object]:
    expected = {(layer, expert) for layer in range(1, 27) for expert in range(64)}
    assembler = TopologyMessageAssembler(
        instance_id=instance_id,
        num_layers=27,
        experts_per_layer=64,
    )
    installed_version = 0
    installed_routes = {}
    channel = grpc.aio.insecure_channel(endpoint)
    stub = lifecycle_pb2_grpc.TopologyServiceStub(channel)
    request = lifecycle_pb2.WatchTopologyRequest(
        instance_id=instance_id,
        current_version=0,
    )
    try:
        async with asyncio.timeout(timeout):
            async for message in stub.WatchTopology(request, wait_for_ready=True):
                completed = assembler.consume(
                    message,
                    installed_version=installed_version,
                    installed_routes=installed_routes,
                )
                if completed is None:
                    continue
                installed_version, installed_routes = completed
                ready = {key for key, replicas in installed_routes.items() if replicas}
                missing = sorted(expected - ready)
                if not missing:
                    return {
                        "ready": True,
                        "instance_id": instance_id,
                        "topology_version": installed_version,
                        "expected_routes": len(expected),
                        "ready_routes": len(expected & ready),
                        "workers": sorted(
                            {
                                replica.identity.worker_id
                                for key in expected
                                for replica in installed_routes[key]
                            }
                        ),
                    }
    finally:
        await channel.close()
    raise RuntimeError("topology stream ended before all 1664 expert routes were ready")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:5002")
    parser.add_argument("--instance-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    result = asyncio.run(check(args.endpoint, args.instance_id, args.timeout))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
