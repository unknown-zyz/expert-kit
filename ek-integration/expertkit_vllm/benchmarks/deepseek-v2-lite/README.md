# DeepSeek-V2-Lite four-stage benchmark

This deployment uses one RTX 5090 vLLM Frontend and one four-slot CPU Torch
Worker. Runtime state is isolated under
`/home/zhangyz/expert-kit/output/deepseek-v2-pipeline-dev-py`; PostgreSQL uses
port 55432 so it does not conflict with the Host monitoring database on 5432.

From the repository worktree, define:

```bash
export EK_BENCH_ROOT=/home/zhangyz/expert-kit/output/deepseek-v2-pipeline-dev-py
export EK_MODEL_ROOT=/data/models/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat
export EK_DATASET=/data/datasets/ShareGPT52K/sg_52k.json
export EK_BENCH_CONFIG=$PWD/ek-integration/expertkit_vllm/benchmarks/deepseek-v2-lite
mkdir -p "$EK_BENCH_ROOT"/{postgres,weight-cache,worker-cache,logs,results,nsys}
```

Initialize the isolated database once and build the services:

```bash
initdb -D "$EK_BENCH_ROOT/postgres" -A trust -U dev
pg_ctl -D "$EK_BENCH_ROOT/postgres" -l "$EK_BENCH_ROOT/logs/postgres.log" \
  -o "-p 55432 -h 127.0.0.1 -k $EK_BENCH_ROOT/postgres" start
createdb -h 127.0.0.1 -p 55432 -U dev dev
cargo build --release --bin ek-cli
env -u EK_CONFIG target/release/ek-cli \
  --config "$EK_BENCH_CONFIG/controller.yaml" db migrate
env -u EK_CONFIG target/release/ek-cli \
  --config "$EK_BENCH_CONFIG/controller.yaml" weight build \
  --model "$EK_MODEL_ROOT" --cache-dir "$EK_BENCH_ROOT/weight-cache"
```

Start Weight Server, register/schedule the model, then start Controller and
Worker in independent terminals. The fresh database assigns instance ID 1:

```bash
env -u EK_CONFIG target/release/ek-cli --config "$EK_BENCH_CONFIG/controller.yaml" \
  weight-server --model "$EK_MODEL_ROOT"

env -u EK_CONFIG target/release/ek-cli --config "$EK_BENCH_CONFIG/controller.yaml" \
  model upsert --name DeepSeek-V2-Lite-Chat
env -u EK_CONFIG target/release/ek-cli --config "$EK_BENCH_CONFIG/controller.yaml" \
  schedule static --inventory "$EK_BENCH_CONFIG/inventory.yaml"
env -u EK_CONFIG target/release/ek-cli \
  --config "$EK_BENCH_CONFIG/controller.yaml" controller

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ek-worker/.venv/bin/ek-worker --config "$EK_BENCH_CONFIG/worker.yaml"
```

Wait for all routes, then run the two modes in separate processes:

```bash
PYTHONPATH=ek-proto/src:ek-transport/src \
  .venv/bin/python \
  ek-integration/expertkit_vllm/scripts/check_deepseek_topology.py

export EK_ADDR=127.0.0.1:5002 EK_INSTANCE_ID=1 EK_CLIENT_TIMEOUT=600
export PYTHONPATH=$PWD/ek-proto/src:$PWD/ek-transport/src:$PWD/ek-integration/expertkit_vllm

.venv/bin/python -m expertkit_vllm.benchmark run \
  --model "$EK_MODEL_ROOT" --dataset "$EK_DATASET" --mode sync \
  --batch-sizes 4 8 16 32 64 --output "$EK_BENCH_ROOT/results/sync.json"
.venv/bin/python -m expertkit_vllm.benchmark run \
  --model "$EK_MODEL_ROOT" --dataset "$EK_DATASET" --mode pipeline \
  --batch-sizes 4 8 16 32 64 --output "$EK_BENCH_ROOT/results/pipeline.json"
.venv/bin/python -m expertkit_vllm.benchmark compare \
  --sync "$EK_BENCH_ROOT/results/sync.json" \
  --pipeline "$EK_BENCH_ROOT/results/pipeline.json" \
  --output "$EK_BENCH_ROOT/results/comparison.json"
```

The benchmark pins both modes to vLLM's legacy GPU model runner. The reviewed
four-stage patch targets `vllm/v1/worker/gpu_model_runner.py`; vLLM 0.25.1's
automatically selected V2 runner does not execute that patch. Pipeline startup
rejects an explicit `VLLM_USE_V2_MODEL_RUNNER=1` instead of silently running
without uBatching.

For the detailed cross-process Nsight Systems profile, use the service launcher.
It starts and stops the isolated PostgreSQL, Weight Server, Controller, and CPU
Worker and enables the profile-only NVTX context. Replace `MODE` with `sync` and
`pipeline` in two separate runs:

```bash
nsys profile --trace=cuda,nvtx,osrt --trace-fork-before-exec=true \
  --sample=none --cpuctxsw=process-tree --cuda-event-trace=false \
  --resolve-symbols=false --capture-range=cudaProfilerApi \
  --capture-range-end=stop --wait=primary --force-overwrite=true \
  --output="$EK_BENCH_ROOT/nsys-detailed-MODE-b64" \
  .venv/bin/python \
  ek-integration/expertkit_vllm/scripts/run_deepseek_nsys_profile.py \
  --mode MODE --batch-size 64 --output-tokens 32
```

Analyze the two reports and render the documentation figures:

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/analyze_deepseek_nsys.py \
  --sync "$EK_BENCH_ROOT/nsys-detailed-sync-b64.nsys-rep" \
  --pipeline "$EK_BENCH_ROOT/nsys-detailed-pipeline-b64.nsys-rep" \
  --output "$EK_BENCH_ROOT/nsys-detailed-analysis-b64.json"

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/render_deepseek_pipeline.py \
  "$EK_BENCH_ROOT/nsys-detailed-analysis-b64.json" \
  --heatmap doc/assets/deepseek-v2-lite-four-stage-share.svg \
  --timeline doc/assets/deepseek-v2-lite-four-stage-overlap.svg
```

Use `--wait=primary`; `--wait=all` can wait for a reparented nsys agent after
the primary process and services have already exited. Open `.nsys-rep` with
Nsight Systems UI on a graphical workstation, or use `nsys stats` on the server.

Stop Worker and Controller with `Ctrl-C`, then Weight Server. Stop PostgreSQL
with `pg_ctl -D "$EK_BENCH_ROOT/postgres" stop`. The cache is not deleted
automatically; remove only the explicit `$EK_BENCH_ROOT` directory when the
reports are no longer needed.
