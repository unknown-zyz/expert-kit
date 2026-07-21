# vLLM ExpertMesh Plugin

ExpertMesh Plugin for vLLM framework.

## Installation

Install the plugin in development mode:

```bash
pip install -e .
```

## Usage

### 1. Setup Expert-Kit Service

First, ensure your Expert-Kit service is running and accessible. Refer to [Deploying Qwen3-30B-A3B with Expert-Kit](https://github.com/expert-kit/expert-kit/blob/dev/doc/tutorial/standalone/qwen3-moe-a3b-demo.md) for details.

### 2. Model Configuration

Expert-Kit configuration can be set through model configuration parameters or environment variables. The plugin supports the following configuration options:

#### Configuration Parameters

- `ek_mode`: Operation mode, default: `"expert_mode"`
- `ek_backend_addr`: Address of your Expert-Kit service, default: `"localhost:5002"`
- `ek_debug_mode`: Enable debug mode, default: `False`
- `ek_client_timeout`: gRPC timeout in seconds, default: `2`
- `ek_model_name`: Model name for Expert-Kit service (required)

#### Method 1: Model Configuration

When loading a model with vLLM, add Expert-Kit parameters to your model configuration:

```python
from vllm import LLM

# Configure Expert-Kit through model config
model_config = {
    "ek_mode": "expert_mode",
    "ek_backend_addr": "localhost:5002",
    "ek_debug_mode": False,
    "ek_client_timeout": 2,
    "ek_model_name": "Qwen/Qwen3-MoE-A3B"
}

# Create LLM with Expert-Kit configuration
llm = LLM(
    model="Qwen/Qwen3-MoE-A3B", 
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_config=model_config
)
```

#### Method 2: Environment Variables

Alternatively, configure Expert-Kit using environment variables:

```bash
export EK_ENABLE=1
export EK_MODE="expert_mode"
export EK_ADDR="localhost:5002"
export EK_DEBUG_MODE="0"
export EK_CLIENT_TIMEOUT="2"
export EK_MODEL_NAME="Qwen/Qwen3-MoE-A3B"
```

Note: Environment variables take precedence over model configuration parameters.

### 3. Enable Expert-Kit Plugin

Set the `EK_ENABLE` environment variable to activate the plugin:

```bash
export EK_ENABLE=1
```

### 4. Generate Text

Generate text as you normally would with vLLM:

```python
# Enable ExpertKit
import os
os.environ["EK_ENABLE"] = "1"

from vllm import LLM

# Method 1: Using model config
llm = LLM(
    model="Qwen/Qwen3-MoE-A3B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    model_config={
        "ek_backend_addr": "localhost:5002",
        "ek_model_name": "Qwen/Qwen3-MoE-A3B"
    }
)

# Generate text
outputs = llm.generate("Hello, world!", max_tokens=100)
print(outputs[0].outputs[0].text)
```

## Supported Models

This plugin currently supports:
- **Qwen3-MoE-A3B**: `Qwen/Qwen3-MoE-A3B` (requires vLLM >= 0.8.4)
- **DeepSeek-V2**: `deepseek-ai/deepseek-v2-base`

## Architecture

This plugin replaces the `DeepseekV2MoE` implementation with `ExpertKitMoE`, which routes expert computation to Expert-Kit service.

## Requirements

- vLLM >= 0.8.4 (required for Qwen3-MoE support)
- grpcio >= 1.71.0
- Protobuf >= 5.29.4

## Configuration Priority

Configuration parameters are resolved in the following order (higher priority overrides lower):

1. Environment variables (highest priority)
2. Model configuration parameters
3. Default values (lowest priority)

## Deployment Example


```python
from vllm import LLM
import os

os.environ["VLLM_MLA_DISABLE"] = "1"

os.environ["EK_ENABLE"] = "1"
os.environ["EK_MODEL_NAME"] = "qwen3-30b-a3b"
os.environ["EK_MODE"] = "expert_mode"
os.environ["EK_ADDR"] = "localhost:5002"
os.environ["EK_CLIENT_TIMEOUT"] = "2"
os.environ["EK_DEBUG_MODE"] = "0"

model_root = os.environ["QWEN3_30B_A3B_ROOT"]

prompts = [
    "Hello, my name is",
    "The president of the United",
]

llm = LLM(
        model=model_root,
        trust_remote_code=True,

        max_model_len=16,
        enforce_eager=True,
        cpu_offload_gb=64,
        max_num_batched_tokens=1024
    )

outputs = llm.generate(prompts)

```

## Troubleshooting

### Common Issues

1. **Missing EK_MODEL_NAME**: Ensure `ek_model_name` is set in model config or `EK_MODEL_NAME` environment variable is set.

2. **Connection timeout**: Increase `ek_client_timeout` value if your Expert-Kit service is slow to respond.

3. **Debug mode**: Set `ek_debug_mode=True` or `EK_DEBUG_MODE=1` to enable detailed logging.

## Four-stage Expert Pipeline

For a file-by-file review guide and end-to-end experiment reproduction steps,
see [`doc/ae-dbo-code-review-and-reproduction.md`](../../doc/ae-dbo-code-review-and-reproduction.md).

The Expert-Kit pipeline targets vLLM 0.25.1 and keeps four microbatches in
flight across Attention (`A`), dispatch (`A2E`), remote expert execution (`E`),
and combine (`E2A`). Apply the tracked vLLM patch after installing the editable
plugin:

```bash
uv pip install vllm==0.25.1
uv pip install --no-deps -e ek-integration/expertkit_vllm
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --apply
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --check
```

Enable the pipeline and use eager execution with four uBatches. Do not combine
`ubatch_size=4` with `enable_dbo=True`, because vLLM gives the latter a fixed
two-uBatch interpretation.

```python
os.environ["EK_PIPELINE_ENABLE"] = "1"
os.environ["EK_PIPELINE_TRACE"] = "/tmp/expertkit-pipeline-trace.json"

llm = LLM(
    model=os.environ["QWEN3_30B_A3B_ROOT"],
    enforce_eager=True,
    ubatch_size=4,
    dbo_decode_token_threshold=0,
    dbo_prefill_token_threshold=0,
)
```

The batch must be a uniform decode batch containing at least four requests.
Prefill, mixed prefill/decode, and smaller decode batches use the compatible
non-uBatch execution path. The first pipeline version requires a
Controller-to-Worker `grpc` inventory; SHM and RDMA continue to use the legacy
synchronous path.

Validate the generated Chrome trace with:

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/validate_pipeline_trace.py \
  /tmp/expertkit-pipeline-trace.json
```

The validator requires all four stages for complete remote calls, four observed
uBatch IDs, overlapping `E(uBatch i)` and `A(uBatch j)` intervals, and multiple
concurrently in-flight expert calls. A periodic snapshot may contain a small
number of still-running calls at its tail; incomplete calls elsewhere fail
validation. Render a time-scaled SVG from a real trace with:

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/render_pipeline_trace.py \
  /tmp/expertkit-pipeline-trace.json \
  doc/assets/ae-dbo-four-stage-overlap.svg
```

For hardware timing, build the release Worker and profile the complete service
process tree. `--trace-fork-before-exec=true` is required for NVTX ranges in
vLLM's forkserver-created EngineCore:

```bash
cargo build --release --bin ek-cli
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --sample=none --cpuctxsw=process-tree \
  --cuda-event-trace=true --resolve-symbols=false \
  --wait=primary --force-overwrite=true \
  --output=/tmp/expertkit-four-stage \
  .venv/bin/python \
  ek-integration/expertkit_vllm/scripts/run_nsys_pipeline_profile.py \
  --config dev/hello-world.config.yaml \
  --model qwen3-30b-a3b

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/analyze_nsys_pipeline.py \
  /tmp/expertkit-four-stage.nsys-rep \
  /tmp/expertkit-four-stage-analysis.json
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/render_nsys_pipeline.py \
  /tmp/expertkit-four-stage-analysis.json \
  doc/assets/ae-dbo-nsys-four-stage-overlap.svg
```

The analyzer counts actual CUDA kernels as `A`, Worker `exp.forward()` on-CPU
intervals as `E`, and the non-compute intervals between them as `A2E`/`E2A`.
It reports how much communication overlaps another uBatch's hardware compute.
See `doc/ae-dbo-pipeline-design.md` for the exact definitions and current
measured result.

Do not enable `EK_PIPELINE_TRACE` during performance benchmarks because trace
serialization and file I/O materially affect throughput. Run the matched
sync/pipeline benchmark with:

```bash
timeout 1800s .venv/bin/python \
  ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py \
  --warmup 1 --repetitions 5 --max-tokens 32 \
  --output /tmp/expertkit-pipeline-benchmark.json
```

See `doc/ae-dbo-pipeline-design.md` for source-level changes, correctness
levels, the real overlap diagram, and current performance results. To restore
pristine vLLM source files, run the patch script with `--reverse`.
