# Qwen3-30B-A3B four-stage benchmark

Run the complete Python Worker experiment from the repository root:

```bash
/home/zhangyz/expert-kit/.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/run_qwen3_experiment.py
```

Defaults are the local Qwen checkpoint, ShareGPT52K, batch 64, 256 fixed input
tokens, 32 output tokens, one warmup and five measured runs. The script:

1. applies/checks the versioned vLLM 0.25.1 pipeline patch, initializes isolated
   PostgreSQL on port 55432 and idempotently builds the Expert Kit weight index;
2. registers and statically schedules all 48 × 128 expert routes;
3. starts Weight Server, Controller and the Python CPU Worker for sync and
   pipeline performance runs;
4. profiles one extra run per mode with Nsight Systems;
5. writes strict TPOT, throughput, token equality, global/per-layer A/A2E/E/E2A
   time, communication hiding, raw NSys reports and two SVG figures.

Results are under
`/home/zhangyz/expert-kit/output/qwen3-30b-a3b-pipeline-dev-py/results/`.
Normal performance runs do not enable NSys. Use `--skip-nsys` for a shorter
throughput-only run or `--resume` to reuse completed outputs.

The default service endpoints are PostgreSQL 55432, Weight Server 6543,
Controller 5001/5002 and Worker gRPC/peer 51061/51062. The launcher refuses to
reuse a listening endpoint and only stops process groups that it started.

The ShareGPT prompts are truncated to exactly 256 local tokenizer IDs. Both
modes receive the same manifest because the benchmark's deterministic seed and
local tokenization are identical. Pipeline patch v3 also isolates attention
metadata by uBatch id, so uneven dynamic batches are supported when requests
finish at different scheduler steps.

Stage definitions are:

- A: union of CUDA kernel active time in the decoder layer scope, including
  local attention/router/combine work;
- A2E: Frontend dispatch start through first Worker expert compute start;
- E: Python Worker Torch backend expert-compute wall interval;
- E2A: last Worker expert compute completion through Frontend result readiness.

These are per-call work times. They may overlap across micro-batches, so their
shares must not be interpreted as end-to-end wall-time percentages.
