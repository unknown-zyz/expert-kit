# vLLM Expert-Kit 集成验证结果

验证日期：2026-07-19（Asia/Shanghai）
总体状态：**PARTIAL** —— 本地加载、真实 Expert RPC 数值等价和四阶段 trace 通过；模型级严格 token 等价与当前 CPU Worker 性能目标未通过。完整设计与最新结论见 [`ae-dbo-pipeline-design.md`](ae-dbo-pipeline-design.md)。

## 环境

- Python：`.venv/bin/python`，Python 3.12
- vLLM：0.25.1
- Torch：2.11.0，CUDA 13.0，`torch.cuda.is_available() = True`
- GPU：NVIDIA GeForce RTX 5090，32607 MiB
- 模型变量：`QWEN3_30B_A3B_ROOT=$(realpath ./qwen3-30b-a3b)`
- 模型软链接：`qwen3-30b-a3b -> /data/models/huggingface/Qwen/Qwen3-30B-A3B/`
- 模型文件：完整 16 个 safetensors 分片、索引和 tokenizer，共约 57GB
- 配置：`EK_CONFIG=$(realpath ./dev/hello-world.config.yaml)`
- PostgreSQL：通过环境对应的 DSN 覆盖示例占位符；Controller/Worker 可读取现有状态

## 根因与修复

此前本地验证入口使用 `model="Qwen/Qwen3-30B-A3B"`。该字符串不是项目内路径，而是 Hugging Face Hub 仓库 ID。vLLM 的默认加载器用 `os.path.isdir()` 判断它不是本地目录，随后调用 Hub snapshot 下载。`~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B` 当时只有 config、tokenizer、索引和约 13GB 的 `.incomplete` 权重，因此运行停留在 `Starting to load model`。这不是软链接解析失败或本地磁盘加载缓慢。

本次修复：

1. `pipeline_benchmark.py` 在导入 vLLM 前解析并校验 `QWEN3_30B_A3B_ROOT`，把本地绝对路径传给 `LLM`。
2. 设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，防止测试回退到 Hub。
3. 为 vLLM 0.25.1 补充 `GrpcExpert.is_internal_router = False`，修复 `AttributeError: 'GrpcExpert' object has no attribute 'is_internal_router'`，并让 Qwen3MoE 使用已有 gate 生成 router logits。
4. 插件优先读取文档约定的 `EK_MODE`，同时保留 `EXPERTKIT_MODE` 兼容回退。
5. 将 `EK_MODEL_NAME` 从 `qwen3` 修正为配置和数据库中注册的 `qwen3-30b-a3b`。错误名称会导致 Controller 报 `failed to select client for expert qwen3/...`。
6. 测试 RPC timeout 提高到 120 秒，以覆盖 vLLM 启动 profile 和 CPU Worker 的远程 Expert 调用。

## 服务链验证：PASS

启动命令：

```bash
cargo run --release --bin ek-cli -- --config "$EK_CONFIG" weight-server --model "$QWEN3_30B_A3B_ROOT"
cargo run --release --bin ek-cli -- --config "$EK_CONFIG" controller
cargo run --release --bin ek-cli -- --config "$EK_CONFIG" worker
```

实际状态：

- weight-server 监听 `0.0.0.0:6543`
- Controller 监听 `0.0.0.0:5001` 和 `0.0.0.0:5002`
- Worker 监听 `0.0.0.0:51234`，加载 6144 个 experts
- Controller 持续记录 `forward request in controller done`
- Worker 记录 `model=qwen3-30b-a3b` 的真实 expert activation

## vLLM 生成验证：PASS

命令：

```bash
source .venv/bin/activate
timeout 600s python \
  ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py \
  --child-mode pipeline \
  --child-output /tmp/expert-kit-vllm-corrected.json \
  --warmup 0 \
  --repetitions 1 \
  > /tmp/expert-kit-vllm-corrected.log 2>&1
```

关键结果：

```text
model='/data/models/huggingface/Qwen/Qwen3-30B-A3B'
Resolved architecture: Qwen3MoeForCausalLM
Loading safetensors checkpoint shards: 100% Completed | 16/16
Loading weights took 0.38 seconds
🚀 ExpertKitClient Init: ek_addr(localhost:5002), timeout(120s)
init engine (profile, create kv cache, warmup model) took 52.34 s
VLLM_RC=0
ELAPSED_SECONDS=132
```

实际生成文本：

```text
Prompt: 'Hello, my name is'
Generated text: ' Sarah. I have a question about the use of "a" and "an" in English. When should I use "a" and when should I use'

Prompt: 'The president of the United'
Generated text: ' States is the commander-in-chief of the armed forces, and the president is also the head of state. The president is the head of the executive branch of the'
```

## 性能观察

- 第一次显式本地加载 16 个分片耗时 11.10 秒；页缓存命中后的最终运行耗时 0.38 秒。此前 300 秒停顿来自 Hub 未完成下载。
- 引擎 profile、KV cache 创建和 warmup 耗时 52.34 秒。
- 两个 prompt 各生成 32 tokens，生成阶段约 69 秒，约 0.93 output tokens/s；当前 Worker 使用 CPU。
- 约 13GB 的未完成 Hub 缓存及其锁目录已按要求删除；`/data/models` 下的正式权重未受影响。修复后的测试使用离线本地路径，不再创建该模型的 Hub 权重缓存。

## 四阶段调度验证：PASS；模型/性能验收：FAIL

实现基于 vLLM 0.25.1 的 uBatch/DBO 调度器，并把一次远程 MoE 调用拆分为：

```text
A (Attention) -> A2E (CUDA D2H + 序列化/Dispatch) -> E (远程 Expert RPC)
              -> E2A (Combine + pinned-memory H2D)
```

vLLM 补丁不直接修改第三方源码仓库，而由版本和 SHA256 严格校验的脚本应用到当前环境：

```bash
python ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --check
python ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --apply
```

运行命令：

```bash
EK_PIPELINE_ENABLE=1 \
EK_PIPELINE_TRACE=/tmp/expertkit-pipeline-trace.json \
timeout 900s .venv/bin/python \
  ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py \
  --child-mode pipeline \
  --child-output /tmp/expertkit-pipeline-result.json \
  --trace /tmp/expertkit-pipeline-trace.json \
  --warmup 0 \
  --repetitions 1

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/validate_pipeline_trace.py \
  /tmp/expertkit-pipeline-trace.json
```

最终 trace 验证结果：

```json
{
  "calls": 6016,
  "microbatches": [0, 1, 2, 3],
  "compute_comm_overlap": true,
  "concurrent_experts": true
}
```

以下为早期单轮生成记录；它只证明服务链可以完成请求，不作为正式性能或严格等价结论：

```text
Prompt: 'The key idea behind a pipeline is'
Generated text: ' to have a sequence of stages, each of which performs a specific task, and the output of one stage is the input to the next. This allows for efficient'

Prompt: 'In a distant future, humanity'
Generated text: ' has colonized the Moon and Mars, and is now exploring the outer planets. The United Nations has established a space agency, the Interplanetary Exploration Agency ('

Prompt: 'The president of the United'
Generated text: ' States is the commander in chief of the armed forces, and the president has the power to make treaties and appoint ambassadors, which are examples of what type of powers'

Prompt: 'Hello, my name is'
Generated text: ' Sarah. I have a question about the use of "a" and "an" in English. When should I use "a" and when should I use'
```

约束与观察：

- 当前补丁只对至少四请求的 uniform-decode batch 启用四微批；prefill、mixed batch 和不足四请求自动走同步路径，避免 vLLM 0.25.1 的 mixed-batch 切片限制。
- 流水模式当前只支持 Controller 到 Worker 的 gRPC channel；SHM/RDMA 会返回明确错误，不会静默串行化。
- 本次验收证明了阶段重叠和功能正确性，但不宣称吞吐提升：当前 Worker 为 CPU、vLLM 使用 eager 模式，单次结果还包含远程 RPC 与 Python 调度开销；需要 GPU Worker、固定输入和多轮预热基准才能判断收益。
- 正序诊断中，`The key idea behind a pipeline is` 位于第 4 微批时曾出现重复标点；同一服务链的同步基线正常，反序完整运行四条均正常，且独立并发 RPC 与串行结果逐元素最大误差为 0。当前判断更接近调度顺序下的 greedy 解码敏感性，未发现请求串线或 Expert 张量误差；结果文档保留此观察，不能据此宣称 bitwise deterministic。

## 2026-07-20 严格回归与正式基准

- 本地单元测试：`7 passed, 1 skipped`；启用真实服务后：`8 passed`。
- live RPC 使用相同 BF16 hidden states 分别串行和四路并发调用，输出 `torch.equal()` 为真，关联元数据无串线。
- 新 trace：35,454 个完整远程调用、4 个周期快照尾部调用；四 uBatch、A/E 重叠和并发 E 均通过。
- 固定四 prompt、32 tokens、1 次 warmup + 5 次计时、关闭 trace：

| 模式 | 中位延迟 | 中位吞吐 | p95 吞吐 |
|---|---:|---:|---:|
| 同步 | 24.875s | 5.146 tokens/s | 5.247 tokens/s |
| 四阶段流水 | 27.778s | 4.608 tokens/s | 4.652 tokens/s |

流水相对同步吞吐 **-10.45%**，延迟 **+11.67%**，因此当前环境不能宣称性能提升。模型 token 严格等价为 `false`；同步模式自身跨轮也并非 bitwise deterministic，但流水中重复文本退化仍保留为待修复问题。

## 自动检查

```text
pytest ek-integration/expertkit_vllm/tests/test_pipeline.py -q: 4 passed
ruff（本次变更的非生成 Python 文件）: PASS
cargo check -p ek-computation: PASS
cargo build --release --bin ek-cli: PASS
cargo fmt --all -- --check: PASS
vLLM patch --check 和 pristine dry-run: PASS
```

`cargo test -p ek-computation` 在本机被 RDMA/ibverbs 绑定的 ABI size 编译期断言阻塞（多个 `ibv_*` 类型触发 `E0080`），发生在测试二进制编译阶段，不是本次 Controller/Worker 或流水逻辑断言失败。

## 2026-07-20 Nsight Systems 硬件阶段验证

使用 nsys 2026.1.3 同时采集 RTX 5090 上的 Attention CUDA kernels、D2H/H2D memcpy、Worker CPU context switches 和跨进程 NVTX。分析器只统计正式 `EK_PROFILE_WINDOW`，自动选中稳定 decode 的 layers 18–19、四个 uBatch、八个完整调用。

| 阶段 | 平均时间 | 通信掩盖率 | 暴露通信（八调用合计） |
|---|---:|---:|---:|
| A | 2.098ms | — | — |
| A2E | 6.073ms | 98.9% | 0.526ms |
| E | 3.543ms | — | — |
| E2A | 1.417ms | 98.3% | 0.198ms |
| 总通信 | — | **98.8%** | **0.724ms** |

状态：**PASS**。完整 profile window 内其他 uBatch 的 Attention/Expert 计算对选中 layers 18–19 的通信掩盖超过预设的 95% 标准；仅使用图中两个 layer 的可见计算时为 91.1%，是有限画布的保守下界。纵向四阶段图见 [`assets/ae-dbo-nsys-four-stage-overlap.svg`](assets/ae-dbo-nsys-four-stage-overlap.svg)，完整口径、命令和 nsys 2026.1 的 forkserver/telemetry 规避说明见 [`ae-dbo-pipeline-design.md`](ae-dbo-pipeline-design.md)。
