# vLLM 四阶段流水与 Python Worker 适配

## 1. 总结

本实现将 Expert Kit 的四 uBatch 调度适配到 `dev-py` 的 v2 架构：

```text
vLLM A -> Expert Kit Transport A2E -> Python Worker E -> Transport/vLLM E2A
```

与旧的 Rust v1 路径不同，Controller 只向 Frontend 发布 Worker topology，不代理
计算。vLLM 的 `RemoteMoERunner` 直接向 `ek-transport` 提交 routed layer；Transport
再按 topology 拆成 physical Worker batches，并通过 gRPC 或 SHM 并发发送到 Python
Worker。

流水启用后，一个 uBatch 等待 Transport Future 时会释放 vLLM 模型执行权，其他
ready uBatch 可以继续执行 Attention 和 Router。Transport 的私有 asyncio loop
同时推进各 uBatch 的 staging、Worker RPC 和聚合。DeepSeek-V2-Lite 的真实正确性、
吞吐和 nsys 结果见
[`benchmarks/deepseek-v2-lite-four-stage.md`](benchmarks/deepseek-v2-lite-four-stage.md)。

## 2. 四阶段边界

| 阶段 | `dev-py` 中的工作 |
|---|---|
| A | vLLM Attention、本地 gate、Router/Top-k 和 routing validation |
| A2E | topology grouping、buffer admission、D2H staging、编码及 Worker dispatch |
| E | Python Worker 等待固定 execution slot、输入准备和 Backend expert computation |
| E2A | Worker response、Frontend decode/H2D、FP32 partial aggregation、shared expert 与 scale/add |

`BlockingRoutedMoEClient.submit_execute()` 返回 `RoutedMoECall`。其
`future` 在私有 asyncio loop 中运行完整 A2E/E/Transport E2A；vLLM 调用补丁新增的
`dbo_wait_for_future()`。Future 完成后，`RoutedMoECall.result()` 在恢复后的模型
线程上把 output CUDA event 安装到当前 stream，避免把依赖错误地安装到 Transport
线程的 CUDA stream。

原 `BlockingRoutedMoEClient.execute()` 保持公开行为不变，内部实现为 submit 后
立即 result。未开启流水、非 uBatch、prefill、mixed batch 或请求不足时仍可按同步
语义执行。

## 3. 并发控制

Frontend 并发由四个 vLLM uBatch 和各 Worker topology capacity 共同约束。Python
Worker 的真实计算并发由 Worker YAML 控制：

```yaml
worker:
  max_active_batches_per_device: 4

transport:
  max_pending_batches_per_device: 4
```

`max_active_batches_per_device` 创建相同数量的 `ExecutionSlot`、执行线程和独立 CUDA
stream；Backend 必须声明 `supports_concurrent_batches=True`。Torch 和 fused Backend
支持多 slot，GGML Backend 会拒绝大于 1 的 active-batch 配置。pending 参数只控制
Transport 等待区，不增加同时计算数量。

本分支不使用 `EK_WORKER_THREADS`。该环境变量属于旧 Rust SHM/RDMA Worker，不能
控制 Python Worker。

## 4. 安装与启用

Python Worker、Controller、Weight Server、数据库和 expert placement 按
[`tutorial/standalone/qwen3-moe-a3b-demo.md`](tutorial/standalone/qwen3-moe-a3b-demo.md)
准备。vLLM integration 仍固定为 vLLM 0.25.1：

```bash
source .venv/bin/activate
pip install -e ek-proto
pip install -e ek-transport
pip install -e ek-integration/expertkit_vllm

python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --apply
python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --check
```

运行 Frontend 前设置：

```bash
export EK_ENABLE=1
export EK_PIPELINE_ENABLE=1
export EK_ADDR=127.0.0.1:5002
export EK_INSTANCE_ID=1
export EK_CLIENT_TIMEOUT=120
```

vLLM 当前按 eager、四 uBatch 验证：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/absolute/path/to/Qwen3-30B-A3B",
    enforce_eager=True,
    ubatch_size=4,
    dbo_decode_token_threshold=0,
    dbo_prefill_token_threshold=0,
    max_model_len=256,
)
outputs = llm.generate(
    ["Prompt 0", "Prompt 1", "Prompt 2", "Prompt 3"],
    SamplingParams(temperature=0, max_tokens=16, seed=0),
    use_tqdm=False,
)
```

不要同时设置 `enable_dbo=True` 和 `ubatch_size=4`；vLLM 0.25.1 对前者采用固定双
uBatch 解释。流水补丁只对 single-DP uniform decode 放宽 uBatch 条件，不改变
prefill/mixed-batch 调度。

## 5. 补丁和兼容性

补丁修改 vLLM 0.25.1 的五个文件：

| 文件 | 修改 |
|---|---|
| `vllm/v1/worker/ubatching.py` | completion-driven ready queue、Future wait、失败广播 |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 子线程异常传播、single-DP metadata |
| `vllm/v1/worker/gpu_model_runner.py` | single-DP uniform decode 切分 |
| `vllm/v1/worker/gpu_worker.py` | 按显式 `num_ubatches` 分配 workspace，而非固定 1/2 份 |
| `vllm/config/vllm.py` | 外部 Expert pipeline 不要求原生 All-to-All backend |

plugin 在 `EK_PIPELINE_ENABLE=1` 时检查安装版本、patch version 和
`dbo_wait_for_future`，并强制 `VLLM_USE_V2_MODEL_RUNNER=0`。原因是 vLLM
0.25.1 会自动选择新的 V2 Model Runner，而当前补丁修改的是 legacy
`v1/worker/gpu_model_runner.py`；不固定 runner 会出现参数显示 `ubatch_size=4`
但实际只有 uBatch 0 的静默失效。用户显式设置 V2 时 plugin 直接报错。
补丁脚本对五个文件校验 SHA256；部分 patch 或未知
site-packages 修改会直接失败。恢复原版：

```bash
python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --reverse
```

v2 computation Proto 没有为流水新增 request ID。每个 direct Worker unary/SHM call
由自己的 asyncio Future 关联响应，核心正确性不依赖跨进程 profile metadata。

## 6. 正确性验证

```bash
# vLLM integration、调度和 plugin
pytest ek-integration/expertkit_vllm/tests -q

# Transport blocking/submit Future
pytest \
  ek-transport/tests/unit/transports/grpc/test_routed_client.py -q

# Python Worker 1/4 execution slots
pytest \
  ek-worker/tests/integration/execution/test_executor.py \
  -q -k fixed_slots
```

测试覆盖：

- 四个 uBatch 先依次进入 A，再按 Transport Future 的真实完成顺序恢复；
- 任一 uBatch 失败会广播给 sibling，线程不会永久等待；
- submit API 允许多个 routed layer Future 同时在途；
- blocking API、deadline、cancel 和 close 保持兼容；
- Worker active slots 为 1/4 时，实际 Backend 并发不超过配置且所有输出逐元素一致。

本分支已在 RTX 5090 + CPU Worker 上完成 DeepSeek-V2-Lite batch 4–64 验收。服务链
和冒烟 token 等价通过，但完整 sync 基线自身不是 bitwise deterministic，且 pipeline
吞吐未提升，因此不能声明严格模型 token 等价或性能收益；详细证据和复现命令见上方
结果文档。
