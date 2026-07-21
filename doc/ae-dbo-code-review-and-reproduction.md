# AE-DBO 四阶段流水：代码 Review 与实验复现手册

本文面向需要 Review `a7dcc9f`（`feat: add four-stage ExpertKit vLLM
pipeline`）或重新运行实验的开发者。设计原理和已有性能分析分别见
[`ae-dbo-pipeline-design.md`](ae-dbo-pipeline-design.md) 和
[`ae-dbo-profile-comparison.md`](ae-dbo-profile-comparison.md)；本文重点回答：

1. 每个提交文件为什么修改，Review 时应关注什么；
2. 当前脚本实际执行什么、输入输出是什么；
3. 如何从环境准备开始复现正确性、trace、nsys 和性能实验。

> 当前实现已经验证四阶段调度、跨进程关联和 Expert RPC 数值等价，但模型级
> strict token equivalence 仍为 **FAIL**。性能结果不能替代正确性验收。

## 1. 一次请求如何流过四个阶段

四阶段的逻辑边界为：

```text
vLLM uBatch thread
  A   : 上一层 E2A 结束后的本地计算、Attention、Router/Top-k、路由准备
  A2E : GPU→CPU、safetensors 序列化、Client→Controller gRPC、请求拆分
  E   : Controller→Worker、Worker 排队、exp.forward()、Expert 返回
  E2A : Controller 聚合、Client 反序列化、CPU→GPU、weighted combine
```

`GrpcExpert.forward()` 为每次 MoE 调用生成稳定的 `request_id`，并从 vLLM DBO
上下文读取 `microbatch_id`。客户端先异步提交远程调用，然后通过补丁新增的
`dbo_wait_for_future()` 释放当前 uBatch 的 Python 模型执行权。调度器选择其他
ready uBatch 推进；远程 Future 完成后，原 uBatch 重新进入 ready queue，恢复
forward context 和 CUDA stream，再执行 E2A。

请求携带 `request_id/microbatch_id/layer_id` 穿过 Client、Controller 和 Worker，
响应返回相同字段。客户端在接收张量前逐项校验，避免并发响应串线。Controller
对 `pipeline_enabled=true` 的请求创建 request-local executor，不再让远程等待
占住旧的全局 executor mutex。当前 request-local 路径只支持 gRPC Worker；SHM
和 RDMA 仍使用原同步路径。

## 2. 推荐 Review 顺序

建议按“协议 → vLLM 调度 → Python Client → Controller/Worker → 可观测性和测试”
阅读，而不是按 Git 文件名排序。

### 2.1 协议和兼容性

`ek-proto/ek/worker/v1/expert.proto` 在 proto3 消息尾部追加字段：

| 消息 | 字段 | 作用 |
|---|---|---|
| `ForwardReq` | `request_id` | 唯一关联一次 Client→Controller Expert 调用 |
| `ForwardReq` | `microbatch_id` | 标识 vLLM uBatch |
| `ForwardReq` | `layer_id` | 标识 MoE layer |
| `ForwardReq` | `pipeline_enabled` | 选择 Controller request-local executor |
| `ForwardResp` | `request_id/microbatch_id/layer_id` | 客户端校验响应关联 |

字段号只追加不复用，旧客户端不发送时得到 proto3 默认值。`expertkit_torch` 和
`expertkit_vllm` 都内置同一个 protobuf descriptor，因此两套 `pb2/pb2_grpc/pyi`
必须同步生成。Torch 包没有加入 vLLM 流水业务逻辑；其修改只是共享协议生成物。

Review 重点：字段号不能更改；所有构造 `ForwardResp` 的服务（包括 mock server）
都必须回传关联字段；同步路径的默认字段不能触发误判。

### 2.2 vLLM 0.25.1 补丁

仓库不直接跟踪 `.venv/site-packages/vllm`，修改保存在
`ek-integration/expertkit_vllm/patches/vllm-0.25.1-expertkit-pipeline.patch`：

| vLLM 文件 | 修改目的 | Review 重点 |
|---|---|---|
| `vllm/v1/worker/ubatching.py` | 用 completion-driven ready queue 替换固定 ring；增加 Future suspend/resume 和 sibling failure broadcast | 任意时刻最多一个 Python model thread active；Future 竞态不丢唤醒；异常不会让其他线程死锁 |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 收集 uBatch 子线程异常并回抛；允许 single-DP metadata | 结果仍按 uBatch ID 拼接；失败路径能够结束所有线程 |
| `vllm/v1/worker/gpu_model_runner.py` | single-DP uniform decode 也可按阈值切 uBatch | prefill、mixed batch、请求不足时不强行切分 |
| `vllm/config/vllm.py` | 外部 Expert pipeline 时解除 DeepEP/NIXL All-to-All backend 限制 | 仅 `EK_PIPELINE_ENABLE=1` 放宽，不影响普通 vLLM DBO |
| `vllm/model_executor/models/qwen3_moe.py` | 为 Qwen3 Attention launch 增加 layer/uBatch NVTX | 仅 `EK_NSYS_PROFILE=1` 生效，不进入正常性能路径 |

补丁固定版本是 vLLM `0.25.1`。`plugin.py` 在启用流水时同时检查发行版版本和
`dbo_wait_for_future` 是否存在，避免在未打补丁的环境静默运行错误调度。

查看补丁和当前安装源码：

```bash
less ek-integration/expertkit_vllm/patches/vllm-0.25.1-expertkit-pipeline.patch

VLLM_PACKAGE_ROOT=$(
  .venv/bin/python -c \
  'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)'
)
rg -n \
  'UBatchCoordinator|dbo_wait_for_future|external_expert_pipeline' \
  "$VLLM_PACKAGE_ROOT"
```

补丁脚本先用 SHA256 判断五个文件是全部 pristine 还是全部 patched；未知内容或
部分应用会中止，不会覆盖本地 vLLM 修改：

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --check
# 修改 site-packages：
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --apply
# 恢复原版：
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --reverse
```

### 2.3 Python Expert 与异步 gRPC

`grpc_expert.py` 把原阻塞 Expert 调用拆成 submit/wait/combine：

- Router/Top-k 和路由准备仍在当前 uBatch 的模型线程执行；
- `submit_forward_expert()` 把 CUDA hidden state 异步复制到 pinned host buffer；
- 四线程 Python executor 负责等待 D2H event、序列化和阻塞 gRPC；
- `a2e_future` 标记输入可序列化的时刻，主 Future 返回 Expert 输出；
- 当前 uBatch 等待 Future 时由 completion-driven scheduler 让出执行权；
- 返回张量先校验关联字段，再 pin memory、H2D，最后 weighted combine。

`grpc_client.py` 保留 `forward_expert()` 阻塞兼容接口；它内部也调用新的异步接口，
但设置 `pipeline_enabled=false`。`PIPELINE_WORKERS=4` 是 vLLM Client 侧并发提交槽，
不是 Worker Expert 计算并发上限。

`pipeline_trace.py` 使用 `perf_counter_ns()` 生成 Chrome trace，适合验证阶段顺序和
逻辑重叠；每 4096 个事件原子快照一次，以应对 EngineCore 被 signal 结束时不运行
`atexit`。`profile.py` 只在 `EK_NSYS_PROFILE=1` 时发出细粒度 Python NVTX。

Review 重点：D2H event 必须在序列化前完成；response metadata 不匹配必须失败；
gRPC 异常必须同时完成/失败相关 Future；trace 不能默认开启。

### 2.4 Controller 与 Worker

Controller 的 pipeline 分支为每个 Client 请求建立独立 `NaiveExecutor`，把请求
元数据保存在 ingress state，按 Expert 聚合 token 后并发调用 Worker，再按原 token
位置聚合输出。普通同步请求继续使用共享 executor。流水模式遇到 Expert 选择失败
会直接返回错误；SHM/RDMA 会明确拒绝，避免部分结果或永久等待。

Worker gRPC service 将同步 `gate_sync.forward_sync()` 放进 Tokio
`spawn_blocking()`。新增 NVTX 将排队、Expert lookup、safetensors load、输入设备
转换、Expert math 和输出序列化分开。`nvtx_shim.c` 在系统有 NVTX header 时调用
NVTX C API；没有 header 时编译为空操作，所以 profiling 依赖缺失不会阻止构建。

这里有一个重要的参数边界：

- `EK_WORKER_PARALLEL` 传给 `tch::set_num_threads()`，控制单个 Torch CPU op 的
  intra-op 线程数；GGML backend 也读取它；
- `EK_WORKER_THREADS` 只在 `worker/mod.rs` 的 SHM 和 RDMA 消费循环中读取；
- gRPC 分支使用 Tokio `spawn_blocking()`，当前没有读取 `EK_WORKER_THREADS`。

因此 `run_nsys_pipeline_profile.py --worker-threads N` 虽然会导出
`EK_WORKER_THREADS=N`，但对本实验使用的 gRPC Worker 没有确定的并发控制作用。
历史文档中的 “Worker×1/Worker×4” 应理解为对应报告中**实际观测到的最大 Expert
并发**，不能仅凭该 CLI 参数建立因果关系。要做严格线程扫描，需要先给 gRPC 路径
增加显式 semaphore/专用 blocking pool；这不属于本次文档修改。

### 2.5 可观测性口径

三类标记使用不同前缀：

| 前缀 | 进程/来源 | 示例 |
|---|---|---|
| `EK:` | 四阶段或 Worker Expert | `EK:A:...`、`EK:E:...:expert=e3` |
| `EKC:` | Python Client 细分 | `CLIENT_CONTROLLER_GRPC`、`H2D_COPY` |
| `EKR:` | Rust Controller/Worker 细分 | `WORKER_QUEUE`、`WORKER_EXPERT_MATH` |

Chrome trace 中的 A/E 是逻辑墙钟阶段，不能证明 GPU/CPU 硬件同时运行。nsys 分析器
把 Attention NVTX 投影到 CUDA kernel，把 Worker Expert NVTX 与 OS scheduled-in
区间相交，才作为硬件计算；A2E/E2A 是两者之间的非计算关键路径。图为了表达完整
MoE 阶段，会把 Router、Top-k 和 weighted combine 归入 A/E 方块，但定量掩盖率
只统计可投影到硬件 activity 的区间。

## 3. 提交文件逐项索引

下表覆盖提交 `a7dcc9f` 的全部 47 个文件。生成物和结果资产也列出，便于确认它们
是否应随源文件变化而重新生成。

| 文件 | 具体改动与 Review 目的 |
|---|---|
| `Cargo.toml` | workspace 增加 `cc`，供 NVTX C shim 构建 |
| `Cargo.lock` | 锁定新增 build dependency 的解析结果 |
| `ek-computation/Cargo.toml` | 为该 crate 引入 workspace `cc` build dependency |
| `ek-computation/build.rs` | 编译 NVTX shim；查找 CUDA/NVTX include；找不到 header 时降级为空操作 |
| `ek-computation/src/bin/mock_server.rs` | mock 响应回显 request/uBatch/layer metadata |
| `ek-computation/src/controller/executor.rs` | request-local pipeline executor、关联元数据、gRPC 并发错误传播及 Controller NVTX |
| `ek-computation/src/controller/service/compute.rs` | 按 `pipeline_enabled` 分流共享 executor 与 request-local executor |
| `ek-computation/src/ffn/mod.rs` | 标记 Worker 输入转换、设备传输、Expert math 和输出序列化 |
| `ek-computation/src/worker/core.rs` | 透传响应 metadata；标记 lookup/load/完整 Expert 调用 |
| `ek-computation/src/worker/mod.rs` | 注册 `profile` 模块；原有 SHM/RDMA 线程逻辑未改 |
| `ek-computation/src/worker/nvtx_shim.c` | 封装 push/pop 和 async start/end；无 NVTX 时提供 no-op ABI |
| `ek-computation/src/worker/profile.rs` | Rust RAII NVTX guards；Drop 保证 range 配对 |
| `ek-computation/src/worker/server.rs` | 标记 gRPC 请求和 `spawn_blocking` 排队；打印关联 metadata |
| `ek-proto/ek/worker/v1/expert.proto` | 新增 request/uBatch/layer/pipeline 协议字段 |
| `ek-integration/expertkit_torch/expertkit_torch/pbpy/ek/worker/v1/expert_pb2.py` | 共享 proto 的 Python runtime 生成物 |
| `ek-integration/expertkit_torch/expertkit_torch/pbpy/ek/worker/v1/expert_pb2.pyi` | 共享 proto 的类型提示生成物 |
| `ek-integration/expertkit_torch/expertkit_torch/pbpy/ek/worker/v1/expert_pb2_grpc.py` | 共享 gRPC stub 生成物；没有流水业务逻辑 |
| `ek-integration/expertkit_vllm/expertkit_vllm/pbpy/ek/worker/v1/expert_pb2.py` | vLLM 包内相同 proto runtime 生成物 |
| `ek-integration/expertkit_vllm/expertkit_vllm/pbpy/ek/worker/v1/expert_pb2.pyi` | vLLM 包内相同类型提示生成物 |
| `ek-integration/expertkit_vllm/expertkit_vllm/pbpy/ek/worker/v1/expert_pb2_grpc.py` | vLLM 包内相同 gRPC stub 生成物 |
| `ek-integration/expertkit_vllm/expertkit_vllm/experts/grpc_expert.py` | 将 MoE remote Expert 改为四阶段 submit/yield/resume/combine |
| `ek-integration/expertkit_vllm/expertkit_vllm/grpc_client.py` | pinned D2H、异步请求池、关联校验、H2D 和阻塞兼容 API |
| `ek-integration/expertkit_vllm/expertkit_vllm/pipeline_trace.py` | Chrome trace 收集、周期快照和 NVTX 四阶段标记 |
| `ek-integration/expertkit_vllm/expertkit_vllm/profile.py` | Client 细粒度 NVTX context manager |
| `ek-integration/expertkit_vllm/expertkit_vllm/plugin.py` | 流水启用时检查 vLLM 版本和补丁 API |
| `ek-integration/expertkit_vllm/expertkit_vllm/utils/config.py` | 读取 `EK_PIPELINE_ENABLE` 和 `EK_PIPELINE_TRACE` |
| `ek-integration/expertkit_vllm/setup.py` | 将 vLLM 依赖固定为 `0.25.1` |
| `ek-integration/expertkit_vllm/patches/vllm-0.25.1-expertkit-pipeline.patch` | 可审计、可应用/回退的 vLLM 源码统一补丁 |
| `ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py` | 固定 workload；隔离 sync/pipeline 进程；保存性能、文本和 token IDs |
| `ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py` | 基于版本和 hash 安全检查/应用/回退补丁 |
| `ek-integration/expertkit_vllm/scripts/run_nsys_pipeline_profile.py` | 独占端口启动三个服务、运行单模式 workload、反向停止并保存日志 |
| `ek-integration/expertkit_vllm/scripts/validate_pipeline_trace.py` | 检查四阶段完整性、四个 uBatch、计算/通信重叠和并发 in-flight E |
| `ek-integration/expertkit_vllm/scripts/render_pipeline_trace.py` | 从 Chrome trace 选取真实窗口并生成时间比例 SVG |
| `ek-integration/expertkit_vllm/scripts/analyze_nsys_pipeline.py` | 导出/读取 nsys SQLite，关联 NVTX/CUDA/OSRT，计算硬件掩盖率 |
| `ek-integration/expertkit_vllm/scripts/render_nsys_pipeline.py` | 将两个相邻层按 A/A2E/E/E2A 四行绘制为 SVG |
| `ek-integration/expertkit_vllm/scripts/analyze_nsys_comparison.py` | 跨进程 request-ID 关联三种报告，分解 RPC/queue/math/memcpy 并合并 benchmark |
| `ek-integration/expertkit_vllm/scripts/render_nsys_comparison.py` | 绘制同步与流水关键路径对比图 |
| `ek-integration/expertkit_vllm/tests/test_pipeline.py` | 异步协议、调度竞态、失败广播、trace/nsys 分析和 live RPC 回归测试 |
| `ek-integration/expertkit_vllm/README.md` | 增加补丁、启用条件、trace、nsys 和 benchmark 快速入口 |
| `pyproject.toml` | 开发依赖增加 pytest |
| `uv.lock` | 同步 Python 依赖锁文件 |
| `doc/ae-dbo-pipeline-plan.md` | 保留实施前计划，并标明与最终四 uBatch 实现的差异 |
| `doc/ae-dbo-pipeline-design.md` | 最终架构、阶段定义、源码修改、正确性和性能总结 |
| `doc/ae-dbo-profile-comparison.md` | 三份 profile 的跨进程指标与瓶颈分析 |
| `doc/vllm-expertkit-test-result.md` | vLLM 集成环境、加载路径、服务链和实际测试状态记录 |
| `doc/assets/ae-dbo-nsys-four-stage-overlap.svg` | 由 nsys analysis JSON 生成的四行两层时间图 |
| `doc/assets/ae-dbo-sync-vs-pipeline-profile.svg` | 由 comparison JSON 生成的模式对比图 |

## 4. 实验前准备

### 4.1 环境和模型

以下命令均从仓库根目录运行：

```bash
source .venv/bin/activate
export QWEN3_30B_A3B_ROOT="$(realpath ./qwen3-30b-a3b)"
export EK_CONFIG="$(realpath ./dev/hello-world.config.yaml)"

test -f "$QWEN3_30B_A3B_ROOT/config.json"
nvidia-smi
.venv/bin/python - <<'PY'
import importlib.metadata
import torch
print("vllm", importlib.metadata.version("vllm"))
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

`Path.resolve(strict=True)` 会解析模型软链接，vLLM 收到的是 `/data/models/...` 的
本地绝对路径，不会把 `Qwen/Qwen3-30B-A3B` 当作项目内相对路径，也不会在离线模式
下从 Hugging Face 下载。

确认 `EK_CONFIG` 的数据库 DSN 在当前机器可达，且配置/Inventory 使用 gRPC：

```bash
rg -n 'db_dsn|channel|device|6543|5001|5002|51234' \
  "$EK_CONFIG" dev/local.inventory.yaml
ss -ltn | rg ':(5432|6543|5001|5002|51234)\b' || true
```

不要把机器专用数据库地址、凭据、模型权重或 `/tmp` profile 报告提交到仓库。

### 4.2 构建、数据库和插件

```bash
uv sync --extra cu130
uv pip install vllm==0.25.1
uv pip install --no-deps -e ek-integration/expertkit_vllm

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --apply
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/apply_vllm_pipeline_patch.py --check

cargo build --release --bin ek-cli
target/release/ek-cli --config "$EK_CONFIG" db migrate
target/release/ek-cli --config "$EK_CONFIG" \
  model upsert --name qwen3-30b-a3b
target/release/ek-cli --config "$EK_CONFIG" \
  schedule static --inventory dev/local.inventory.yaml
```

迁移、模型注册和静态调度是持久数据库状态。统一实验启动器不会替用户猜测或重写
这些状态；它只启动 weight-server、Controller 和 Worker。

## 5. 当前脚本的职责和参数

### 5.1 `pipeline_benchmark.py`

父模式（不传 `--child-mode`）依次创建两个独立 Python/vLLM 进程运行 sync 和
pipeline，最终输出统一 JSON。固定输入是四条 prompt；`temperature=0`、`seed=0`，
但当前完整服务链仍非 bitwise deterministic。

```bash
timeout 1800s .venv/bin/python \
  ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py \
  --warmup 1 --repetitions 5 --max-tokens 32 \
  --output /tmp/expertkit-pipeline-benchmark.json
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--output` | `/tmp/expertkit-pipeline-benchmark.json` | 父模式综合结果 |
| `--trace` | 无 | 仅 pipeline child 写 Chrome trace；开启后不用于性能结论 |
| `--warmup` | `1` | 每个模式不计时 warmup 次数 |
| `--repetitions` | `5` | 每个模式测量次数 |
| `--max-tokens` | `32` | 每条 prompt 最大输出 token 数 |
| `--nsys-capture` | 关闭 | 首次测量外包围 `EK_PROFILE_WINDOW` |
| `--child-mode` | 无 | 内部/单模式使用：`sync` 或 `pipeline` |
| `--child-output` | 无 | child 模式必填的 JSON 路径 |

输出包含环境版本、模型解析路径、每轮延迟/吞吐、文本、token IDs、模式内稳定性、
sync/pipeline 首个分歧 token 以及吞吐/延迟变化。父模式要求服务已提前启动；它不会
启动或停止 Expert-Kit。

### 5.2 `run_nsys_pipeline_profile.py`

该启动器要求四个服务端口空闲，按 weight-server → Controller → Worker 启动，
等待端口后运行 benchmark child，最后按反序停止自己创建的进程。异常时服务日志
保留在 `--log-dir`。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--config` | 必填 | Expert-Kit YAML，启动前解析为绝对路径 |
| `--model` | 必填 | 模型目录或软链接，启动前严格解析 |
| `--mode` | `pipeline` | 单次运行 sync 或 pipeline |
| `--worker-threads` | `1` | 导出 `EK_WORKER_THREADS`；**当前不控制 gRPC 并发** |
| `--warmup` | `1` | benchmark warmup |
| `--repetitions` | `1` | benchmark 测量次数 |
| `--max-tokens` | `4` | 每条 prompt 输出长度 |
| `--no-profile-window` | 关闭 | 关闭 NVTX 标记，用于正式性能测试 |
| `--log-dir` | `/tmp/ek-nsys-logs` | 三个服务的 stdout/stderr |
| `--benchmark-output` | `/tmp/ek-nsys-workload.json` | 单模式 benchmark JSON |

它还固定设置 `EK_WORKER_PARALLEL=1`、离线模型模式和 120 秒 RPC timeout。它不会
启动 PostgreSQL、执行迁移、注册模型或静态调度。

### 5.3 分析、验证和绘图脚本

| 脚本 | 输入 | 输出/成功条件 |
|---|---|---|
| `validate_pipeline_trace.py TRACE` | Chrome trace JSON | stdout JSON；阶段完整、四 uBatch、有 compute/comm overlap、有 concurrent in-flight E |
| `render_pipeline_trace.py TRACE SVG` | Chrome trace JSON | 真实逻辑墙钟窗口 SVG |
| `analyze_nsys_pipeline.py REP JSON` | `.nsys-rep` 或 `.sqlite` | 四阶段调用、所选层、硬件掩盖率 JSON；REP 会先导出临时 SQLite |
| `render_nsys_pipeline.py JSON SVG` | 单份 nsys analysis | 四行、两个相邻层的硬件时间图 |
| `analyze_nsys_comparison.py ...` | 三份 REP/SQLite，可选三份 benchmark JSON | 跨进程请求分解和模式对比 JSON |
| `render_nsys_comparison.py JSON SVG` | comparison JSON | 关键路径对比 SVG |

SVG renderer 只消费 JSON，不重新解释原始 nsys 数据；更改统计口径应 Review analyzer，
更改布局和标签才 Review renderer。

## 6. 正确性实验

### 6.1 无服务单元测试

```bash
.venv/bin/python -m pytest \
  ek-integration/expertkit_vllm/tests/test_pipeline.py -q
```

覆盖异步 tensor/metadata、阻塞兼容接口、response mismatch、RPC failure、Future
完成顺序、sibling abort、Chrome trace validator/renderer、nsys analyzer/renderer
以及三模式 comparison。live test 在未设置开关时应显示 skipped。

### 6.2 真实 Expert RPC 数值等价

先用三个独立终端启动服务：

```bash
# Terminal 1
target/release/ek-cli --config "$EK_CONFIG" \
  weight-server --model "$QWEN3_30B_A3B_ROOT"

# Terminal 2
target/release/ek-cli --config "$EK_CONFIG" controller

# Terminal 3
target/release/ek-cli --config "$EK_CONFIG" worker
```

端口就绪后：

```bash
EK_PIPELINE_LIVE_TEST=1 \
QWEN3_30B_A3B_ROOT="$QWEN3_30B_A3B_ROOT" \
EK_MODEL_NAME=qwen3-30b-a3b \
EK_ADDR=localhost:5002 \
EK_CLIENT_TIMEOUT=120 \
.venv/bin/python -m pytest \
  ek-integration/expertkit_vllm/tests/test_pipeline.py \
  -q -k live_pipeline_rpc
```

该测试构造 batch size 1/2/3/4 的固定 BF16 hidden states，对同一组 Experts 先串行
再四路并发调用，要求 shape、dtype 和 `torch.equal()` 全部一致。它不加载 vLLM
完整模型，因此不能替代 token 等价验证。

### 6.3 模型级 token 等价和性能

服务保持运行，执行 5 轮父模式 benchmark。检查：

```bash
.venv/bin/python - <<'PY'
import json
p = json.load(open('/tmp/expertkit-pipeline-benchmark.json'))
print(json.dumps(p['comparison'], indent=2))
PY
```

`strict_token_equivalence=true` 才表示 sync 第一轮与每个 pipeline 轮次的每条输出
token IDs 完全相同；`sync_within_mode_stability` 和
`pipeline_within_mode_stability` 分别判断各模式自身是否稳定。当前历史结果中同步
自身也不稳定，严格 token 等价为 **FAIL**；应记录首个分歧位置，不要只比较文本。

## 7. Trace 与绘图实验

### 7.1 Chrome trace：验证调度关系

Chrome trace 是低成本逻辑 trace，不是精确硬件计时。服务已启动时运行：

```bash
EK_PIPELINE_ENABLE=1 \
EK_PIPELINE_TRACE=/tmp/expertkit-pipeline-trace.json \
timeout 900s .venv/bin/python \
  ek-integration/expertkit_vllm/benchmark/pipeline_benchmark.py \
  --child-mode pipeline \
  --child-output /tmp/expertkit-pipeline-result.json \
  --trace /tmp/expertkit-pipeline-trace.json \
  --warmup 0 --repetitions 1 --max-tokens 4

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/validate_pipeline_trace.py \
  /tmp/expertkit-pipeline-trace.json

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/render_pipeline_trace.py \
  /tmp/expertkit-pipeline-trace.json \
  /tmp/ae-dbo-four-stage-overlap.svg
```

validator 的 `concurrent_experts` 表示多个远程 E Future 同时 in-flight，不等于多个
CPU Expert 同时获得硬件执行时间。后者必须查看 nsys 的 Worker scheduled-in 与
`EK:E` 交集。

### 7.2 nsys：验证硬件计算能否掩盖通信

先确认 `nsys --version`，并让 6543/5001/5002/51234 空闲。profile 使用四 prompt、
warmup 1、测量 1、每条 8 tokens；正式性能测试使用每条 32 tokens 且不开 nsys。

同步 profile：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --sample=none --cpuctxsw=process-tree \
  --cuda-event-trace=true --resolve-symbols=false \
  --wait=primary --force-overwrite=true \
  --output=/tmp/expertkit-sync-detailed \
  .venv/bin/python \
  ek-integration/expertkit_vllm/scripts/run_nsys_pipeline_profile.py \
  --config "$EK_CONFIG" \
  --model "$QWEN3_30B_A3B_ROOT" \
  --mode sync --worker-threads 1 \
  --warmup 1 --repetitions 1 --max-tokens 8 \
  --log-dir /tmp/ek-nsys-sync-logs \
  --benchmark-output /tmp/ek-nsys-sync-workload.json
```

流水 profile 至少再执行一次，把 `--mode` 改为 `pipeline`，并使用不同的 output、
log-dir 和 benchmark-output。若要比较不同 gRPC Worker 并发上限，当前脚本不足以
构造严格对照；不要只修改 `--worker-threads` 就宣称线程数实验。

生成单份四阶段分析和图：

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/analyze_nsys_pipeline.py \
  /tmp/expertkit-pipeline-detailed.nsys-rep \
  /tmp/expertkit-four-stage-analysis.json

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/render_nsys_pipeline.py \
  /tmp/expertkit-four-stage-analysis.json \
  /tmp/ae-dbo-nsys-four-stage-overlap.svg
```

`--trace-fork-before-exec=true` 用于捕获 forkserver 创建的 EngineCore Python NVTX。
脚本设置 `VLLM_NO_USAGE_STATS=1`，规避 nsys 2026.1 退出阶段辅助进程问题。分析器
只统计 `EK_PROFILE_WINDOW`；模型加载和 warmup 虽在原始报告中，但不进入结果。

### 7.3 三模式对比和历史结果解释

有三份报告和对应无 profile benchmark 后：

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/analyze_nsys_comparison.py \
  --sync /tmp/expertkit-sync-detailed.nsys-rep \
  --pipeline-serial /tmp/expertkit-pipeline-serial-detailed.nsys-rep \
  --pipeline-parallel /tmp/expertkit-pipeline-parallel-detailed.nsys-rep \
  --benchmark-sync /tmp/expertkit-benchmark-sync.json \
  --benchmark-pipeline-serial /tmp/expertkit-benchmark-pipeline-serial.json \
  --benchmark-pipeline-parallel /tmp/expertkit-benchmark-pipeline-parallel.json \
  --output /tmp/expertkit-nsys-comparison.json

.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/render_nsys_comparison.py \
  /tmp/expertkit-nsys-comparison.json \
  /tmp/ae-dbo-sync-vs-pipeline-profile.svg
```

历史数据为 sync 5.405 tokens/s、pipeline/观测并发 1 为 5.089 tokens/s、
pipeline/观测并发 4 为 6.470 tokens/s，后者相对同步为 `+19.70%`。这些报告说明
“出现四路实际 Expert 并发时吞吐更高”，但由于当前 gRPC 启动器没有用
`--worker-threads` 锁定并发上限，不能把提升严格归因于该参数。复现报告应同时记录
analyzer 的 `expert_max_concurrency`、Worker queue、Expert math、CPU affinity/NUMA
和每次运行的原始命令。

## 8. 查看 nsys 结果

有桌面环境时：

```bash
nsys-ui /tmp/expertkit-pipeline-detailed.nsys-rep
```

在 GUI 中定位 `EK_PROFILE_WINDOW`，展开 EngineCore CUDA HW 行查看 kernel 和
HtoD/DtoH memcpy，展开 Controller/Worker 进程的 NVTX 与 CPU thread 行。搜索
`EK:A:`、`EK:A2E:`、`EK:E:`、`EK:E2A:` 或具体 request ID 可关联阶段。

SSH 服务器没有 `$DISPLAY` 时直接运行 `nsys-ui` 会出现 Qt xcb/OpenGL 错误；这是
GUI 无显示服务，不是 `.nsys-rep` 损坏。把报告复制到安装相同或更高版本 Nsight
Systems 的桌面机器查看，或在服务器使用：

```bash
nsys stats --report nvtx_pushpop_trace \
  /tmp/expertkit-pipeline-detailed.nsys-rep
nsys stats --report nvtx_gpu_proj_trace \
  /tmp/expertkit-pipeline-detailed.nsys-rep
nsys stats --report cuda_gpu_trace:nvtx-name \
  /tmp/expertkit-pipeline-detailed.nsys-rep
```

也可以 `nsys export --type sqlite` 后用 SQLite 工具查看，但跨 NVTX、CUDA runtime
correlation 和 OS context switch 的关联复杂；定量复现优先使用仓库 analyzer。

## 9. 结果验收与常见失败

| 检查 | 通过标准 | 常见失败解释 |
|---|---|---|
| 补丁状态 | `vLLM 0.25.1 ... patched` | 版本不符、部分 patch、site-packages 有未知修改 |
| Python 回归 | 非 live 测试全过，live test 仅按开关 skip | protobuf 版本漂移、补丁 API 缺失 |
| Live Expert RPC | 串行/并发 BF16 输出 `torch.equal` | 服务/模型未注册、路由或响应关联错误 |
| Chrome trace | 四 uBatch、四阶段完整、有逻辑重叠 | batch 不足、非 uniform decode、trace 尾部以外缺阶段 |
| nsys overlap | 能投影 CUDA/CPU 硬件区间并给出掩盖率 | forkserver NVTX 未采集、CUPTI activity 缺失 |
| 性能 | 无 trace/nsys，独立进程，多轮中位数 | 把 profile 开销或 warmup 算入吞吐 |
| Token 等价 | token IDs 完全相同且模式内稳定 | 当前已知未通过，必须保留首个分歧证据 |

复现实验至少保存：Git commit、vLLM/Torch/CUDA/driver/nsys 版本、解析后的模型路径、
配置与 inventory 的非敏感摘要、完整命令、服务日志、benchmark JSON、analysis JSON
和 `.nsys-rep` 路径。原始模型、数据库凭据和 profile 报告不要提交 Git。
