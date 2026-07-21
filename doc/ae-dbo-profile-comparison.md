# AE-DBO 同步/四阶段流水 Profile 对比

## 1. 结论

本轮在同一台 RTX 5090、同一模型和同一 Expert-Kit 服务配置下，对三种模式分别做了 Nsight Systems profile 和关闭 profile 的五轮吞吐测试：

| 模式 | Worker 线程 | 中位吞吐 | 相对同步 | 中位延迟 | 相对同步 |
|---|---:|---:|---:|---:|---:|
| sync | 1 | 5.405 tokens/s | 基线 | 23.681s | 基线 |
| pipeline | 1 | 5.089 tokens/s | **-5.85%** | 25.152s | **+6.21%** |
| pipeline | 4 | 6.470 tokens/s | **+19.70%** | 19.784s | **-16.46%** |

因此，“流水线必然提升吞吐”并不成立：流水只提供重叠机会，后端必须能消费并发。单线程 Worker 仍串行执行 Expert，micro-batch 拆分增加的 RPC 数、排队和本地调度超过了重叠收益；四线程 Worker 把 Expert 最大并发提高到 4 后，排队下降，端到端吞吐转为正收益。

当前主要瓶颈不是 gRPC channel 初始化，也不是 H2D/D2H PCIe memcpy，而是 Worker 队列和 CPU Expert 计算。Worker×4 又使单个 Expert 数学计算从 0.452ms 增至 0.933ms，说明 CPU 线程/内存带宽争用已成为下一阶段限制。

![同步与流水关键路径对比](assets/ae-dbo-sync-vs-pipeline-profile.svg)

图中的条形是一次 vLLM→Controller Expert 请求的跨进程平均关键路径，不是整个 token 或整层耗时。`Worker RPC span` 内部包含 Controller→Worker gRPC、Worker 排队、Expert 数学计算、序列化和返回；下面进一步拆分。

## 2. 测试口径

环境：

- GPU：NVIDIA GeForce RTX 5090，驱动 610.43.03；
- Nsight Systems：2026.1.3；
- vLLM：0.25.1；
- Torch：2.11.0+cu130，CUDA 13.0；
- 模型：项目软链接 `qwen3-30b-a3b`，解析到 `/data/models/huggingface/Qwen/Qwen3-30B-A3B`；
- Expert：Expert-Kit CPU Worker；
- 服务端口：weight-server 6543、Controller 5001/5002、Worker 51234。

Profile workload 为四条固定 prompt、每条最多生成 8 tokens、一次 warmup、一次测量。三份报告均使用 `--trace=cuda,nvtx,osrt`、`--trace-fork-before-exec=true` 和 `--cuda-event-trace=true`。分析器只统计 `EK_PROFILE_WINDOW` 内事件，模型加载和 warmup 不进入表格。

正式性能 workload 为相同四条 prompt、每条 32 tokens、一次 warmup、五次测量；未设置 `EK_NSYS_PROFILE` 和 `EK_PIPELINE_TRACE`。每种模式使用独立服务和 vLLM 进程。

## 3. Nsight 关键路径

### 3.1 跨进程请求分解

| 指标（平均每个客户端 Expert 请求） | sync / W×1 | pipeline / W×1 | pipeline / W×4 |
|---|---:|---:|---:|
| Client→Controller 完整 gRPC | 13.874ms | 12.823ms | 8.921ms |
| Client→Controller ingress | 0.171ms | 0.196ms | 0.244ms |
| Controller 请求准备 | 0.153ms | 0.098ms | 0.868ms |
| Worker RPC 跨度 | 13.155ms | 12.232ms | 7.270ms |
| Controller 聚合/收尾 | 0.120ms | 0.083ms | 0.095ms |
| Controller→Client return | 0.276ms | 0.213ms | 0.444ms |

`Controller 请求准备` 的 Worker×4 均值被少量调度长尾拉高；其中位数为 0.100ms，p95 为 3.785ms。Worker×4 的完整请求均值 8.921ms、中位数 6.182ms、p95 21.956ms，也说明并发降低平均等待的同时仍存在 CPU 调度长尾。

### 3.2 Worker 内部拆分

| 指标 | sync / W×1 | pipeline / W×1 | pipeline / W×4 |
|---|---:|---:|---:|
| 客户端 Expert 请求数 | 432 | 1296 | 1296 |
| Controller→Worker RPC 数 | 12455 | 14042 | 13984 |
| Worker RPC / 客户端请求 | 28.83 | 10.83 | 10.79 |
| Expert 最大并发 | 1 | 1 | 4 |
| 单个 Controller→Worker gRPC | 6.829ms | 9.727ms | 3.450ms |
| 单个 Worker 排队 | 6.223ms | 9.074ms | 2.009ms |
| 单个 `exp.forward()` 数学计算 | 0.421ms | 0.452ms | 0.933ms |

这组数据定位了单线程负收益的直接原因：pipeline 把客户端请求数增加到同步的 3 倍，Controller→Worker RPC 总数增加约 12.7%，但 Worker Expert 最大并发仍为 1。单个 RPC 的排队从 6.223ms 增至 9.074ms，Controller→Worker gRPC 的绝大部分时间实际是在等待 Worker，而不是传输。

Worker×4 把单 RPC 排队从 9.074ms 降到 2.009ms，Controller→Worker gRPC 从 9.727ms 降到 3.450ms；这足以令吞吐提升 19.70%。代价是四个 CPU Expert 同时执行时，单次数学计算增至 0.933ms（约 2.06 倍）。因此继续增加线程未必继续提升，下一步应扫描 2/3/4/6/8 线程，并同时观察 CPU 利用率、NUMA 和内存带宽。

### 3.3 Attention 与本地 GPU 计算

流水报告内，`Qwen3MoeAttention` 的 CUDA kernel 并集平均为 1.796ms（Worker×1）和 1.918ms（Worker×4）；Router/Top-k kernel 平均约 0.0067ms，weighted combine kernel 平均约 0.0037ms。图中阶段显示仍按既定口径把 Router、weighted combine 等归入 Attention/Expert 块，但定量表保留独立硬件数值。

同步报告包含 432 个 Attention、Router 和 weighted-combine NVTX range，但两次同步采集中 CUPTI kernel/memcpy activity 都在测量窗口开始前停止，导致这些 range 无法投影到 GPU activity。该情况不能解释成同步模式 GPU 时间为零。同步的跨进程 NVTX、Controller 和 Worker 数据完整；GPU 计算横向判断只使用两份流水报告，正式吞吐结论使用无 profile 数据。

## 4. 通信瓶颈定位

### 4.1 不是 gRPC 初始化

每个 vLLM EngineCore 只创建一次 channel。三种模式的 channel 创建为 2.33–2.36ms，ready/握手为 2.34–2.56ms，均发生在模型初始化阶段并被 warmup 摊销，不在测量窗口关键路径中。当前瓶颈不是“每个请求重复初始化 gRPC”。

### 4.2 H2D/D2H 不是主要瓶颈

| pipeline 模式 | 实际 D2H memcpy | 实际 H2D memcpy | D2H CPU 包围区间 | H2D CPU 包围区间 |
|---|---:|---:|---:|---:|
| Worker×1 | 0.00046ms | 0.00282ms | 0.0730ms | 0.0610ms |
| Worker×4 | 0.00047ms | 0.00417ms | 0.1335ms | 0.1012ms |

实际 CUDA memcpy 是微秒级，比 8.9–12.8ms 的客户端 gRPC 关键路径小三个数量级。CPU 包围区间还包含 pinned buffer 分配、PyTorch dispatch 和 enqueue，不应当作 PCIe 传输时间。

同步模式由于上述 CUPTI 缺口没有实际 memcpy 投影；其 CPU 包围区间为 D2H 0.052ms、H2D 0.088ms，也足以排除“几十毫秒 H2D”这一假设，但不能替代同步实际 GPU memcpy 数值。

### 4.3 gRPC 时间主要是服务端排队

Controller→Worker gRPC 的残余时间可粗略写成：

```text
gRPC wall ≈ Worker queue + Expert math + framing/dispatch/serialization/return
```

按均值相减，最后一项约为同步 0.19ms、pipeline/Worker×1 0.20ms、pipeline/Worker×4 0.51ms。这是包含多项软件开销的上界，不是纯网络 RTT。相比之下，Worker 排队分别为 6.223ms、9.074ms、2.009ms。因此当前“通信慢”主要是 RPC 生命周期内的后端排队，而非 localhost 网络、channel 握手或 PCIe memcpy。

序列化本身也较小：Client safetensors 输入保存平均为 0.14–0.18ms；Controller 输入 load 约 0.009–0.014ms；Worker safetensors load 约 0.001–0.003ms。它们可以继续优化，但不是当前首要瓶颈。

## 5. 为什么单线程流水吞吐下降、四线程又提升

单线程模式的链路是：

1. 配置 `ubatch_size=4` 后，请求被拆得更细，测量窗口内客户端 Expert 请求从 432 增至 1296；
2. 每个请求包含的唯一 Expert 更少，但总 Worker RPC 从 12455 增至 14042；
3. Worker 仍只有一个执行槽，多个 in-flight Future 只是在队列中重叠，并没有 Expert 硬件并行；
4. Worker 排队增至 9.074ms，Router/路由准备和 Python Future 调度也按更多请求重复执行；
5. 被隐藏的 Client/Controller 等待不足以抵消新增开销，最终吞吐下降 5.85%。

四线程模式没有改变请求拆分，但允许四个 Expert 真正并行：Worker 排队下降 77.9%，单客户端请求关键路径从 12.823ms 降至 8.921ms。虽然 CPU Expert 单次计算变慢，净收益仍为吞吐 +19.70%。

当前 Worker×4 的优化优先级应为：

1. 降低 Expert CPU 争用：线程数扫描、固定 CPU affinity/NUMA、检查 GGML 内部线程与外层 Worker 线程是否过度订阅；
2. 降低 p95 Worker queue 和 Controller 调度长尾；
3. 合并相邻 micro-batch 的相同 Expert 请求，恢复 Expert GEMM batch size，同时保留 A/E 流水；
4. 最后再优化 safetensors 和 gRPC framing。H2D/D2H 当前不是优先项。

## 6. 正确性说明

三种五轮 benchmark 都生成 128 tokens/轮，但三种模式的跨轮 token IDs 均不稳定；同步模式自身也不能稳定复现第一轮。pipeline 与同步第一轮也存在分歧。因此当前只能确认：

- 请求元数据和响应关联检查通过；
- 串行与并发 Expert RPC 对固定 BF16 输入逐元素相等；
- 四阶段顺序和重叠 trace 通过；
- 模型级严格 token 等价仍为 **FAIL**。

由于同步基线自身非 bitwise deterministic，不能把所有生成分歧归因于流水，也不能用本轮吞吐提升替代正确性验收。应先建立逐层 hidden/logits 对照，定位同步自身不稳定和 pipeline 首个分歧层。

## 7. 复现命令

先构建 release Worker：

```bash
cargo build --release --bin ek-cli
```

以下是同步 profile；另外两种模式分别把输出名改为 `pipeline-serial-detailed`/`pipeline-parallel-detailed`，把 `--mode` 改为 `pipeline`，并把 `--worker-threads` 改为 1/4：

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
  --config dev/hello-world.config.yaml \
  --model qwen3-30b-a3b \
  --mode sync --worker-threads 1 \
  --warmup 1 --repetitions 1 --max-tokens 8 \
  --log-dir /tmp/ek-nsys-sync-logs \
  --benchmark-output /tmp/ek-nsys-sync-workload.json
```

无 profile 性能基准使用同一启动器并增加 `--no-profile-window`：

```bash
.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/run_nsys_pipeline_profile.py \
  --config dev/hello-world.config.yaml \
  --model qwen3-30b-a3b \
  --mode sync --worker-threads 1 \
  --no-profile-window \
  --warmup 1 --repetitions 5 --max-tokens 32 \
  --log-dir /tmp/ek-benchmark-sync-logs \
  --benchmark-output /tmp/expertkit-benchmark-sync.json
```

再分别运行 pipeline/Worker×1 和 pipeline/Worker×4，并生成统一 JSON 和 SVG：

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
  doc/assets/ae-dbo-sync-vs-pipeline-profile.svg
```

本轮原始报告保存在 `/tmp`：同步约 40MB、pipeline/Worker×1 约 51MB、pipeline/Worker×4 约 52MB。`/tmp` 文件不会提交仓库；仓库保留复现脚本、定量结论和生成图。

GUI 查看：

```bash
nsys-ui /tmp/expertkit-sync-detailed.nsys-rep
```

远程主机没有图形桌面时，不应直接在 SSH shell 运行 `nsys-ui`；将 `.nsys-rep` 复制到有桌面的机器并用相同或更高版本 Nsight Systems 打开。服务器端可使用 `nsys stats` 或本项目分析器。
