# DeepSeek-V2-Lite 四阶段流水测试结果

## 1. 结论

测试日期：2026-07-21/22。总体状态：**FAIL（功能可运行，但严格 token 等价和吞吐提升均未达到验收条件）**。

- **PASS**：本地模型路径成功解析；vLLM 只加载 2.54 GiB 非 routed-expert 权重；1664 个 routed experts 全部由 CPU Python Worker 加载并可通过 gRPC 推理。
- **PASS**：sync/pipeline 冒烟测试的 4 条请求 × 2 输出 token 完全一致。
- **PASS**：batch 4、8、16、32、64 均完成 1 次 warmup + 5 次正式测量，每请求固定生成 32 token。
- **FAIL**：sync 与 pipeline 各自的 5 次重复都不是 bitwise deterministic；首个跨模式差异位于 batch 4、请求 `77Mgsll`、输出 token 18。
- **FAIL**：pipeline 中位 output throughput 没有提升，speedup 为 0.9515–0.9982。
- **PASS**：四份 nsys CUDA/NVTX 报告成功生成；数据表明 H2D/D2H 不是主要瓶颈，时间主要消耗在 GPU 之外的 CPU E、gRPC 和等待。

严格比较失败不能单独归因于流水线，因为不开流水线的 sync 基线本身也会在相同输入和 seed 下变化。该问题需要先解决底层 BF16/并行归约的非确定性，再重新执行 token 级验收。

## 2. 环境和实验设置

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32607 MiB |
| Driver | 610.43.03 |
| CPU | AMD Ryzen 9 9950X，16 核/32 线程 |
| Torch/CUDA | 2.11.0+cu130 / CUDA 13.0 |
| vLLM | 0.25.1 + Expert Kit pipeline v2 patch |
| 模型 | `/data/models/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat` |
| 数据集 | `/data/datasets/ShareGPT52K/sg_52k.json` |
| Frontend | 单 RTX 5090，BF16，eager，prefix cache 关闭 |
| Worker | Python Torch CPU，4 active slots，OMP/MKL threads=4 |
| batch | 4、8、16、32、64 |
| micro-batch | pipeline 固定 4；sync 不切分 |
| prompt | seed 42，从 ShareGPT 第一轮筛选 4–512 token |
| generation | temperature=0，seed=0，ignore EOS，固定 32 token |
| 重复 | 每个 batch 1 warmup + 5 measured runs |

隔离服务端口为 PostgreSQL 55432、Weight Server 6543、Controller 5001/5002、Worker 51051。Controller heartbeat timeout 设为 600 秒；30 秒会在 vLLM 的 32768-token profile dummy 阻塞 Python Worker event loop 时误摘除健康 Worker。

## 3. 性能结果

以下是 5 次正式测量中位数。`speedup = pipeline output tok/s / sync output tok/s`。

| batch | sync s | pipeline s | sync out tok/s | pipeline out tok/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 4 | 10.675 | 10.754 | 11.991 | 11.902 | 0.9926 |
| 8 | 16.213 | 16.242 | 15.789 | 15.761 | 0.9982 |
| 16 | 21.832 | 22.944 | 23.452 | 22.316 | 0.9515 |
| 32 | 27.037 | 27.205 | 37.873 | 37.640 | 0.9938 |
| 64 | 33.310 | 33.497 | 61.483 | 61.140 | 0.9944 |

流水运行时 Worker 实时 CPU 利用率约 374–379%，说明多个 uBatch RPC 确实能同时推进。但同步模式把同层更多 token 合并成较大的 expert GEMM，本身也能使用 OMP 线程。流水把它拆成多个更小的矩阵和更多 RPC，导致小矩阵效率、序列化和调度开销抵消了并发收益。batch 16 的退化最明显。

## 4. nsys 结果与瓶颈

nsys 2026.1.3 使用 `cudaProfilerApi` 只采集 warmup 后的一次生成。Frontend 是 CUDA 进程；CPU Worker 是独立进程，因此下表中的 GPU kernel 是 A（包含 Attention、Router、shared expert 和其它本地 GPU compute），Memcpy 是可由 CUDA 观察到的 A2E/E2A staging。CPU E 和 gRPC Host 等待位于 capture wall 与 GPU 活动之间的巨大空白中。

| 模式 | batch | capture span ms | GPU kernel sum ms | kernel/span | CUDA memcpy ms | memcpy MiB |
|---|---:|---:|---:|---:|---:|---:|
| sync | 4 | 11127.7 | 129.705 | 1.17% | 4.543 | 148.4 |
| pipeline | 4 | 11169.3 | 129.705 | 1.16% | 4.517 | 148.4 |
| sync | 64 | 33147.8 | 182.003 | 0.55% | 35.453 | 2331.1 |
| pipeline | 64 | 32974.3 | 182.265 | 0.55% | 35.239 | 2331.1 |

结论：GPU compute 和 CUDA memcpy 合计远小于端到端时间，H2D/D2H 不是主要瓶颈。当前主要瓶颈是 CPU routed-expert 计算以及与它串联的 gRPC 编解码、Worker admission/排队和 Host wait。pipeline batch 64 的单次 nsys span 比 sync 快约 0.52%，但五次中位数反而慢约 0.56%，属于运行波动，不能宣称提升。

报告位置：

```text
/home/zhangyz/expert-kit/output/deepseek-v2-pipeline-dev-py/
  sync.json
  pipeline.json
  comparison.json
  nsys-sync-b4.nsys-rep
  nsys-pipeline-b4.nsys-rep
  nsys-sync-b64.nsys-rep
  nsys-pipeline-b64.nsys-rep
```

查看汇总：

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum \
  /home/zhangyz/expert-kit/output/deepseek-v2-pipeline-dev-py/nsys-pipeline-b64.nsys-rep
```

图形界面需要有 X11/Wayland display 的桌面。远程无 display 的服务器不能直接运行 `nsys-ui`；可把 `.nsys-rep` 放到安装了同版本或更新版 Nsight Systems 的桌面打开。

## 5. 正确性分析

完整报告的 `deterministic=false` 不是请求失败：所有请求均生成了固定 32 token，服务无 RPC/shape/dtype 错误。它表示同一模式的 5 次输出 SHA256 不完全相同。sync 和 pipeline 在各 batch 都出现变化，且多个 hash 在两种模式中交叉出现；这符合接近决策边界的 logits 被底层浮点归约细微变化放大，而不是流水线固定地产生另一套结果。

当前正确性证据分三层：

1. 单元/调度测试验证 Future completion 顺序、失败传播、四 uBatch workspace 和 Worker slot 上限。
2. 真实 4×2-token 冒烟测试验证端到端输出 token 严格相同。
3. 4–64×32-token 压测验证所有请求完成，但暴露已有同步路径也不 bitwise deterministic，因此严格模型级验收失败。

要完成严格验收，应先定位并固定 sync 的数值非确定性（例如确定性 GEMM/归约、单线程 CPU expert 对照、逐层 hidden-state 最大误差和首个路由分歧），然后重新运行同一 manifest。

## 6. 代码修复和原因

- `RemoteMoERunner` 新增零存储 `w13_weight/w2_weight` sink，满足 vLLM 0.25.1 DeepSeek-V2 model-level loader 的参数名契约，同时不把 26.8 GiB routed expert 权重放回 GPU。
- 模型构造期缓存 `num_layers`，避免 custom op forward 在 vLLM config context 外首次创建 Transport client 时失败。
- pipeline v2 patch 让 `gpu_worker.py` 按 `parallel_config.num_ubatches` 分配 workspace；原代码只为 `enable_dbo` 分配 2 份，显式 `ubatch_size=4` 会不足。
- Python Torch Worker 接受 `device: cpu`；Linux 内存检查使用 `/proc/meminfo` 的 `MemAvailable`，避免 `SC_AVPHYS_PAGES` 忽略可回收 page cache、把 159 GiB 可用内存误报成约 3.8 GiB。
- benchmark 在离线 `LLM.generate()` 不提供 per-request metrics 时把 TTFT/TPOT 记录为 `null`，不伪造数据；吞吐仍使用同步后的端到端 wall time。
- `nsys_benchmark.py` 避免 nsys 跟踪 `py-cpuinfo` 调用外部 `file` 时的子进程死锁，仅替代 architecture 探测，不改变推理代码。

完整启动和复现命令见 [`deepseek-v2-lite` benchmark README](../../ek-integration/expertkit_vllm/benchmarks/deepseek-v2-lite/README.md)。
