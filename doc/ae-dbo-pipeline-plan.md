# AE 分离 DBO 流水线改造计划

> 本文件是实施前的历史计划，其中“双 microbatch”等内容已被实际四
> micro-batch 实现替代。当前设计、真实测试和性能结论见
> [`ae-dbo-pipeline-design.md`](ae-dbo-pipeline-design.md)。

## 1. 背景与目标

当前 vLLM 负责 Attention 和请求执行，Expert-Kit 负责 MoE Expert 的加载、定位与计算。vLLM 插件中的 `GrpcExpert` 在本地完成 Router top-k，通过同步 gRPC 将 Attention 后的 hidden states 和 Expert ID 发送到 Expert-Kit，再在本地完成 top-k 输出加权合并。

目标是在不改变模型结果和请求级调度语义的前提下，引入双 microbatch 的 AE 分离流水线：

```text
A（Attention）→ A2E（发送）→ E（远程 Expert）→ E2A（返回/合并）
```

让一个 microbatch 等待远程 Expert 时，另一个 microbatch 可以执行 Attention 或其他可重叠阶段。

## 2. 设计边界

第一阶段不修改 Scheduler 的请求选择、优先级和 KV Cache 分配逻辑。Scheduler 仍然决定当前 step 的 batch，执行层负责将 batch 拆分为两个 microbatch 并协调其生命周期。

如果后续要求跨 scheduler step 保留未完成的远程 Expert 请求、根据 Worker 背压动态改变 batch，或让 A/E 阶段独立排队，再扩展 Scheduler 与 ModelRunner 之间的状态协议。

## 3. 核心改动

### 3.1 `GpuModelRunner`

- 在 `execute_model`/输入准备阶段将当前 batch 拆为两个 token slice。
- 为两个 microbatch 分别生成 Attention metadata，确保序列位置、KV Cache 和输入切片一一对应。
- 引入 `UBatchWrapper` 和 `UBatchContext`，管理两个执行线程、compute/communication stream、CPU/GPU event 与异常传播。
- 保证一个 step 中两个 microbatch 最终都完成后才向上层返回结果。

### 3.2 `GrpcExpert` 与模型层

将当前同步 `forward()` 改造成可分阶段的远程 Expert 调用，至少抽象出：

1. `prepare_dispatch`：完成 Router 结果整理、Expert ID 构造和输入封装；
2. `submit_a2e`：异步提交 hidden states 和路由信息；
3. `wait_expert`：等待远程 Worker 计算完成，支持超时和取消；
4. `finish_e2a`：恢复 token/top-k 顺序、转换设备与 dtype，并执行 routing weight 加权合并。

`moe_mode` 下的 `ExpertKitMoE` 使用相同协议。shared experts 仍在 vLLM 本地执行，并与 routed experts 的远程阶段明确区分。

### 3.3 Expert-Kit Controller 与协议

- 为每个请求增加稳定的 request ID、microbatch ID 和 layer ID；不能继续依赖硬编码的 `instance_id`。
- 将当前一次性 `Forward` 语义扩展为可异步提交/完成的语义，保留 gRPC、shm、RDMA 三种 Worker 通道。
- 保留按 Expert ID 聚合 token 的逻辑，并在返回时恢复原始 token 与 top-k 顺序。
- 增加超时、取消、Worker 失败和部分结果回收逻辑，避免微批次状态泄漏。

### 3.4 Kernel 与切换点

vLLM 原生 DBO 的 yield 点主要针对 GPU All-to-All dispatch/combine。当前远程 Expert 路径绕过 fused-MoE modular kernel，因此仅修改 `modular_kernel.py` 不足以实现 AE 分离。

切换点应首先放在 A2E/E2A 的异步状态机或 `UBatchContext` 中。只有在远程通信被封装为 vLLM 可感知的通信阶段后，才复用 `yield_and_switch_from_compute_to_comm` 等 kernel 级切换机制。

两个 microbatch 必须经过完全一致数量的切换点；任何提前返回、异常或路由数量差异都必须进入统一的失败路径，否则可能造成死锁。

## 4. 实施阶段

### 阶段一：基线与双 microbatch

- 固定 vLLM 版本和 Expert-Kit 配置。
- 先实现 batch 拆分、独立 Attention metadata 和结果拼接。
- 暂时保留同步 RPC，验证双 microbatch 不改变精度和请求语义。

### 阶段二：异步 A2E/E2A

- 改造 Python 客户端和 Controller 请求关联协议。
- 引入两个 microbatch 的异步提交、完成通知、超时和取消。
- 验证 A、A2E、E、E2A 的阶段顺序和并发上限。

### 阶段三：重叠与性能优化

- 增加 CPU/GPU event 和通信 stream 协调。
- 评估 gRPC 序列化、Controller 聚合和 Worker 计算是否成为瓶颈。
- 在确认收益后再引入 kernel yield 或更细粒度的通信分块。

## 5. 验收标准

- 单 microbatch 与双 microbatch 的 logits、生成结果和异常行为一致。
- 支持 decode、prefill、padding 和不同 top-k 路由分布。
- Worker 超时、连接失败、取消和部分返回不会死锁或泄漏状态。
- 监控中能分别观察 A、A2E、E、E2A 延迟、队列长度和吞吐。
- 在相同硬件和 batch 配置下，端到端延迟或吞吐相较同步基线有可重复收益。
