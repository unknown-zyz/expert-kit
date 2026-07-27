# Qwen3-30B-A3B 四阶段实验

使用一键脚本运行同步模式和四阶段 pipeline，并生成 TPOT、吞吐、Attention、A2E、Expert、E2A 以及逐层时间报告：

```bash
/home/zhangyz/expert-kit/.venv/bin/python \
  ek-integration/expertkit_vllm/scripts/run_qwen3_four_stage_benchmark.py \
  --model /data/models/huggingface/Qwen/Qwen3-30B-A3B \
  --config dev/hello-world.config.yaml \
  --dataset /data/datasets/ShareGPT52K/sg_52k.json \
  --batch-size 64 \
  --max-tokens 32 \
  --warmup 1 \
  --repetitions 5 \
  --output-dir output/qwen3-30b-a3b-four-stage
```

脚本会启动并停止本轮使用的 Weight Server、Controller 和 Worker。数据库必须已经可访问且模型已经完成 Expert-Kit 注册与调度；脚本不会修改现有 YAML 配置。

当前 vLLM four-stage uBatch wrapper 要求同一 batch 的输入 token 长度一致。脚本默认从 ShareGPT 中筛选至少 256 token 的样本，并截取为统一 256 token IDs；这仍然使用 ShareGPT 内容，但避免了不同 block-table 长度触发 vLLM 的 uBatch 限制。可用 `--input-tokens` 调整该值。

报告输出在 `output/qwen3-30b-a3b-four-stage/`，主要文件是：

- `qwen3-four-stage-report.md`：汇总报告；
- `qwen3-four-stage-report.json`：完整原始指标、生成结果和 Nsight 分析；
- `qwen3-four-stage-overlap.svg`：四阶段 micro-batch 时间图；
- `sync.nsys-rep`、`pipeline.nsys-rep`：原始 Nsight Systems 报告。

TPOT 使用批次平均口径：测量窗口总耗时除以该批次生成的总 output token 数，包含 prefill，不能等同于纯 decode TPOT。Attention 使用 CUDA kernel active time，Expert 使用 Worker Expert 硬件计算时间，A2E/E2A 包含跨进程传输、序列化、排队和结果聚合。

如需只做正常性能测试而不采集 Nsight：

```bash
.../run_qwen3_four_stage_benchmark.py \
  --config dev/hello-world.config.yaml \
  --skip-nsys
```
