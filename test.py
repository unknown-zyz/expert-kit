import os
from pathlib import Path

model_root_value = os.environ.get("QWEN3_30B_A3B_ROOT")
if not model_root_value:
    raise RuntimeError("QWEN3_30B_A3B_ROOT must point to the local model directory")

model_root = Path(model_root_value).expanduser().resolve(strict=True)
required_files = ("config.json", "model.safetensors.index.json", "tokenizer.json")
missing_files = [name for name in required_files if not (model_root / name).is_file()]
if missing_files:
    raise RuntimeError(
        f"Incomplete model directory {model_root}: missing {', '.join(missing_files)}"
    )

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_MLA_DISABLE", "1")
os.environ.setdefault("EK_ENABLE", "1")
os.environ.setdefault("EK_MODEL_NAME", "qwen3-30b-a3b")
os.environ.setdefault("EK_MODE", "expert_mode")
os.environ.setdefault("EK_ADDR", "localhost:5002")
os.environ.setdefault("EK_CLIENT_TIMEOUT", "120")
os.environ.setdefault("EK_DEBUG_MODE", "0")
os.environ.setdefault("EK_PIPELINE_ENABLE", "1")
os.environ.setdefault("EK_PIPELINE_TRACE", "/tmp/expertkit-pipeline-trace.json")
pipeline_enabled = os.environ["EK_PIPELINE_ENABLE"] == "1"

from vllm import LLM, SamplingParams  # noqa: E402

prompts = [
    "Hello, my name is",
    "The president of the United",
    "In a distant future, humanity",
    "The key idea behind a pipeline is",
]
if os.environ.get("EK_TEST_REVERSE_PROMPTS") == "1":
    prompts.reverse()

max_tokens = int(os.environ.get("EK_TEST_MAX_TOKENS", "32"))

llm = LLM(
    model=str(model_root),
    trust_remote_code=True,
    max_model_len=256,
    enforce_eager=True,
    cpu_offload_gb=64,
    max_num_batched_tokens=1024,
    ubatch_size=4 if pipeline_enabled else 0,
    dbo_decode_token_threshold=0,
    dbo_prefill_token_threshold=0,
)

outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens))
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated text: {output.outputs[0].text!r}")
