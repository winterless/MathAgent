# Minimal Python pipeline (stage1 + stage2 + stage3, JSONL between stages)

## Input format
`--input` is a JSONL file. Each line must be a JSON object with at least one of:

- `prompt`: full instruction text (preferred), or
- `question`: task text, or
- `text`: task text

Example: `datasets/example_input.jsonl`

## Run (real OpenAI-compatible API)
Fill env vars (leave blank until you have the API):

```bash
# 本地 OpenAI-compatible 服务（最常见：vLLM，默认端口 8000）
export LLM_BASE_URL="http://127.0.0.1:8000"
# 多数本地服务不校验 key，可留空；如你的服务需要再填写
export LLM_API_KEY=""
# 必须与你的服务端实际暴露的 model id 完全一致（可用 curl 查看）：
# curl http://127.0.0.1:8000/v1/models
# 你当前 vLLM 暴露的是（注意：默认就是 --model 的完整路径）：
export LLM_MODEL="Qwen3-8B"
#
# 如果你想用更短的名字（如 Qwen3-8B），需要你启动 vLLM 时加：
# python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-8B --served-model-name Qwen3-8B --port 8000

# 如果你用的是 OpenAI 官方接口，改为：
# export LLM_BASE_URL="https://api.openai.com"
# export LLM_API_KEY="sk-..."
# export LLM_MODEL="gpt-4o-mini"
python src/run_pipeline.py --input datasets/example_input.jsonl --out datasets/out/demo
```

Optional flags:
- `--sleep`: seconds to sleep between LLM calls (rate limit)

Outputs (all JSONL) are written under `--out`, including:
- `example_input.jsonl` (copy of your input)
- `stage1_raw_generations.jsonl` (important: full raw solver outputs + extracted answers)
- `stage1_output.jsonl` (evaluator-style long `content`, sample.jsonl-like)
- `stage2_output.jsonl` (minimal: only re-evaluates rows missing boxed)
- `stage3_output.jsonl` (minimal: only re-evaluates rows still missing boxed)

