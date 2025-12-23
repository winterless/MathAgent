# Minimal Python pipeline (stage1+stage2+stage3, JSONL between stages)

## Input format
`--input` is a JSONL file. Each line must be a JSON object with at least one of:

- `prompt`: full instruction text (preferred), or
- `question`: task text, or
- `text`: task text

Example: `datasets/example_input.jsonl`

## Run (mock mode, no real LLM needed)

```bash
export LLM_MOCK=1
python src/run_pipeline.py --input datasets/example_input.jsonl --out datasets/out/demo
```

## Run (script backend; default)
If you do **not** set `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`, `MathAgent` will default to running a local script backend.

Default script path:
- `/home/unlimitediw/workspace/TYDeepResearch/AgenticRLModelTraining/model/scripts/call_model.py`

Override the script path / python:
Set `LLM_SCRIPT_PATH` (and optionally `LLM_SCRIPT_PYTHON`) in your environment.

## Run (real OpenAI-compatible API)
Fill env vars (leave blank until you have the API):

```bash
export LLM_BASE_URL="YOUR_BASE_URL"
export LLM_API_KEY="YOUR_API_KEY"
export LLM_MODEL="YOUR_MODEL"
python src/run_pipeline.py --input YOUR_INPUT.jsonl --out datasets/out/real
```

Optional flags:
- `--sleep`: seconds to sleep between LLM calls (rate limit)

Outputs (all JSONL) are written under `--out`, including:
- `stage1_input.jsonl`, `stage1.jsonl`
- `stage2_input.jsonl`, `stage2.jsonl`
- `stage3_input.jsonl`, `stage3.jsonl`
- `final_stage2.jsonl`, `final_stage3.jsonl`, merged `final.jsonl`
- `discarded.jsonl` (stage3 still unstable)

