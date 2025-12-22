# Minimal Python pipeline (JSONL between stages)

## Input format
`--input` is a JSONL file. Each line must be a JSON object with at least:

- `id`: any string/int
- `question`: the problem text

Example: `src/examples/input.jsonl`

## Run (mock mode, no real LLM needed)

```bash
export LLM_MOCK=1
python src/run_pipeline.py --input src/examples/input.jsonl --out runs/demo
```

## Run (real OpenAI-compatible API)
Fill env vars (leave blank until you have the API):

```bash
export LLM_BASE_URL="YOUR_BASE_URL"
export LLM_API_KEY="YOUR_API_KEY"
export LLM_MODEL="YOUR_MODEL"
python src/run_pipeline.py --input YOUR_INPUT.jsonl --out runs/real
```

Optional flags:
- `--samples`: answers per stage (default 8)
- `--n`: stable if majority_count > n (default 7)
- `--sleep`: seconds to sleep between LLM calls (rate limit)

Outputs (all JSONL) are written under `--out`, including:
- `stage1.jsonl`, `stage2.jsonl`, `stage3.jsonl`
- `final_stage2.jsonl`, `final_stage3.jsonl`, merged `final.jsonl`
- `easy_end.jsonl` (stage1 stable -> treat as easy, stop)
- `discarded.jsonl` (stage3 still unstable)

