# Minimal Python pipeline (stage1 + stage2 + stage3, JSONL between stages)

## Input format
`--input` is a JSONL file. Each line must be a JSON object with at least one of:

- `prompt`: full instruction text (preferred), or
- `question`: task text, or
- `text`: task text

Example: `datasets/example_input.jsonl`

## Run (real OpenAI-compatible API)
### Use the model routing config (single mode)

Edit `config/llm_models.json` to describe your **three models** (e.g. 30B 快思考 / 30B 慢思考 / 基础模型),
and how each pipeline stage routes to them, then run:

```bash
python src/run_pipeline.py --input datasets/example_input.jsonl --out datasets/out/demo --llm-config config/llm_models.json
```

Optional flags:
- `--sleep`: seconds to sleep between LLM calls (rate limit)

Outputs (all JSONL) are written under `--out`, including:
- `example_input.stage0.jsonl` (copy of your input)
- `stage1_raw_generations.stage1.jsonl` (important: full raw solver outputs + extracted answers + per-attempt verdicts)
- `stage1_output.stage1.jsonl` (evaluator-style long `content`, sample.jsonl-like wrapper)
- `stage2_archive.stage2.jsonl` (per-problem archive for Stage2: 8 attempts + ok/bad + llm_call_counts)
- `stage3_archive.stage3.jsonl` (per-problem archive for Stage3)
- `accepted_bank.stage_final.jsonl` (accepted problems from stage2/stage3)
- `discarded_hard.stage_final.jsonl` (discarded problems after stage3)

