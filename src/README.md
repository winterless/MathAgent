# Minimal Python pipeline (stage1 + stage2 + stage3, JSONL between stages)

## Input format
`--input` can be either:
- a JSONL file, or
- a directory containing `*.jsonl` files (each file runs independently, using filename stem as output prefix)

Each line must be a JSON object with at least one of:

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
- `example_input.stage0.jsonl` (single-file mode: copy of your input)
- `stage1/stage1_raw_generations.stage1.jsonl` (single-file mode)
- `stage1/stage1_output.stage1.jsonl` (single-file mode)
- `stage2/stage2_archive.stage2.jsonl` (single-file mode)
- `stage3/stage3_archive.stage3.jsonl` (single-file mode)
- `accepted_bank.stage_final.jsonl` (single-file mode)

Directory mode additionally writes these **prefixed** artifacts per input file:
- `stage1/<prefix>.stage1_output.stage1.jsonl`, `stage1/<prefix>.stage1_raw_generations.stage1.jsonl`, `stage1/<prefix>.status.stage1.jsonl`
- `stage2/<prefix>.stage2_archive.stage2.jsonl`, `stage2/<prefix>.status.stage2.jsonl`
- `stage3/<prefix>.stage3_archive.stage3.jsonl`, `stage3/<prefix>.status.stage3.jsonl`
- `<prefix>.accepted_bank.stage_final.jsonl`

