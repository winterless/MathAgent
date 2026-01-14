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

## vLLM connection refused auto-recovery (no config changes)

If your vLLM server occasionally crashes/restarts (common under very long generations), the client may raise:

- `RuntimeError: LLM network error: <urlopen error [Errno 111] Connection refused>`

This means **the server is not listening** on `base_url` at that moment. Without changing `config/llm_models.json`,
you can enable a best-effort recovery mechanism via environment variables (implemented in `src/infra/llm_client.py`):

- `MATHAGENT_LLM_WAIT_ON_CONNREFUSED_S`: seconds to wait for the server to come back (default `0`, disabled)
- `MATHAGENT_LLM_RESTART_CMD`: optional shell command to restart vLLM (run at most once per request on connection refused)
- `MATHAGENT_LLM_HEALTH_URL`: optional health check URL (default: `<base_url>/v1/models`)
- `MATHAGENT_LLM_CONNREFUSED_LOG`: set to `1` to print recovery logs to stderr

Example (recommended if you already run vLLM under systemd/docker restart policies; just wait for recovery):

```bash
export MATHAGENT_LLM_WAIT_ON_CONNREFUSED_S=120
export MATHAGENT_LLM_CONNREFUSED_LOG=1
PYTHONPATH=src python3 src/run_pipeline.py --input datasets/example_input.jsonl --out datasets/out/demo --llm-config config/llm_models.json
```

Outputs (all JSONL) are written under `--out`.

Per input file, artifacts are always **prefixed by the input filename stem**:
- `stage1/<prefix>.stage1_output.stage1.jsonl`, `stage1/<prefix>.stage1_raw_generations.stage1.jsonl`, `stage1/<prefix>.status.stage1.jsonl`
- `stage2/<prefix>.stage2_archive.stage2.jsonl`, `stage2/<prefix>.status.stage2.jsonl`
- `stage3/<prefix>.stage3_archive.stage3.jsonl`, `stage3/<prefix>.status.stage3.jsonl`
- `<prefix>.stage0.jsonl` (copy of your input, for traceability)
- `<prefix>.accepted_bank.stage_final.jsonl`

## Run (Route-A: modular modes, infer/eval split)

This repo also supports **modular execution** via `--mode`, so you can run only the next step and stop
(useful for distributed workers pulling tasks from a shared data source).

Note: the project uses `src/` as import root, so we recommend running with `PYTHONPATH=src`.

### Stage1: infer -> eval

Stage1 infer (produce raw solve outputs only):

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode stage1_infer \
  --input datasets/input \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

Note: `stage1_infer` expects **raw input JSONL**. If you pass an `--out` root directory as `--input`,
it will only work when that directory already contains `*.stage0.jsonl` copies of the original inputs.

Stage1 eval (prefer consuming existing `stage1_output` and produce `status.stage1` + `stage2_input`):
  - It first parses boxed counts from `stage1_output.output` (no LLM needed): `\\boxed{解答正确：x，解答错误：y}`
  - Only when parsing fails will it best-effort call LLM to re-produce boxed counts
  - Backward compat: if no `stage1_output` is found, it will fall back to consuming `stage1_infer` and generating `stage1_output`

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode stage1_eval \
  --input datasets/out/demo_modular \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

### Stage2: infer -> eval

Stage2 infer (prefer consuming `stage2_input` task list; fallback: derive it from `stage1_output` + `status.stage1`):

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode stage2_infer \
  --input datasets/out/demo_modular \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

Stage2 eval (consume `stage2_infer` and produce `stage2_archive` + `status.stage2` + `stage3_input`):

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode stage2_eval \
  --input datasets/out/demo_modular \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

### Stage3: infer -> eval

Stage3 infer (consume `stage3_input` and produce `stage3_infer`):

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode stage3_infer \
  --input datasets/out/demo_modular \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

Stage3 eval (consume `stage3_infer` and produce `stage3_archive` + `status.stage3`, and write `result/`):

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode stage3_eval \
  --input datasets/out/demo_modular \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

