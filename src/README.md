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

## vLLM auto-recovery (config-based)

If your vLLM server occasionally crashes/restarts (common under very long generations), the client may raise:

- `RuntimeError: LLM network error: <urlopen error [Errno 111] Connection refused>`

This means **the server is not listening** on `base_url` at that moment. Recovery is configured via
`config/llm_models.json` under `options` (no environment variables required).

## vLLM auto-start from config (no external script)

You can optionally configure auto-start and recovery **inside `config/llm_models.json`** under `options`.
This reuses the same start command when a connection-refused error happens.

Example:

```json
{
  "options": {
    "vllm_autostart": true,
    "vllm_model_path": "/path/to/model",
    "vllm_model_name": "Qwen3-8B",
    "vllm_start_cmd": "python -m vllm.entrypoints.openai.api_server --model \"{model_path}\" --served-model-name {model_name} --dtype auto --tensor-parallel-size 1 --max-model-len 32384 --gpu-memory-utilization 0.85 --swap-space 8 --port 8000",
    "vllm_wait_s": 60,
    "vllm_health_url": "http://127.0.0.1:8000/v1/models",
    "vllm_restart_on_connrefused": true,
    "vllm_connrefused_log": true
  }
}
```

Notes:
- `vllm_autostart=true` launches the command at pipeline startup if health check is not OK.
- `vllm_restart_on_connrefused=true` reuses the same command on `Errno 111` recovery.
- `vllm_health_url` can be left empty to use `<base_url>/v1/models`.
- `vllm_start_with_bash=true` runs the command via `bash -lc` (useful for conda/venv activation).
- `vllm_log_path` captures vLLM stdout/stderr for troubleshooting (default `/tmp/mathagent_vllm.log`).
- `vllm_restart_on_runtime_error=true` restarts vLLM when outputs contain runtime-like errors (e.g. `LLM HTTPError 503`).
- `vllm_restart_cooldown_s` throttles restarts to avoid loops.
- `vllm_restart_cmd` can include both stop + start (e.g. `pkill ...; python -m vllm.entrypoints.openai.api_server ...`).
- If `vllm_restart_cmd` is empty, the pipeline automatically combines `vllm_stop_cmd; vllm_start_cmd`.
- `vllm_shutdown_on_exit=true` stops vLLM when the pipeline exits (uses `vllm_stop_cmd`).
- `vllm_stop_cmd` should be a safe stop command for your environment (e.g. `pkill -f vllm.entrypoints.openai.api_server`).
 - `vllm_model_path` / `vllm_model_name` are injected into `vllm_start_cmd` via `{model_path}` and `{model_name}` placeholders.

## Startup cleanup: remove rows with connection errors so reruns can reprocess them

If some produced artifacts contain connection-related LLM error markers (e.g. `[LLM_ERROR ... connection refused ...]`),
re-running the pipeline may skip those uuids because they already appear in `stage*_infer` / `status.stage*` files.

By default, `src/run_pipeline.py` performs a best-effort cleanup on startup:

- It scans JSONL files under `--out` (derived artifacts like `stage*_infer`, `stage*_archive`, `status.stage*`, `accepted_bank`, `result`)
- It collects uuids whose rows contain connection/network error markers
- It removes those uuids from the derived artifacts (so the next run will include them again)

It does **not** delete task lists (`stage2_input.stage2.jsonl`, `stage3_input.stage3.jsonl`), nor `stage1_output.stage1.jsonl`, nor `stage0` copies.

To disable this behavior, set `options.purge_conn_errors_on_start=false` in `config/llm_models.json`.

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

### Rebuild result/ from existing artifacts

If you already have `accepted_bank` / `stage2_archive` / `stage3_archive` and want to regenerate `result/*.result.stage_final.jsonl`:

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode result_rebuild \
  --input datasets/out/demo_modular
```

Rebuild rules:
- Only include rows with `majority_count >= min_votes_to_accept`
- Only include rows where one attempt matches the majority answer
- Result schema: `{"uuid": ..., "text": "<raw_text>"}` (raw_text from the matching attempt)
- If `--input` is a parent directory that contains multiple run subdirectories
  (each with their own `stage1/2/3`), the subdirectory name is prepended to the
  result prefix to avoid collisions. All results are written under the parent `result/`.

Optional:
- `options.result_rebuild_use_infer=true` allows rebuilding from `stage2_infer`/`stage3_infer`
  when archives are missing. In this mode, majority vote is computed from `extracted_answers`,
  and the matching `raw_model_outputs` is used as `text`.

Output compaction:
- `options.compact_outputs=true` keeps only per-stage `status.*.jsonl` and `stage*_infer.*.jsonl`.
  It disables writing `stage*_archive`, `stage*_raw_generations`, `stage*_input`, `accepted_bank`,
  and `result` in normal modes. Parsing of `stage1_output` is still supported for compatibility.
  In full mode, the pipeline now emits `stage1/2/3_infer` so the remaining artifacts stay minimal.
- Summary file: `result/summary.result_rebuild.json` (counts, estimated tokens, stage breakdown)

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

