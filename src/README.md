# Minimal Python pipeline v2 (infer/eval/result_rebuild)

## Input format
`--input` can be either:
- a JSONL file, or
- a directory containing `*.jsonl` files (each file runs independently, using filename stem as output prefix)

Each line must be a JSON object with at least one of:

- `prompt`: full instruction text (preferred), or
- `question`: task text, or
- `text`: task text

Example: `datasets/input/example_input.jsonl`

## Run (real OpenAI-compatible API)
### Use the model routing config (v2)

Edit `config/llm_models.json` to describe your **three models** (e.g. 30B 快思考 / 30B 慢思考 / 基础模型),
and how each pipeline stage routes to them.

The CLI entrypoint `src/run_pipeline.py` provides only **three generic ops**:
- `infer`
- `eval`
- `result_rebuild`

There is **no `--stage` argument**. Stage is inferred from the stage directory name:
`<run_dir>/<stage>/`.

Directory layout:

- `<run_dir>/stage1/input.jsonl`
- `<run_dir>/stage1/infer.jsonl`
- `<run_dir>/stage1/status.jsonl`
- `<run_dir>/stage2/input.jsonl` (emitted by `eval` of stage1 when needed)
- ...

Example (stage1):

```bash
# Prepare stage1 input
mkdir -p datasets/out/demo_v2/stage1
cp datasets/input/example_input.jsonl datasets/out/demo_v2/stage1/input.jsonl

# 1) infer
PYTHONPATH=src python3 src/run_pipeline.py --mode infer --input datasets/out/demo_v2/stage1 --llm-config config/llm_models.json

# 2) eval (majority vote)
PYTHONPATH=src python3 src/run_pipeline.py --mode eval --input datasets/out/demo_v2/stage1 --llm-config config/llm_models.json
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

Outputs (all JSONL) are written under the stage directories inside your run directory.

### Rebuild result/ from existing artifacts

If you already have stage `infer.jsonl` + `status.jsonl` and want to regenerate `result/result.stage_final.jsonl`:

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode result_rebuild \
  --input datasets/out/demo_v2
```

Rebuild rules:
- Prefer using per-uuid status metadata (`vote_majority_answer_idxs`) to decide exactly which attempts
  should be written (requires `majority_count >= min_votes_to_accept`).
- Result schema: `{"uuid": ..., "text": "<question + reasoning + answer>"}`
- To avoid collisions, each attempt appends its index to the uuid: `<uuid>-<attempt_idx>`
Output compaction:
- `options.compact_outputs=true` only affects optional debug artifacts in other modules.
  Pipeline v2 always writes the core artifacts: `input.jsonl`, `infer.jsonl`, `status.jsonl`, and `result/result.stage_final.jsonl`.

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

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode result_rebuild \
  --input datasets/out/demo_modular \
  --out datasets/out/demo_modular \
  --llm-config config/llm_models.json
```

