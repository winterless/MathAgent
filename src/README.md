# Minimal Python pipeline v2

## What it does
`src/run_pipeline.py` exposes **three ops**:
- **`infer`**: produce `<out>/infer.jsonl`
- **`eval`**: consume `<input>/infer.jsonl` and produce `<out>/status.jsonl`
- **`result_rebuild`**: consume `<run_dir>/**/{infer.jsonl,status.jsonl}` and produce `<run_dir>/result/result.stage_final.jsonl`

Stage is **not** a separate CLI argument. It is implied by the **output directory name** (e.g. `.../stage1`, `.../stage2`).
Chaining stages is **data-driven**: downstream `infer` picks rows where upstream `status.jsonl` has `accepted=false`.

## Input data format (raw dataset)
When `--mode infer` consumes a raw dataset (`--input` is a JSONL file or a directory), each line must be a JSON object with at least one of:
- `question`
- `prompt`
- `text`

Which field is used is controlled by `config/llm_models.json`:
- `options.input.question_key_priority` (default `["question","prompt","text"]`)
- `options.input.raw_input_glob` (when `--input` is a directory; default `"*.jsonl"`)

## End-to-end reference (stage1 → stage3)
All commands below go together as a single reference run:

```bash
RUN_DIR=datasets/out/demo_3

# Stage1
PYTHONPATH=src python3 src/run_pipeline.py --mode infer --input datasets/input --out "$RUN_DIR/stage1" --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode eval  --input "$RUN_DIR/stage1" --out "$RUN_DIR/stage1" --llm-config config/llm_models.json

# Stage2 (infer reads infer+status from stage1; writes stage2/infer.jsonl)
PYTHONPATH=src python3 src/run_pipeline.py --mode infer --input "$RUN_DIR/stage1" --out "$RUN_DIR/stage2" --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode eval  --input "$RUN_DIR/stage2" --out "$RUN_DIR/stage2" --llm-config config/llm_models.json

# Stage3 (infer reads infer+status from stage2; writes stage3/infer.jsonl)
PYTHONPATH=src python3 src/run_pipeline.py --mode infer --input "$RUN_DIR/stage2" --out "$RUN_DIR/stage3" --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode eval  --input "$RUN_DIR/stage3" --out "$RUN_DIR/stage3" --llm-config config/llm_models.json

# Final result
PYTHONPATH=src python3 src/run_pipeline.py --mode result_rebuild --input "$RUN_DIR"
```

Optional flag:
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

If you already have stage `infer.jsonl` + `status.jsonl` and want to regenerate `result/result.stage_final.jsonl`, use the `result_rebuild` command in the end-to-end block above.

Rebuild rules:
- Prefer using per-uuid status metadata (`vote_majority_answer_idxs`) to decide exactly which attempts
  should be written (requires `majority_count >= min_votes_to_accept`).
- Result schema: `{"uuid": ..., "text": "<question + reasoning + answer>"}`
- To avoid collisions, each attempt appends its index to the uuid: `<uuid>-<attempt_idx>`
Output compaction:
- `options.compact_outputs=true` only affects optional debug artifacts in other modules.
  Pipeline v2 always writes the core artifacts: `infer.jsonl`, `status.jsonl`, and `result/result.stage_final.jsonl`.

