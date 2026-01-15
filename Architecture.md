## MathAgent Architecture (as-implemented)

![Architecture](others/diagrams/architecture.png)

### 源码分层（真实目录）

- **CLI / Orchestration**：`src/run_pipeline.py`
  - 唯一入口，统一实现 **full** 与 **Route-A modular**（`stage1_infer/stage1_eval/stage2_infer/stage2_eval/stage3_infer/stage3_eval`）
  - 负责工件落盘、断点续跑（done-set/status）、路由（是否进入下一阶段）
  - 启动时可进行 **connection-error 数据清理**（见“运行时稳定性”）
- **Core (Domain)**：`src/core/`
  - `stages.py`：输入归一化、选项映射、答案抽取（`FINAL:` / `\\boxed{}`）、规则等价判别、Stage1 boxed 统计解析
  - `prompt_assemble.py`：拼装 Stage1 evaluator `prompt`（sample.jsonl 风格）
  - `voting.py`：多数投票（majority vote）
- **IO**：`src/dataio/`
  - `jsonl_io.py`：JSONL 读写（`write_jsonl_atomic` 原子写入）
  - `sample_schema.py`：输入/输出 schema wrapper 归一（对齐 sample 风格字段）
- **Infra (LLM)**：`src/infra/`
  - `llm_router.py`：读取 `config/llm_models.json`（models/routes/stage_params/thresholds/options），并对外提供统一 `generate_n()`
  - `llm_client.py`：OpenAI-compatible HTTP 客户端（retry/backoff、debug 打印、可选 SSE stream），以及 **connection refused 恢复等待/可选重启**（env 控制）

### 配置中心：`config/llm_models.json`

- **models**：不同“模型实例”的基础参数（如 base/think_fast/think_slow，对应 endpoint/model/timeout/retry）
- **routes**：`stage_name -> model_key` 路由（例如 `stage2_solve` 走 think_fast）
- **stage_params**：每个 stage 的采样参数（`n/temperature/max_tokens`）
- **thresholds**
  - `min_votes_to_accept`：接受阈值（共识强度）。Stage2/Stage3 的 `final_vote_count >= min_votes_to_accept` 视为“本阶段收敛”
- **options**
  - `finish_early`：solve prompt 的“提前结束/猜答案”提示（软控制）
  - `think_tag_default / think_tag_by_stage / think_tag_by_profile`：向 user content 注入 tag（例如 `/think`、`/no_think`）
  - `debug_print_prompts / debug_print_outputs / debug_stream_outputs`：调试打印与可选流式输出（仅影响日志，不影响落盘结构）
  - `generate_n_max_workers`：`LLMClient.generate_n()` 内部并发上限（注意并发过高会放大 vLLM 负载峰值）

### 工件（Artifacts）与断点续跑（Done Sets）

实现中“是否跳过某个 uuid”的依据不是“文件存在”，而是 **done-set/status**：

- **Stage1 eval**：用 `stage1/<prefix>.status.stage1.jsonl` 的 uuid map 判断是否已处理
- **Stage2 infer**：用 `stage2/<prefix>.stage2_infer.stage2.jsonl` 的 uuid set 判断是否已 infer
- **Stage2 eval**：用 `stage2/<prefix>.status.stage2.jsonl` 的 uuid set 判断是否已 eval
- **Stage3 infer**：用 `stage3/<prefix>.stage3_infer.stage3.jsonl` 的 uuid set 判断是否已 infer
- **Stage3 eval**：用 `stage3/<prefix>.status.stage3.jsonl` 的 uuid set 判断是否已 eval

因此：一旦某个 uuid 被写进了上述“done 文件”，后续重跑默认会跳过它（除非你清理对应行）。

### Pipeline 数据流（实现口径）

#### Stage0：输入副本

- 单文件模式输出：`stage0.jsonl`
- 目录模式输出：`<prefix>.stage0.jsonl`

#### Stage1（solve + eval/路由）

- **solve（full 或 stage1_infer）**
  - 对每题采样 `stage_params.stage1_solve.n`
  - 落盘：`stage1/<prefix>.stage1_infer.stage1.jsonl`（modular）或 full 模式中对应的 raw 工件
  - 内部会做：
    - 选项映射注入：`append_choice_map_if_any(normalize_for_model(question))`
    - 抽取：`extract_boxed_answer` / `extract_final_answer`，选择题会进一步 `standardize_choice_answer`
    - 投票：`majority_vote(...)`
- **eval（stage1_eval）**
  - **优先解析**：从 `stage1_output.stage1.jsonl` 的 `output` 里解析 `\\boxed{解答正确：x，解答错误：y}`（不需要 LLM）
  - **解析失败兜底**：才会用 stored `prompt + [GOLD_STANDARD_ANSWER]=...` 调 `stage1_eval`
  - 路由（实现口径）：`ok >= min_votes_to_accept` 则 accepted，否则进入 Stage2（并写 `stage2_input.stage2.jsonl`）

> 注意：Stage1 的“是否进入 Stage2”是基于 **eval 统计的 ok**，不是基于 Stage2/3 的 `final_vote_count` 规则。

#### Stage2（infer + eval/归档/路由）

- **infer（stage2_infer）**
  - 输入优先：`stage2_input.stage2.jsonl`；兼容：也可直接给 `stage1_output.stage1.jsonl`，代码会从 `status.stage1` 反推任务列表
  - 对每题采样 `stage_params.stage2_solve.n`，得到 `raw_model_outputs[]`
  - **抽取器**：调用 LLM 做批量抽取（`stage2_extract`），把每个 raw 输出抽取为 canonical answer（JSON 数组）
  - 落盘：`stage2/<prefix>.stage2_infer.stage2.jsonl`
- **eval（stage2_eval）**
  - 对每个采样答案，先 `rule_equivalent(pred,gold,choice_map)`；只有返回 `None` 才升级到 LLM 裁决 `stage2_judge`
  - 记录 attempts（raw_text/extracted/判别结果）并投票（会过滤空答案，避免 error 导致伪共识）
  - 生成 `majority_answer` 与 `final_answer/final_vote_count`（规则：majority_count >= min_votes_to_accept 则 final_source=majority，否则 fallback=gold）
  - 路由：`final_vote_count < min_votes_to_accept` 进入 Stage3，否则 accepted

#### Stage3（infer + eval/归档/路由）

Stage3 结构与 Stage2 类似：

- **infer**：`stage3_input.stage3.jsonl` -> `stage3_infer.stage3.jsonl`
- **eval**：写 `stage3_archive`、`status.stage3`，并产出 `accepted_bank` 与（仅当 majority 收敛时）`result`

#### Result / Accepted Bank（实现口径）

- `accepted_bank.stage_final.jsonl`：Stage2/Stage3 的最终接受记录（可能是 majority，也可能是 fallback）
- `result/*.result.stage_final.jsonl`：**只收录** `final_source="majority"` 的结果（实现明确排除了 fallback）

### 运行时稳定性（Resilience）

#### 1) vLLM connection refused 自动恢复（不改 config）

当 vLLM 挂掉或重启时，客户端可能报 `Errno 111 Connection refused`。`src/infra/llm_client.py` 支持 env 驱动恢复：

- `MATHAGENT_LLM_WAIT_ON_CONNREFUSED_S`：等待恢复的秒数（默认 0，关闭）
- `MATHAGENT_LLM_RESTART_CMD`：可选重启命令（最多尝试一次，然后轮询健康检查）
- `MATHAGENT_LLM_HEALTH_URL`：健康检查 URL（默认 `<base_url>/v1/models`）
- `MATHAGENT_LLM_CONNREFUSED_LOG`：打印恢复日志

#### 2) 启动时清理 connection-error 数据（保证 rerun 能“带上这条数据”）

由于 pipeline 会把 uuid 写入 infer/status，导致 rerun 跳过；`src/run_pipeline.py` 启动时默认会：

- 扫描 `--out` 下的 JSONL（archive/status/infer/raw_generations/accepted_bank/result）
- 找到包含 connection/network error marker 的 uuid（例如 `[LLM_ERROR ... connection refused ...]`）
- 从上述“派生工件”中删除该 uuid 对应的行（**不会删除** stage0、stage2_input、stage3_input、stage1_output）

可通过 `MATHAGENT_DISABLE_PURGE_CONN_ERRORS=1` 关闭。

### 冗余逻辑与可简化点（基于当前实现）

下面这些点会让逻辑重复、难维护，是潜在的重构收益点：

- **`run_pipeline.py` 内的重复 helper**：
  - 文件中既有全局 `_select_answer/_llm_judge_equivalence/_as_choice_letter`，又在 `_run_one_input()` 内重复定义了一套（行为基本一致）。
  - 建议：保留一处实现（最好是全局/可测试函数），统一 full/modular 调用。
- **Stage1 eval 的双入口兼容路径**：
  - `stage1_eval` 既支持从 `stage1_output` 解析 boxed counts，也支持从 `stage1_infer` 回退生成 `stage1_output`（Backward-compat 分支）。
  - 建议：如果不再需要旧工件格式，可删掉 infer-based 兼容分支，或把“工件升级”做成独立命令。
- **Stage2 infer 的双入口**：
  - 既支持读 `stage2_input`，也支持从 `stage1_output + status.stage1` 反推任务列表（兼容逻辑会扩大复杂度）。
  - 建议：固定使用 `stage2_input` 作为唯一入口，让 `stage1_eval` 明确负责产出它。
- **判别链路分散**：
  - Stage2/3 eval 里既有规则判别，又有 LLM judge，且 attempts 的字段（boxed/extracted/normalized）在不同 stage 的含义略有差异。
  - 建议：提取统一的 `Attempt` 结构与判别流水线（extract -> normalize -> rule -> llm_judge -> vote -> select）。



