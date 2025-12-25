## MathAgent Architecture

![Architecture](others/diagrams/architecture.png)

### 源码分层（真实目录）

- **CLI / Orchestration**：`src/run_pipeline.py`
  - 唯一入口：读取输入 JSONL → Stage1 → Stage2 → Stage3 → 落盘各阶段 JSONL
  - 统一实现了 Stage1/2/3 的 **抽取/判别/路由策略**
- **Core (Domain)**：`src/core/`
  - `stages.py`：归一化、选项映射、答案抽取、规则等价判别、boxed 统计解析
  - `prompt_assemble.py`：拼装 Stage1 evaluator 的 `prompt`（sample.jsonl 风格）
  - `voting.py`：多数投票（Stage1 的 8 次采样）
- **IO**：`src/dataio/`
  - `jsonl_io.py`：JSONL 读写（`write_jsonl_atomic` 原子写入）
  - `sample_schema.py`：输出 wrapper/schema 归一（对齐 sample 风格字段）
- **Infra (LLM)**：`src/infra/`
  - `llm_router.py`：从 `config/llm_models.json` 读取 models/routes/stage_params/thresholds/options，并对外提供统一 API
  - `llm_client.py`：OpenAI-compatible HTTP 客户端（支持 retry/backoff；debug 打印；支持 SSE stream 流式输出）

### 配置中心：`config/llm_models.json`

- **models**：不同“模型实例”的基础参数（如 base/think_fast/think_slow，对应不同 endpoint/model/timeout/retry）
- **routes**：`stage_name -> model_key` 的路由（例如 `stage2_solve` 走 think_fast）
- **stage_params**：每个 stage 的采样参数（`n/temperature/max_tokens`）
- **thresholds**
  - `min_ok_to_accept`：**接受阈值**。在 N=8 时，`min_ok_to_accept=5` 等价于 **bad≥4 则进入下一阶段**（Stage1→2、Stage2→3、Stage3→discard）
- **options**
  - `finish_early`：solve 类 prompt 的“提前结束/猜答案”提示策略（软控制）
  - `think_tag_default` / `think_tag_by_stage`：用户侧 tag 注入（例如 `/no_think`）
  - `debug_print_prompts` / `debug_print_outputs`：打印 prompt / raw 输出
  - `debug_stream_outputs`：当开启 debug_print_outputs 时，**对输出采用流式打印**（`stream=true`）

### 运行时数据流（Pipeline）

#### 输入

- `--input <path>.jsonl`
  - 每行至少包含 `question` 和 `answer`（`normalize_record()` 会兼容 `prompt/text` 等字段）

#### Stage0（输入副本）

- 输出：`example_input.stage0.jsonl`（输入文件 copy）

#### Stage1：solve(8) + judge(规则优先) + eval(长评测)

- **solve**（`stage1_solve` / `prompt_mode="problem"`）
  - 对每题采样 `n=8` 次，保存原始输出
- **抽取器（Stage1/2/3 共用）**
  - 优先 `extract_boxed_answer(raw)`，否则 `extract_final_answer(raw)`
  - 再对 MCQ 做 `standardize_choice_answer(...)`
- **判别器（Stage1/2/3 共用）**
  - 先 `rule_equivalent(pred, gold, choice_map=...) -> True/False/None`
  - 仅当规则返回 `None` 时，才调用 LLM 兜底裁决（`*_judge`，只允许输出：一致/不一致/不确定）
- **eval**（`stage1_eval` / `prompt_mode="raw_prompt_eval"`）
  - 只做“对比与统计”，输出长 Markdown；若缺失 boxed 统计，会最多重试 1 次
- Stage1 产物
  - `stage1_raw_generations.stage1.jsonl`：每题的 8 次 raw + 抽取结果 + attempts verdict + `llm_call_counts`
  - `stage1_output.stage1.jsonl`：sample 风格 wrapper（包含 evaluator 的长 `content`）

#### Stage2：只处理“难题”，boxed_solve + 规则判别 + 归档（带 checkpoint）

- **进入条件（来自 Stage1）**
  - 使用 Stage1 eval 的 boxed 统计（或 fallback 到 Stage1 judge 得到的 ok/bad）
  - `ok < min_ok_to_accept` 则进入 Stage2
- **solve**（`stage2_solve` / `prompt_mode="boxed_solve"`）
  - 只给 question（含选项映射），强制最终答案进 `\\boxed{...}`
- **判别/归档**
  - 同 Stage1 的抽取 + 规则优先 + LLM 兜底
  - 产出 `stage2_archive.stage2.jsonl`（每题聚合 attempts、ok/bad、llm_call_counts）
- **checkpoint**
  - Stage2 全部跑完后，会先写一次 `stage2_archive.stage2.jsonl`，再进入 Stage3（避免 Stage3 很慢导致 Stage2 文件迟迟不可见）

#### Stage3：复用 Stage2 逻辑，仍不通过则丢弃

- **进入条件**：Stage2 后 `ok < min_ok_to_accept`
- **执行内容**：同 Stage2（boxed_solve + 抽取判别 + 归档）
- **结束**
  - 通过：写入 `accepted_bank.stage_final.jsonl`
  - 不通过：写入 `discarded_hard.stage_final.jsonl`

### 走读代码路线（建议顺序）

1. **入口与文件落盘**：`src/run_pipeline.py`
   - `main()`：输出文件名、Stage1→Stage2→Stage3 主循环、路由阈值（`min_ok_to_accept`）
   - `_llm_judge_equivalence()`：LLM 裁决的严格解析
2. **抽取与规则判别**：`src/core/stages.py`
   - `extract_boxed_answer()` / `extract_final_answer()`：最终答案抽取优先级
   - `rule_equivalent()`：规则优先判别（MCQ 直接决策，无法决策才返回 None）
   - `extract_boxed_counts*()`：Stage1 eval 的 ok/bad 统计解析
3. **Stage1 evaluator prompt**：`src/core/prompt_assemble.py`
   - `assemble_stored_prompt(...)`：把 8 个答案拼成 evaluator 输入
4. **LLM 配置/路由/参数注入**：`src/infra/llm_router.py`
   - `stage_params()` / `threshold_int()` / `option_bool()` / `think_tag_for_stage()`
5. **LLM 调用与 debug**：`src/infra/llm_client.py`
   - `generate_n()`：按 mode 组装 prompt、debug 打印、可选 streaming 输出
   - `chat_once()`：HTTP 调用 + retry/backoff；`stream=true` 的 SSE 解析



