## MathAgent Architecture

![Architecture](others/diagrams/architecture.png)

### 源码分层（真实目录）

- **CLI / Orchestration**：`src/run_pipeline.py`
  - 唯一入口：读取输入 JSONL → Stage1 → Stage2 → Stage3 → 落盘各阶段 JSONL
  - 统一实现了 Stage1/2/3 的 **抽取/判别/路由策略**
- **Core (Domain)**：`src/core/`
  - `stages.py`：归一化、选项映射、答案抽取、规则等价判别、boxed 统计解析
  - `prompt_assemble.py`：拼装 Stage1 evaluator 的 `prompt`（sample.jsonl 风格）
  - `voting.py`：多数投票（Stage1 的 n 次采样，n 来自 `stage_params.stage1_solve.n`）
- **IO**：`src/dataio/`
  - `jsonl_io.py`：JSONL 读写（`write_jsonl_atomic` 原子写入）
  - `sample_schema.py`：输出 wrapper/schema 归一（对齐 sample 风格字段）
- **Infra (LLM)**：`src/infra/`
  - `llm_router.py`：从 `config/llm_models.json` 读取 models/routes/stage_params/thresholds/options，并对外提供统一 API
  - `llm_client.py`：OpenAI-compatible HTTP 客户端（支持 retry/backoff；debug 打印；支持 SSE stream 流式输出）

### 配置中心：`config/llm_models.json`

- **models**：不同“模型实例”的基础参数（如 base/think_fast/think_slow，对应不同 endpoint/model/timeout/retry）
- **routes**：`stage_name -> model_key` 的路由（例如 `stage2_solve` 走 think_fast）
- **stage_params**：每个 stage 的采样参数（`n/temperature/max_tokens`）。例如把 `stage1_solve.n` 从 4 改为 8，就会对每题采样 8 次（Stage2/Stage3 同理）。
- **thresholds**
  - `min_votes_to_accept`：**接受阈值（共识强度）**。当该 stage 的采样数为 N（例如 `stage_params.stage2_solve.n=N`）时，`min_votes_to_accept=k` 表示 **majority voting 的票数 ≥k 才算“本阶段收敛/可接受”**；否则进入下一阶段（Stage1→2、Stage2→3；Stage3 则进入 gold fallback）
- **options**
  - `finish_early`：solve 类 prompt 的“提前结束/猜答案”提示策略（软控制）
  - `think_tag_default` / `think_tag_by_stage`：用户侧 tag 注入（例如 `/no_think`），stage 维度覆盖（可选）
  - `think_tag_by_profile`：按路由 profile（`routes` 的 model_key：base/think_fast/think_slow）注入 tag（可选）
  - `debug_print_prompts` / `debug_print_outputs`：打印 prompt / raw 输出
  - `debug_stream_outputs`：当开启 debug_print_outputs 时，**对输出采用流式打印**（`stream=true`）

### 本地 Python 推理脚本调用协议（stdin/stdout，必须遵守）

当底层推理选择使用“本地 Python 脚本 runner”时，调用方会通过子进程执行：

- `<python_bin> <py_script>`
  - `python_bin`：可选，默认 `python3`
  - `py_script`：一个可执行的 `.py` 脚本路径（建议绝对路径）

并通过 **stdin/stdout** 与脚本通信（一次调用对应一次请求；需要采样 N 次时会调用 N 次）。

#### 脚本输入（stdin，UTF-8 JSON）

脚本必须从 stdin 读入一个 JSON 对象（stdin 会在写入完请求后关闭；脚本可按 EOF 结束读取）。
请求字段如下（脚本应做到“向前兼容”：允许出现额外字段并忽略它们）：

- `stage_name`：字符串，例如 `"stage1_solve"` / `"stage2_solve"` / `"stage1_eval"` 等（用于区分不同 stage 的系统提示与任务语义）
- `model`：字符串（仅用于标识/日志；脚本可忽略）
- `messages`：数组（OpenAI Chat Messages 结构；**prompt 就在这里**）
  - 典型为：`[{ "role": "system", "content": "..." }, { "role": "user", "content": "..." }]`
- `temperature`：数字
- `max_tokens`：整数
- `stream`：布尔（当前恒为 `false`；脚本无需实现流式输出）

脚本**必须**在合理时间内输出结果并退出（超时会被中断并被视为失败；可触发重试）。

一个最小可用的请求样例（仅示意字段含义）：

```json
{
  "stage_name": "stage2_solve",
  "model": "any_name_for_logging",
  "messages": [
    {"role": "system", "content": "你是一个数学解题助手..."},
    {"role": "user", "content": "题目：..."}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false
}
```

#### 脚本输出（stdout，UTF-8）

脚本 stdout 必须输出“模型最终文本内容”（建议 stdout **只输出结果本身**；所有调试日志请写到 stderr）。
支持两种等价格式（二选一）：

1) **纯文本**：stdout 直接输出 assistant 的 content（任意文本）。

2) **JSON**：stdout 输出一个 JSON 对象，至少满足以下其一：
   - OpenAI-like：`{"choices":[{"message":{"content":"..."}}]}`
   - 或简化形式：`{"content":"..."}`

stderr 可用于打印调试信息。

#### 成功/失败语义（调用方如何判断）

- **成功**：进程退出码为 `0`，且 stdout 能解析出最终文本内容（纯文本或 JSON 皆可）。
- **失败**：退出码非 `0` 或发生超时/异常；调用方会把它当作一次失败（可触发 retry/backoff）。

### 运行时数据流（Pipeline）

#### 输入

- `--input <path>.jsonl`（单文件）
  - **每行必须包含**：`question`（题面文本）与 `answer`（标准答案/label）
  - **强烈建议包含**：`uuid`（用于断点续跑与各阶段对齐；缺失会导致所有行共享同一个 key）
  - 其它字段会按 `datasets/sample.jsonl` 的 canonical keys（见 `src/dataio/sample_schema.py` 的 `CANONICAL_KEYS`）做“保留/补默认值”；**不会自动把 `prompt/text` 映射成 `question`**
- `--input <dir>/`（目录）
  - 目录下每个 `*.jsonl` 文件都会独立跑一遍 pipeline
  - **文件名 stem**（去掉 `.jsonl` 的前缀）会作为该输入的 **输出前缀**，避免多文件输出互相覆盖（并能各自断点续传）
- `--stage1 <stage1_dir>/`（从已有 Stage1 结果开始，跳过 Stage1）
  - 读取该目录下所有 `*stage1_output.stage1.jsonl` 作为“输入行”（它们内部应包含 `uuid/question/answer`）
  - 会直接执行 Stage2 → Stage3（并复用同样的 checkpoint / 输出命名）；**Stage0 也会 copy 这份 stage1_output 文件作为追踪输入**
  - 若该目录下存在对应的 `status.stage1.jsonl` / `<prefix>.status.stage1.jsonl`：仅对 `next_stage=stage2` 的 UUID 进入 Stage2
  - 若没有 status：默认把所有 UUID 当作 Stage2 候选

#### Stage0（输入副本）

- 单文件模式输出：`stage0.jsonl`
- 目录模式输出：`<prefix>.stage0.jsonl`（输入文件 copy）
- 备注：当用 `--stage1` 启动时，这里的“输入副本”是 `stage1_output.stage1.jsonl`（而不是原始 `--input` 数据集）

#### Stage1：solve(n) + judge(规则优先) + eval(长评测)

- **solve**（`stage1_solve` / `prompt_mode="problem"`）
  - 对每题采样 `n` 次（来自 `config/llm_models.json` 的 `stage_params.stage1_solve.n`），保存原始输出
- **抽取器（Stage1/2/3 共用）**
  - 优先 `extract_boxed_answer(raw)`，否则 `extract_final_answer(raw)`
  - 再对 MCQ 做 `standardize_choice_answer(...)`
- **判别器（Stage1/2/3 共用）**
  - 先 `rule_equivalent(pred, gold, choice_map=...) -> True/False/None`
  - 仅当规则返回 `None` 时，才调用 LLM 兜底裁决（`*_judge`，只允许输出：一致/不一致/不确定）
- **eval**（`stage1_eval` / `prompt_mode="raw_prompt_eval"`）
  - 只做“对比与统计”，输出长 Markdown；若缺失 boxed 统计，会最多重试 1 次
- Stage1 产物
  - 位于 `--out/stage1/` 下：
    - `stage1_raw_generations.stage1.jsonl`：每题的 n 次 raw + 抽取结果 + attempts verdict + `llm_call_counts`
    - `stage1_output.stage1.jsonl`：sample 风格 wrapper（包含 evaluator 的长 `content`）
  - 目录模式：以上文件名都会变为 `<prefix>.*.jsonl`（例如 `<prefix>.stage1_output.stage1.jsonl`）
  - 断点续传：`<prefix>.status.stage1.jsonl`（每个 uuid 完成即追加一行，记录 ok/bad、投票信息与 next_stage）

#### Stage2：只处理“难题”，boxed_solve + 规则判别 + 归档（带 checkpoint）

- **进入条件（来自 Stage1）**
  - 以 Stage1 的 `majority_answer.majority_count` 作为共识强度（其上限等于该 stage 的 `stage_params.<stage>_solve.n`）
  - `majority_count < min_votes_to_accept` 则进入 Stage2
- **solve**（`stage2_solve` / `prompt_mode="boxed_solve"`）
  - 只给 question（含选项映射），强制最终答案进 `\\boxed{...}`
- **判别/归档**
  - 同 Stage1 的抽取 + 规则优先 + LLM 兜底
  - 产出 `stage2_archive.stage2.jsonl`（每题聚合 attempts、ok/bad、llm_call_counts）
- 断点续传：`<prefix>.status.stage2.jsonl`（每个 uuid 完成即追加一行）
- **checkpoint**
  - Stage2 是逐题 append `stage2_archive.stage2.jsonl` + `status.stage2.jsonl`，所以文件会实时可见；支持中断后继续跑
- 文件位置：均位于 `--out/stage2/` 下（目录模式同样带 `<prefix>.` 前缀）

#### Stage3：复用 Stage2 逻辑，仍不通过则 gold fallback

- **进入条件**：Stage2 后 `majority_count < min_votes_to_accept`
- **执行内容**：同 Stage2（boxed_solve + 抽取判别 + 归档）
- **结束**
  - 若 `majority_count >= min_votes_to_accept`：最终答案取 voting 的 majority
  - 若 `majority_count < min_votes_to_accept`：认为“无法 voting 出结果”，最终答案 **fallback 到输入的 `answer`**
  - 最终都会写入 `accepted_bank.stage_final.jsonl`（`accepted_from` 会标记 `stage3` 或 `stage3_gold_fallback`）
  - 目录模式：写入 `<prefix>.accepted_bank.stage_final.jsonl`
  - 备注：不再产出 `discarded_hard.stage_final.jsonl`（Stage3 改为 gold fallback 兜底）
  - 断点续传：`<prefix>.status.stage3.jsonl`
  - 文件位置：Stage3 的 archive/status 位于 `--out/stage3/` 下；final bank 位于 `--out/` 根目录

#### Final Result 归档：只收 Stage2/Stage3 的“投票收敛答案”

- 当 Stage2 或 Stage3 满足 `majority_count >= min_votes_to_accept` 时，会把该题的结果归档到：
  - `--out/result/result.stage_final.jsonl`（单文件模式）
  - `--out/result/<prefix>.result.stage_final.jsonl`（目录模式/多输入前缀模式）
- 仅收录 `final_source="majority"` 的结果（answer_fallback 不会进入 result；Stage1 也不会进入 result）
- **每个 uuid 一行**，包含：
  - `uuid`, `question`
  - `final_answer`（因为 result 只收录 `final_source="majority"`，这里恒为 majority 的最终答案）
  - `attempts[]`：每次采样的 `raw_text / boxed_answer / verdict / final_answer`

### 走读代码路线（建议顺序）

1. **入口与文件落盘**：`src/run_pipeline.py`
   - `main()`：输出文件名、Stage1→Stage2→Stage3 主循环、路由阈值（`min_votes_to_accept`）
   - `_llm_judge_equivalence()`：LLM 裁决的严格解析
2. **抽取与规则判别**：`src/core/stages.py`
   - `extract_boxed_answer()` / `extract_final_answer()`：最终答案抽取优先级
   - `rule_equivalent()`：规则优先判别（MCQ 直接决策，无法决策才返回 None）
   - `extract_boxed_counts*()`：Stage1 eval 的 ok/bad 统计解析
3. **Stage1 evaluator prompt**：`src/core/prompt_assemble.py`
   - `assemble_stored_prompt(...)`：把 n 个答案拼成 evaluator 输入（n 来自 `stage_params.stage1_solve.n`）
4. **LLM 配置/路由/参数注入**：`src/infra/llm_router.py`
   - `stage_params()` / `threshold_int()` / `option_bool()` / `think_tag_for_stage()`
5. **LLM 调用与 debug**：`src/infra/llm_client.py`
   - `generate_n()`：按 mode 组装 prompt、debug 打印、可选 streaming 输出
   - `chat_once()`：HTTP 调用 + retry/backoff；`stream=true` 的 SSE 解析



