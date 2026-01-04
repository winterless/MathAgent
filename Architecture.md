## MathAgent Architecture

![Architecture](others/diagrams/architecture.png)

### 源码分层（真实目录）

- **CLI / Orchestration**：`src/run_pipeline.py`
  - 唯一入口：支持 `--mode full`（一口气跑完）与 `--mode stageX_infer/stageX_eval`（模块化产出下一步工件后停止）
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
  - `finish_early`：solve prompt 的“提前结束/猜答案”提示策略（软控制）
  - `think_tag_default / think_tag_by_stage / think_tag_by_profile`：向 user content 注入 tag（例如 `/think`、`/no_think`；是否生效取决于你的后端模型实现）
  - `debug_print_prompts`：把请求 prompt 打到 stderr
  - `debug_print_outputs`：把模型输出打到 stderr
  - `debug_stream_outputs`：仅在 `debug_print_outputs=true` 时生效；会用 SSE `stream=true` 调模型并把 token 流式打印到 stderr（落盘仍是最终完整文本）
  - `debug_print_prompts_max_chars / debug_print_outputs_max_chars`：限制 debug 打印长度（避免刷屏）

一个“够用且和实现一致”的 options 示例（可直接放在 `config/llm_models.json` 顶层）：

```json
{
  "options": {
    "finish_early": true,
    "think_tag_by_stage": {
      "stage1_solve": "/no_think",
      "stage2_solve": "/think",
      "stage3_solve": "/think"
    },
    "debug_print_prompts": true,
    "debug_print_prompts_max_chars": 4000,
    "debug_print_outputs": true,
    "debug_print_outputs_max_chars": 2000,
    "debug_stream_outputs": false
  }
}
```

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

#### CLI 选项速查（全部用可复制的真实例子）

> 约定：建议都用 `PYTHONPATH=src` 运行（让 `src/` 作为 import root）。

最常用两种跑法：

- **`--mode full`（默认）**：一口气跑完 Stage1 → Stage2 → Stage3 → `result/`

```bash
PYTHONPATH=src python3 src/run_pipeline.py \
  --mode full \
  --input datasets/example_input.jsonl \
  --out datasets/out/demo \
  --llm-config config/llm_models.json
```

- **模块化（Route-A）**：每次只跑一步，产出“下一步工件”后停止（适合多机共享 out 目录）

```bash
PYTHONPATH=src python3 src/run_pipeline.py --mode stage1_infer --input datasets/example_input.jsonl --out datasets/out/demo_modular --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode stage1_eval  --input datasets/out/demo_modular     --out datasets/out/demo_modular --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode stage2_infer --input datasets/out/demo_modular     --out datasets/out/demo_modular --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode stage2_eval  --input datasets/out/demo_modular     --out datasets/out/demo_modular --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode stage3_infer --input datasets/out/demo_modular     --out datasets/out/demo_modular --llm-config config/llm_models.json
PYTHONPATH=src python3 src/run_pipeline.py --mode stage3_eval  --input datasets/out/demo_modular     --out datasets/out/demo_modular --llm-config config/llm_models.json
```

通用选项（所有 mode 都支持）：

- `--out <dir>`：输出根目录（会创建 `stage1/ stage2/ stage3/ result/`）
- `--llm-config <path>`：模型路由配置（默认 `config/llm_models.json`）
- `--sleep <seconds>`：LLM 调用之间 sleep（限流）

兼容入口（只在 `--mode full` 可用）：

- **`--stage1 <stage1_dir>`**：从已有 `stage1_output.stage1.jsonl` 开始，跳过 Stage1，直接跑 Stage2 → Stage3

```bash
PYTHONPATH=src python3 src/run_pipeline.py --mode full --stage1 datasets/out/demo_modular/stage1 --out datasets/out/replay_from_stage1 --llm-config config/llm_models.json
```

#### Stage0（输入副本）

- 单文件模式输出：`stage0.jsonl`
- 目录模式输出：`<prefix>.stage0.jsonl`（输入文件 copy）
- 备注：当用 `--stage1` 启动时，这里的“输入副本”是 `stage1_output.stage1.jsonl`（而不是原始 `--input` 数据集）

#### Stage1/2/3：统一的数据流（input → infer → raw_generations → eval/archive → status → next_input）

通用组件（Stage1/2/3 共用）：

- **抽取**：优先 `extract_boxed_answer(raw)`，否则 `extract_final_answer(raw)`；MCQ 再做 `standardize_choice_answer(...)`
- **判别**：先 `rule_equivalent(...) -> True/False/None`；仅当规则返回 `None` 时才走 LLM 兜底裁决（`*_judge`）
- **路由**：只看 `final_vote_count < min_votes_to_accept`（即 `status.stageX.jsonl` 里的 `final_vote_count`），决定是否进入下一阶段

Stage1（solve + eval/路由）：

- **infer（modular）**：`stage1_infer` 只负责采样解题
  - 输入：原始输入 JSONL（或 out root 下的 `*.stage0.jsonl`）
  - 输出：`--out/stage1/<prefix>.stage1_infer.stage1.jsonl`
- **eval（modular）**：`stage1_eval` 读 `stage1_infer`，产出评估与路由
  - 输出：
    - `--out/stage1/<prefix>.stage1_raw_generations.stage1.jsonl`
    - `--out/stage1/<prefix>.stage1_output.stage1.jsonl`
    - `--out/stage1/<prefix>.status.stage1.jsonl`
    - `--out/stage2/<prefix>.stage2_input.stage2.jsonl`（下一步任务清单）
- **full 模式**：会直接产出 `stage1_raw_generations / stage1_output / status.stage1`（不产出 `stage1_infer` 和 `stage2_input`）
- **accepted_bank**：Stage1 认为“太简单”，不会写入 `accepted_bank`

Stage2（只处理难题；可直接 accepted 或进入 Stage3）：

- **进入条件**：Stage1 的 `final_vote_count < min_votes_to_accept`
- **infer（modular）**：`stage2_infer` 优先读 `stage2_input.stage2.jsonl`（兼容：也可直接输入 `stage1_output.stage1.jsonl` 来反推任务）
  - 输出：`--out/stage2/<prefix>.stage2_infer.stage2.jsonl`（以及对应的 `stage2_input` 拷贝/派生文件）
- **eval（modular）**：`stage2_eval` 读 `stage2_infer`，生成归档与下一步任务
  - 输出：
    - `--out/stage2/<prefix>.stage2_raw_generations.stage2.jsonl`
    - `--out/stage2/<prefix>.stage2_archive.stage2.jsonl`
    - `--out/stage2/<prefix>.status.stage2.jsonl`
    - `--out/stage3/<prefix>.stage3_input.stage3.jsonl`
- **full 模式**：会直接产出 `stage2_raw_generations / stage2_archive / status.stage2`
- **若 accepted（本阶段收敛）**：写入 `--out/<prefix>.accepted_bank.stage_final.jsonl`（`accepted_from="stage2"`）；若 `final_source="majority"` 同时写入 `--out/result/<prefix>.result.stage_final.jsonl`

Stage3（复用 Stage2 逻辑；不收敛则 gold fallback）：

- **进入条件**：Stage2 的 `final_vote_count < min_votes_to_accept`
- **infer（modular）**：`stage3_infer` 读 `stage3_input`，写 `stage3_infer`
  - 输出：`--out/stage3/<prefix>.stage3_infer.stage3.jsonl`
- **eval（modular）**：`stage3_eval` 读 `stage3_infer`，写归档与最终落盘
  - 输出：
    - `--out/stage3/<prefix>.stage3_raw_generations.stage3.jsonl`
    - `--out/stage3/<prefix>.stage3_archive.stage3.jsonl`
    - `--out/stage3/<prefix>.status.stage3.jsonl`
    - `--out/<prefix>.accepted_bank.stage_final.jsonl`（`accepted_from="stage3"` 或 `stage3_gold_fallback`）
    - `--out/result/<prefix>.result.stage_final.jsonl`（仅当 `final_source="majority"`）
- **full 模式**：会直接产出 `stage3_raw_generations / stage3_archive / status.stage3`，并写入 `accepted_bank/result`

#### Final Result 归档：只收 Stage2/Stage3 的“投票收敛答案”

- 当 Stage2 或 Stage3 满足 `final_source="majority"`（等价于 `final_vote_count >= min_votes_to_accept`）时，会把该题的结果归档到：
  - `--out/result/result.stage_final.jsonl`（单文件模式）
  - `--out/result/<prefix>.result.stage_final.jsonl`（目录模式/多输入前缀模式）
- 仅收录 `final_source="majority"` 的结果（answer_fallback 不会进入 result；Stage1 也不会进入 result）
- **每个 uuid 一行**，包含：
  - `uuid`, `question`
  - `final_answer`（因为 result 只收录 `final_source="majority"`，这里恒为 majority 的最终答案）
  - `attempts[]`：每次采样的 `raw_text / boxed_answer / verdict / final_answer`

### 走读代码路线（建议顺序）

1. **入口与文件落盘**：`src/run_pipeline.py`
   - `main()`：解析 `--mode/--input/--out/--llm-config`；full 与 modular 两套路由；统一阈值 `min_votes_to_accept`
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



