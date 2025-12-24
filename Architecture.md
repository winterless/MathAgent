# MathAgent Architecture（按功能分层）

![Architecture](others/diagrams/architecture.png)

### 分层目录结构（源码）

- **CLI layer**：`src/run_pipeline.py`
  - 唯一入口，串联 Stage1/2/3，负责读写 JSONL（内部调用 `src/core|io|infra` 模块）
- **Core layer（Domain）**：`src/core/`
  - `stages.py`：题目归一化、选项映射、答案抽取（支持 `FINAL:`）、boxed 统计解析
  - `prompt_assemble.py`：拼装 `sample.jsonl` 风格的 `prompt`
  - `voting.py`：多数投票
- **IO layer**：`src/dataio/`
  - `jsonl_io.py`：JSONL 读写（原子写入）
  - `sample_schema.py`：输出 schema 对齐 `datasets/sample.jsonl`
- **Infra layer**：`src/infra/`
  - `llm_client.py`：OpenAI-compatible HTTP 客户端（vLLM/OpenAI 均可）

### Pipeline（运行时数据流）

- **输入**：`--input <path>.jsonl`（包含 `question` 与 `answer`）
- **Stage1 solve**：对每题采样 8 次（允许输出推理，最后一行 `FINAL: ...`），写入 `stage1_raw_generations.jsonl`
- **Stage1 eval**：把 8 个答案拼入 `prompt`，再调用 evaluator 输出长 Markdown，写入 `stage1_output.jsonl`
- **难度记录（最小化即可）**：从 Stage1 eval 的最后一行 `\\boxed{解答正确：x，解答错误：y}` 中解析 `x/y` 并记录难度
  - **难度定义（建议）**：`difficulty = 解答错误数量 = y = 8 - x`
  - **进入 Stage2 的条件**：`y >= 4`（等价于 `x <= 4`）

### Stage2（难题二次推理与规则优先验算）

Stage2 仅对“难题”执行，目标是：用更强/更慢的模型做 8 次推理，**答案必须落在 `\\boxed{...}` 内**，再用“规则优先 + 必要时模型裁决”的方式判定每次推理是否正确，并把完整过程归档成题库条目。

- **Stage2 需要解析/携带的信息**
  - **uuid**
  - **difficulty**（以及 Stage1 的 `x/y` 作为来源）
- **Stage2 路由规则**
  - 若 `stage1_bad < 4`：跳过 Stage2，直接处理下一条
  - 若 `stage1_bad >= 4`：进入 Stage2
- **Stage2 推理（模型调用）**
  - 对每个问题调用 Stage2 模型 **8 次**
  - **提问内容**：只提供 **question**（不再拼接 Stage1 的 8 个答案/长评测 prompt）
  - **强制输出格式**：要求模型把最终答案写入 `\\boxed{...}`（用于后续可靠抽取）
  - 记录每次推理的 **完整原文输出**
- **Stage2 答案抽取与判别（规则优先）**
  - **抽取器**：从每次推理结果中提取 `\\boxed{...}` 内的答案字符串（作为候选 final）
  - **规则系统（非大模型）优先判别是否与 gold 一致**
    - **strip/normalize**：字符串标准化（去空格、大小写、符号归一、全角半角等）
    - **精准匹配**：可直接判定的完全一致
    - **简单等价性**：处理常见等价（如 `A` vs `A`
      / `A.-5` vs `-5` 的选项映射、分数/小数互化等）
    - **数学表达式验证**：对表达式做解析与等价判断（使用专业数学库；数值比较可设容差）
  - **规则无法判别的样本**：再交给大模型做“答案一致性判定”（只裁决，不解题）
- **Stage2 归档（按问题聚合）**
  - 对每个问题，把 8 次推理的完整信息与判别结果存为列表（用于后续训练/分析）
  - 归档条目建议包含：`uuid`、`difficulty(x/y)`、`question`、`gold_answer`、`attempts[]`
    - `attempts[]`：每次推理的 `raw_text`、`boxed_answer`、`verdict(正确/错误/不确定->已裁决)`、`judge_reason(可选)`
- **Stage2 产出与 Stage3 判定**
  - 在“规则优先 + 模型裁决”后得到该题的总体判别结果（统计 `stage2_ok/stage2_bad`）
  - 若 `stage2_bad >= 4`：进入 Stage3
  - 否则：将该题的归档结果写入题库目录（问题、推理结果列表、答案、uuid、难度）

### Stage3（复用 Stage2 的判别/归档逻辑；难题仍可放弃）

- **进入条件**：Stage2 后 `stage2_bad >= 4`
- **执行内容**：重复 Stage2 的
  - 抽取器（boxed）
  - 规则系统优先判别 + 不确定样本的模型裁决
  - 8 次推理完整信息 + 标签列表归档
- **结束条件**
  - 若 Stage3 后 `stage3_bad >= 4`：**放弃该题**（不进入题库）
  - 否则：同样归档到题库目录



