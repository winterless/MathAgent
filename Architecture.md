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
- **Stage2/Stage3（最简）**：仅当上一阶段 output 里解析不到 `\\boxed{解答正确：x，解答错误：y}` 才会重评并写入下一阶段 JSONL



