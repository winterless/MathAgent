# Tools

独立的数据处理工具，与核心 pipeline 代码解耦。

## convert_sample_to_input.py

将 `sample.jsonl` 格式的评估结果转换为下一阶段的输入格式。

### 功能

1. 读取 `sample.jsonl` 格式的评估结果文件
2. 解析 `output.content.choices[0].message.content` 中的评估报告
3. 提取 `\boxed{解答正确：x，解答错误：y}` 统计信息
4. 判断是否满足条件：`解答正确 >= min_votes_to_accept`
5. 如果满足，提取 `uuid` 和 `question`，生成新的 JSONL 文件

### 用法

#### 方式1：指定输入文件和输出文件

```bash
python3 tools/convert_sample_to_input.py \
    --input datasets/sample.jsonl \
    --output datasets/input/next_stage_input.jsonl \
    --min-votes 4 \
    --verbose
```

#### 方式2：指定目录，自动查找并在同路径下创建输出文件（推荐）

```bash
# 在 datasets 目录下查找 sample.jsonl，并在同路径下创建 next_stage_input.jsonl
python3 tools/convert_sample_to_input.py \
    --input datasets \
    --input-filename sample.jsonl \
    --min-votes 4 \
    --verbose
```

### 参数

- `--input`: 输入的 `sample.jsonl` 文件路径或目录（必需）
- `--output`: 输出的 JSONL 文件路径（可选，如果不指定则在输入文件同目录下创建）
- `--input-filename`: 当输入是目录时，要查找的文件名（默认：`sample.jsonl`）
- `--output-filename`: 当未指定 `--output` 时，使用的输出文件名（默认：`next_stage_input.jsonl`）
- `--min-votes`: 最小接受票数阈值（默认：4）
- `--verbose`: 输出详细信息

### 输出格式

输出的 JSONL 文件每行包含：
```json
{"uuid": "...", "question": "..."}
```

### 示例

```bash
# 方式1：指定输入和输出文件
python3 tools/convert_sample_to_input.py \
    --input datasets/sample.jsonl \
    --output datasets/input/next_stage.jsonl \
    --min-votes 4

# 方式2：在目录中查找，自动在同路径下创建输出文件（推荐）
python3 tools/convert_sample_to_input.py \
    --input datasets \
    --input-filename sample.jsonl \
    --min-votes 4 \
    --verbose

# 方式3：自定义输出文件名
python3 tools/convert_sample_to_input.py \
    --input datasets \
    --input-filename sample.jsonl \
    --output-filename stage2_input.jsonl \
    --min-votes 4
```

### 使用场景

当你有 `sample.jsonl` 格式的评估结果，且评估通过（解答正确数量 >= 阈值）时，可以使用此工具将其转换为下一阶段的输入，然后继续执行：

```bash
# 1. 转换评估结果为输入格式（推荐：在目录中查找，自动创建输出文件）
python3 tools/convert_sample_to_input.py \
    --input datasets \
    --input-filename sample.jsonl \
    --min-votes 4

# 输出文件会在 datasets/next_stage_input.jsonl（与 sample.jsonl 同路径）

# 2. 使用转换后的文件作为下一阶段的输入
RUN_DIR=datasets/out/demo_next
PYTHONPATH=src python3 src/run_pipeline.py \
    --mode infer \
    --input datasets/next_stage_input.jsonl \
    --out "$RUN_DIR/stage1" \
    --llm-config config/llm_models.json
```

### 批量处理

工具支持在目录中递归查找多个 `sample.jsonl` 文件，并在每个文件的同路径下创建对应的输出文件：

```bash
# 在 datasets 目录下递归查找所有 sample.jsonl 文件
python3 tools/convert_sample_to_input.py \
    --input datasets \
    --input-filename sample.jsonl \
    --min-votes 4 \
    --verbose

# 例如：
# - datasets/sample.jsonl -> datasets/next_stage_input.jsonl
# - datasets/subdir/sample.jsonl -> datasets/subdir/next_stage_input.jsonl
```
