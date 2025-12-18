# MathAgent：高难度数学题筛选与答案生成流程

![Architecture](diagrams/architecture.png)

```mermaid
flowchart TD
  A[输入：题目（无答案）] --> B[Stage 1：4B 快思考 ×8]
  B --> C{Major voting 一致数 > n ?<br/>例如 n=7}
  C -- 是 --> X[判定：简单题\n结束（不入库）]
  C -- 否 --> D[Stage 2：30B 快思考 ×8]
  D --> E{Major voting 一致数 > n ?<br/>例如 n=7}
  E -- 是 --> S[存储：题目 + 多数答案]
  E -- 否 --> F[Stage 3：30B Thinking ×8]
  F --> G{Major voting 一致数 > n ?<br/>例如 n=7}
  G -- 是 --> S
  G -- 否 --> Y[丢弃：无法稳定产出答案]
```

## 判定与产出

- **一致性判定**：每个 Stage 生成 8 个答案，用 **major voting** 统计“最多数答案”的出现次数；若 **> n** 则认为答案稳定。
- **输出数据**：仅当 Stage 2 或 Stage 3 达到稳定一致性时，**存储（题目 + 多数答案）**；否则按流程 **结束或丢弃**。


