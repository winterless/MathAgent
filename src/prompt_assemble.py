from __future__ import annotations

from typing import List


# This is the prompt stored in JSON, and should match datasets/sample.jsonl style:
# a standard header + (question + standard answer + 8 answers). No output-template here.
STORED_PROMPT_HEADER = (
    "你是一个**判断答案与标准答案一致**的专家。下面提供了一道数学推理题的标准答案和8个不同答案。基于这些内容，完成以下任务。\n\n"
    "###\n"
    "任务：\n"
    "每个解答与标准解答对比，内容一致或相似则为解答正确，否则为解答错误\n\n"
    "要求：\n"
    "1.在对比答案时，对于数值相同但表述不同的答案，认定是一致的。\n"
    "2. 若某个解答的数值与标准答案数值在一定误差内，则认定是一致的。\n"
    "3. 如果某个解答时具体数值或表达式，须重新计算，将其转换为四位小数，例如2π转换为6.2832，3e转换为8.1548，根号6转化为2.4495。\n"
    "4.对比的时候无需关注答案的格式。\n"
    ".禁止解题或对题目进行推理，题目仅在大难表达形式不同（如A vs 34）时，用来确认它们是否指向同一含义。\n"
    "6.若某个解答无内容或者缺少最终答案直接判定该解答错误\n\n"
    "罗列出解答正确和解答错误的解答编号，统计解答1至解答8的解答正确数量和解答错误数量，最终结果输出在boxed中，标注解答正确数量和解答错误数量标识，详例如下：\n"
    "\\boxed{解答正确：5，解答错误：3}\n"
    "###\n\n"
)


def assemble_stored_prompt(*, question: str, standard_answer: str, stage1_answers: List[str]) -> str:
    """
    Assemble the stored prompt (sample.jsonl style):
    - Header (fixed)
    - [题目] block (question)
    - [标准解答] block (standard_answer)
    - 8 answers blocks: [解答1..8]
    """
    lines: List[str] = []
    lines.append(STORED_PROMPT_HEADER.rstrip("\n"))
    lines.append("###")
    lines.append("[题目]")
    lines.append((question or "").strip())
    lines.append("###")
    lines.append("")
    lines.append("###")
    lines.append("[标准解答]")
    lines.append((standard_answer or "").strip())
    lines.append("[标准解答]")
    lines.append("###")
    lines.append("")

    for i in range(8):
        ans = stage1_answers[i] if i < len(stage1_answers) else ""
        lines.append("###")
        lines.append(f"[解答{i+1}]")
        lines.append((ans or "").strip())
        lines.append(f"[解答{i+1}]")
        lines.append("###")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


