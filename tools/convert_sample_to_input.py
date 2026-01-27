#!/usr/bin/env python3
"""
将 sample.jsonl 格式的评估结果转换为下一阶段的输入格式。

用法:
    python3 tools/convert_sample_to_input.py \
        --input datasets/sample.jsonl \
        --output datasets/input/next_stage_input.jsonl \
        --min-votes 4

功能:
    1. 读取 sample.jsonl，解析评估结果
    2. 提取 \boxed{解答正确：x，解答错误：y} 中的统计信息
    3. 判断 解答正确 >= min_votes_to_accept
    4. 如果满足，提取 uuid 和 question，生成新的 JSONL 文件
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_evaluation_stats(content: str) -> Tuple[int, int] | None:
    """
    从评估内容中提取统计信息。
    
    Args:
        content: 评估报告文本内容
        
    Returns:
        (解答正确数量, 解答错误数量) 或 None（如果解析失败）
    """
    # 匹配 \boxed{解答正确：x，解答错误：y} 格式
    # 支持多种可能的格式变体
    patterns = [
        r'\\boxed\{解答正确[：:]\s*(\d+)[，,]\s*解答错误[：:]\s*(\d+)\}',
        r'boxed\{解答正确[：:]\s*(\d+)[，,]\s*解答错误[：:]\s*(\d+)\}',
        r'解答正确[：:]\s*(\d+)[，,]\s*解答错误[：:]\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            try:
                ok_count = int(match.group(1))
                bad_count = int(match.group(2))
                return (ok_count, bad_count)
            except (ValueError, IndexError):
                continue
    
    return None


def parse_sample_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    解析 sample.jsonl 的一行数据，提取评估统计信息。
    
    Args:
        row: sample.jsonl 的一行 JSON 对象
        
    Returns:
        包含统计信息的字典，或 None（如果解析失败）
        {
            "uuid": "...",
            "question": "...",
            "ok_count": 5,
            "bad_count": 3,
            "content": "...",
        }
    """
    uuid = row.get("uuid")
    question = row.get("question")
    
    if not uuid or not question:
        return None
    
    # 提取评估内容
    output = row.get("output", {})
    if not isinstance(output, dict):
        return None
    
    content_obj = output.get("content", {})
    if not isinstance(content_obj, dict):
        return None
    
    choices = content_obj.get("choices", [])
    if not choices or not isinstance(choices, list):
        return None
    
    first_choice = choices[0] if choices else {}
    if not isinstance(first_choice, dict):
        return None
    
    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        return None
    
    content = message.get("content", "")
    if not isinstance(content, str):
        return None
    
    # 提取统计信息
    stats = extract_evaluation_stats(content)
    if stats is None:
        return None
    
    ok_count, bad_count = stats
    
    return {
        "uuid": str(uuid),
        "question": str(question),
        "ok_count": ok_count,
        "bad_count": bad_count,
        "content": content,
    }


def find_sample_files(input_dir: str, filename: str) -> List[Path]:
    """
    在目录中查找指定文件名的文件。
    
    Args:
        input_dir: 输入目录路径
        filename: 要查找的文件名（如 "sample.jsonl"）
        
    Returns:
        找到的文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if input_path.is_file():
        # 如果是文件，直接返回
        return [input_path]
    
    # 在目录中递归查找
    found_files = list(input_path.rglob(filename))
    return sorted(found_files)


def convert_sample_to_input(
    input_path: str,
    output_path: str | None,
    min_votes_to_accept: int,
    verbose: bool = False,
    input_filename: str = "sample.jsonl",
    output_filename: str = "next_stage_input.jsonl",
) -> None:
    """
    将 sample.jsonl 转换为下一阶段的输入格式。
    
    Args:
        input_path: 输入的 sample.jsonl 文件路径或目录
        output_path: 输出的 JSONL 文件路径（如果为 None，则在输入文件同目录下创建）
        min_votes_to_accept: 最小接受票数阈值
        verbose: 是否输出详细信息
        input_filename: 要查找的输入文件名（当 input_path 是目录时）
        output_filename: 输出文件名（当 output_path 为 None 时使用）
    """
    # 查找输入文件
    input_files = find_sample_files(input_path, input_filename)
    
    if not input_files:
        raise FileNotFoundError(
            f"No files matching '{input_filename}' found in: {input_path}"
        )
    
    # 处理每个找到的文件
    for input_file in input_files:
        # 确定输出路径
        if output_path is None:
            # 在输入文件同目录下创建输出文件
            output_file = input_file.parent / output_filename
        else:
            # 如果指定了输出路径，且只有一个输入文件，使用指定的输出路径
            if len(input_files) == 1:
                output_file = Path(output_path)
            else:
                # 多个文件时，在各自目录下创建
                output_file = input_file.parent / output_filename
        
        _convert_single_file(
            input_file=input_file,
            output_file=output_file,
            min_votes_to_accept=min_votes_to_accept,
            verbose=verbose,
        )


def _convert_single_file(
    input_file: Path,
    output_file: Path,
    min_votes_to_accept: int,
    verbose: bool = False,
) -> None:
    """
    转换单个文件。
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_votes_to_accept: 最小接受票数阈值
        verbose: 是否输出详细信息
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    if verbose:
        print(f"Processing: {input_file} -> {output_file}", file=sys.stderr)
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"[WARN] Line {line_no}: Invalid JSON: {e}", file=sys.stderr)
                error_count += 1
                continue
            
            if not isinstance(row, dict):
                if verbose:
                    print(f"[WARN] Line {line_no}: Expected JSON object", file=sys.stderr)
                error_count += 1
                continue
            
            # 解析评估结果
            parsed = parse_sample_row(row)
            if parsed is None:
                if verbose:
                    print(f"[WARN] Line {line_no}: Failed to parse evaluation result (uuid={row.get('uuid', 'unknown')})", file=sys.stderr)
                skipped_count += 1
                continue
            
            # 判断是否满足条件
            if parsed["ok_count"] < min_votes_to_accept:
                if verbose:
                    print(f"[INFO] Line {line_no}: Skipped (ok={parsed['ok_count']} < {min_votes_to_accept}, uuid={parsed['uuid']})", file=sys.stderr)
                skipped_count += 1
                continue
            
            # 生成输出记录（只保留 uuid 和 question）
            output_row = {
                "uuid": parsed["uuid"],
                "question": parsed["question"],
            }
            
            f_out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            converted_count += 1
            
            if verbose:
                print(f"[INFO] Line {line_no}: Converted (ok={parsed['ok_count']}, bad={parsed['bad_count']}, uuid={parsed['uuid']})", file=sys.stderr)
    
    # 输出统计信息
    print(f"Conversion completed for {input_file.name}:", file=sys.stderr)
    print(f"  Total rows: {total_count}", file=sys.stderr)
    print(f"  Converted: {converted_count}", file=sys.stderr)
    print(f"  Skipped: {skipped_count}", file=sys.stderr)
    print(f"  Errors: {error_count}", file=sys.stderr)
    print(f"  Output: {output_file}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sample.jsonl evaluation results to next-stage input format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input sample.jsonl file path or directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (if not specified, created in same directory as input)",
    )
    parser.add_argument(
        "--input-filename",
        default="sample.jsonl",
        help="Input filename to search for when input is a directory (default: sample.jsonl)",
    )
    parser.add_argument(
        "--output-filename",
        default="next_stage_input.jsonl",
        help="Output filename when output path is not specified (default: next_stage_input.jsonl)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=4,
        help="Minimum votes to accept (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    
    args = parser.parse_args()
    
    try:
        convert_sample_to_input(
            input_path=args.input,
            output_path=args.output,
            min_votes_to_accept=args.min_votes,
            verbose=args.verbose,
            input_filename=args.input_filename,
            output_filename=args.output_filename,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
