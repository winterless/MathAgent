from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, Iterator, Optional


JsonDict = Dict[str, Any]


def iter_jsonl(path: str) -> Iterator[JsonDict]:
    """Yield dict objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object (dict) on line {line_no} in {path}")
            yield obj


def write_jsonl_atomic(path: str, rows: Iterable[JsonDict]) -> None:
    """Atomically overwrite a JSONL file."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

    fd: Optional[int] = None
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".jsonl", dir=parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass


def append_jsonl(path: str, rows: Iterable[JsonDict]) -> None:
    """Append dict objects to a JSONL file (creates parent dirs)."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


