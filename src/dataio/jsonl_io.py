from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Iterable, Iterator, Optional

JsonDict = Dict[str, Any]


def iter_jsonl(path: str, *, tolerate_errors: bool = True) -> Iterator[JsonDict]:
    """Yield dict objects from a JSONL file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if tolerate_errors:
                    print(f"[WARN] Invalid JSON on line {line_no} in {path}: {e}", file=sys.stderr, flush=True)
                    continue
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                if tolerate_errors:
                    print(
                        f"[WARN] Expected JSON object (dict) on line {line_no} in {path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
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


def append_jsonl_line(path: str, row: JsonDict) -> None:
    """
    Append a single JSON object line to a JSONL file and fsync.
    This is used for per-uuid checkpointing to support resume.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())



