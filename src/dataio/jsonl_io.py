from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Iterable, Iterator, Optional

JsonDict = Dict[str, Any]


def iter_jsonl(path: str, *, tolerate_errors: bool = True) -> Iterator[JsonDict]:
    """Yield dict objects from a JSONL file (one JSON object per line)."""
    # Read as bytes and decode per-line to survive encoding issues like:
    #   'utf-8' codec can't decode byte 0xe9 in position ...: invalid continuation byte
    # JSON syntax is ASCII; bad bytes should mostly appear inside string fields, so best-effort
    # decoding (replace) is usually enough to keep the pipeline moving.
    with open(path, "rb") as f:
        for line_no, bline in enumerate(f, start=1):
            bline = bline.strip()
            if not bline:
                continue
            try:
                line = bline.decode("utf-8")
            except UnicodeDecodeError as e:
                if not tolerate_errors:
                    raise
                # Best-effort: keep going with replacement characters.
                line = bline.decode("utf-8", errors="replace")
                print(
                    f"[WARN] Non-UTF8 bytes on line {line_no} in {path}: {e}. Decoded with replacement.",
                    file=sys.stderr,
                    flush=True,
                )

            # Parse JSON; if it fails and we had replacement chars, try a legacy single-byte decode.
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if tolerate_errors and "\ufffd" in line:
                    try:
                        obj = json.loads(bline.decode("latin-1"))
                    except Exception:
                        print(f"[WARN] Invalid JSON on line {line_no} in {path}: {e}", file=sys.stderr, flush=True)
                        continue
                else:
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



