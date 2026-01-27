from __future__ import annotations

from typing import Any, Dict, List

JsonDict = Dict[str, Any]

# Default canonical keys (fallback if not in config)
# Only includes fields that are actually used in the pipeline
_DEFAULT_CANONICAL_KEYS = [
    "uuid",           # Required: used for deduplication
    "question",       # Used: extracted via question_key_priority
    "prompt",         # Used: extracted via question_key_priority
    "text",           # Used: extracted via question_key_priority
    "answer",         # Optional: used as gold answer
    "line_number",    # Optional: used for tracking
    "raw_source_path", # Optional: used for tracking source
]


def get_canonical_keys(canonical_keys: List[str] | None = None) -> List[str]:
    """
    Get canonical keys from config or use default.
    
    Args:
        canonical_keys: List of keys from config, or None to use default
    
    Returns:
        List of canonical keys
    """
    if canonical_keys and isinstance(canonical_keys, list) and len(canonical_keys) > 0:
        return [str(k) for k in canonical_keys if isinstance(k, str) and k]
    return _DEFAULT_CANONICAL_KEYS.copy()


def validate_question_key_priority(question_key_priority: List[str], canonical_keys: List[str]) -> None:
    """
    Validate that all keys in question_key_priority exist in canonical_keys.
    
    Args:
        question_key_priority: List of question field names to check
        canonical_keys: List of canonical keys from config
    
    Raises:
        ValueError: If any key in question_key_priority is not in canonical_keys
    """
    if not isinstance(question_key_priority, list):
        return
    
    canonical_set = set(canonical_keys)
    missing = [k for k in question_key_priority if k not in canonical_set]
    
    if missing:
        raise ValueError(
            f"question_key_priority contains keys not in canonical_keys: {missing}. "
            f"Available keys: {canonical_keys}"
        )


def normalize_record(row: JsonDict, *, canonical_keys: List[str] | None = None) -> JsonDict:
    """
    Normalize input record: extract only useful fields and remove extras.
    Only preserves fields that are actually used in the pipeline.
    
    Args:
        row: Input record to normalize
        canonical_keys: List of canonical keys from config. If None, uses default.
    
    Returns:
        Normalized record with only canonical keys (extra fields removed)
    """
    keys = get_canonical_keys(canonical_keys)
    out: JsonDict = {}
    for k in keys:
        if k == "line_number":
            out[k] = row.get(k, None)
        else:
            out[k] = row.get(k, "")
    return out



