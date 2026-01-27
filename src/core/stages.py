from __future__ import annotations


def normalize_for_model(text: str) -> str:
    """
    Normalize common dataset escaping for *model input only*:
    - Convert literal '\\n' to real newlines
    - Unescape one level of backslashes
    - Unescape '\\$' -> '$'
    """
    s = text or ""
    if "\\n" in s:
        s = s.replace("\\n", "\n")
    if "\\\\" in s:
        s = s.replace("\\\\", "\\")
    if "\\$" in s:
        s = s.replace("\\$", "$")
    return s
