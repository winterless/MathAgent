"""
CLI entrypoint for MathAgent.

Design note:
- Pipeline v2 implementation lives in `src/generator/pipeline_v2.py`.
- This file intentionally stays thin (entry only).
"""

from __future__ import annotations

from generator.pipeline_v2 import main


if __name__ == "__main__":
    main()
