"""
CLI entrypoint for MathAgent.

Design note:
- The Route-A implementation + scenario runner live in `src/generator/route_a_impl.py`.
- This file intentionally stays thin (entry only).
"""

from __future__ import annotations

from generator.route_a_impl import main


if __name__ == "__main__":
    main()
