"""Headless command-line entry point.

Usage::

    uv run mar "What is the James Webb Space Telescope?" --web
    uv run python -m multi_agent_research.cli "topic" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import get_settings, public_config_summary
from .factory import build_system
from .logging_config import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mar",
        description="Run a multi-agent research query from the terminal.",
    )
    parser.add_argument("query", help="The research question.")
    parser.add_argument("--web", action="store_true", help="Enable the web-search agent.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON result.")
    parser.add_argument(
        "--show-config", action="store_true", help="Print redacted config and exit."
    )
    return parser


async def _run(query: str, *, enable_web: bool) -> int:
    system = build_system(enable_web=True)
    try:
        result = await system.pipeline.run(query, enable_web=enable_web)
    finally:
        await system.aclose()

    if result.blocked:
        print("BLOCKED:", result.answer_markdown, file=sys.stderr)
        return 2

    print(result.answer_markdown)
    print()
    print("Sources:")
    for citation in result.state.citations.citations:
        print(f"  [{citation.number}] {citation.title} - {citation.url}")
    print()
    print("Latency (ms):", json.dumps(result.metrics.stage_timings()))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    if args.show_config:
        print(json.dumps(public_config_summary(settings), indent=2))
        return 0

    exit_code = asyncio.run(_run(args.query, enable_web=args.web))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
