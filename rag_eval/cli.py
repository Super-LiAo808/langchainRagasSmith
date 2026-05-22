"""命令行入口：LangChain × RAGAS × LangSmith 联合评估。"""

from __future__ import annotations

import argparse
import sys

# 确保加载 .env 与 LangSmith 追踪
import rag_eval.config  # noqa: F401

from rag_eval.evaluation.batch import run_batch_evaluation
from rag_eval.evaluation.single import run_single_evaluation
from rag_eval.integrations.langsmith.runs import print_recent_runs_report
from rag_eval.integrations.langsmith.smoke import run_trace_smoke_test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="LangChain × RAGAS × LangSmith 联合评估 RAG 应用",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("single", help="单条 RAG 问答 + RAGAS 评估")
    sub.add_parser("batch", help="LangSmith 数据集批量评估")
    sub.add_parser("smoke", help="LangSmith 追踪冒烟测试（仅 LLM）")
    check = sub.add_parser("check", help="检查 LangSmith 最近是否有 run 上报")
    check.add_argument("--hours", type=float, default=1.0, help="查询最近 N 小时")

    args = parser.parse_args(argv)

    if args.command == "single":
        run_single_evaluation()
        return 0
    if args.command == "batch":
        run_batch_evaluation()
        return 0
    if args.command == "smoke":
        run_trace_smoke_test()
        return 0
    if args.command == "check":
        return print_recent_runs_report(hours=args.hours)

    return 1


if __name__ == "__main__":
    sys.exit(main())
