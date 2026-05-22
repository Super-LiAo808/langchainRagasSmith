"""LangSmith 追踪冒烟测试。"""

from __future__ import annotations

import os

from rag_eval.config import create_llm, print_langsmith_status


def run_trace_smoke_test(prompt: str = "用一句话介绍纽约市名字的由来。") -> str:
    print_langsmith_status()
    print("LANGCHAIN_TRACING_V2 =", os.environ.get("LANGCHAIN_TRACING_V2"))
    print("LANGCHAIN_PROJECT =", os.environ.get("LANGCHAIN_PROJECT"))
    print("调用 LLM（应产生 LangSmith trace）...")
    llm = create_llm()
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", str(resp))
    print("回复:", content[:200], "...")
    return content
