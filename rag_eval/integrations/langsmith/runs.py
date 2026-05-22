"""查询 LangSmith 最近上报的 run。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from langsmith import Client

from rag_eval.config import langsmith_project_url


def list_recent_root_runs(
    *,
    hours: float = 1.0,
    limit: int = 10,
    project: str | None = None,
) -> list:
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 LANGCHAIN_API_KEY")

    project = project or os.getenv("LANGCHAIN_PROJECT", "test-ragas2")
    client = Client()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return list(
        client.list_runs(
            project_name=project,
            start_time=since,
            limit=limit,
            is_root=True,
        )
    )


def print_recent_runs_report(*, hours: float = 1.0) -> int:
    project = os.getenv("LANGCHAIN_PROJECT", "test-ragas2")
    try:
        runs = list_recent_root_runs(hours=hours)
    except Exception as e:
        print(f"查询 LangSmith 失败: {type(e).__name__}: {e}")
        return 1

    print(f"项目: {project}")
    print(f"近 {hours} 小时内根 run 数量: {len(runs)}")
    for r in runs[:5]:
        print(f"  - id={r.id} name={r.name!r} status={r.status} start={r.start_time}")
    url = langsmith_project_url()
    if runs and url:
        print(f"\n在 LangSmith 打开: {url}")
    elif not runs:
        print("未发现上报的 run，请确认 LANGCHAIN_TRACING_V2=true 且 API Key 有效。")
    return 0 if runs else 2
