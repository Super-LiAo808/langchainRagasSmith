"""默认评估用例与文档源（教程：百度百科纽约）。"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_DOC_URL = "https://baike.baidu.com/item/%E7%BA%BD%E7%BA%A6/6230"
DEFAULT_VERIFY_SSL = False

DATASET_NAME = "ragas test data"
DATASET_DESCRIPTION = "NYC test dataset — LangChain x RAGAS x LangSmith"


@dataclass
class EvalCase:
    query: str
    ground_truths: list[str] = field(default_factory=list)


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        query="纽约市的名字是怎么得来的?",
        ground_truths=[
            "纽约市的名字“纽约”来源于荷兰战败后将新阿姆斯特丹割让给英国的事件。",
        ],
    ),
]

# 单条快速评估默认用第一条
DEFAULT_SINGLE_CASE = DEFAULT_CASES[0]
