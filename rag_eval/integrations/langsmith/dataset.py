"""LangSmith 测试数据集创建与复用。"""

from __future__ import annotations

from langsmith import Client
from langsmith.utils import LangSmithError

from rag_eval.config.cases import EvalCase


def ensure_eval_dataset(
    client: Client,
    cases: list[EvalCase],
    *,
    dataset_name: str,
    description: str,
):
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print("using existing dataset:", dataset.name)
        return dataset
    except LangSmithError:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )
        for case in cases:
            client.create_example(
                inputs={"query": case.query},
                outputs={"ground_truths": case.ground_truths},
                dataset_id=dataset.id,
            )
        print("Created a new dataset:", dataset.name)
        return dataset
