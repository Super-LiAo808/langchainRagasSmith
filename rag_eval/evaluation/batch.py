"""LangSmith 数据集 + run_on_dataset + RAGAS 批量评估。"""

from __future__ import annotations

import os

from langchain_classic.smith import RunEvalConfig, run_on_dataset
from langsmith import Client

from rag_eval.config import print_langsmith_status
from rag_eval.config.cases import (
    DATASET_DESCRIPTION,
    DATASET_NAME,
    DEFAULT_CASES,
    EvalCase,
)
from rag_eval.evaluation.ragas_metrics import build_ragas_evaluators
from rag_eval.integrations.langsmith.dataset import ensure_eval_dataset
from rag_eval.rag import build_qa_chain


def run_batch_evaluation(
    cases: list[EvalCase] | None = None,
    *,
    dataset_name: str = DATASET_NAME,
    dataset_description: str = DATASET_DESCRIPTION,
):
    print_langsmith_status()
    if not os.getenv("LANGCHAIN_API_KEY"):
        print(
            "未配置 LANGCHAIN_API_KEY，无法执行批量评估。\n"
            "请在 .env 中配置后重试，或使用: python -m scripts.run_single"
        )
        return None

    cases = cases or DEFAULT_CASES
    client = Client()
    ensure_eval_dataset(
        client,
        cases,
        dataset_name=dataset_name,
        description=dataset_description,
    )

    print("构建 RAG 索引与 QA 链...")
    qa_chain = build_qa_chain()

    evaluation_config = RunEvalConfig(
        custom_evaluators=build_ragas_evaluators(),
        prediction_key="result",
    )

    print("在 LangSmith 数据集上运行 QA + RAGAS 评估...")
    result = run_on_dataset(
        client,
        dataset_name,
        qa_chain,
        evaluation=evaluation_config,
        input_mapper=lambda x: x,
        verbose=True,
    )
    print("评估完成。")
    return result
