"""RAGAS 指标 → LangSmith RunEvaluator（LangChain 1.x 兼容）。"""

from __future__ import annotations

import typing as t

from langchain_core.documents import Document
from langsmith.evaluation import EvaluationResult
from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.run_config import RunConfig
from ragas.utils import convert_row_v1_to_v2, get_required_columns_v1

from rag_eval.config import create_embeddings, create_llm


def _first_ground_truth(outputs: dict) -> str:
    if "ground_truth" in outputs:
        val = outputs["ground_truth"]
        return val if isinstance(val, str) else str(val)
    if "ground_truths" in outputs:
        gt = outputs["ground_truths"]
        if isinstance(gt, list) and gt:
            return gt[0] if isinstance(gt[0], str) else str(gt[0])
        return str(gt)
    raise ValueError("数据集 outputs 需包含 ground_truth 或 ground_truths")


def _build_chain_eval_row(run: Run, example: Example) -> dict[str, t.Any]:
    if example.inputs is None or example.outputs is None:
        raise ValueError("LangSmith 样例需同时包含 inputs 与 outputs")
    if run.outputs is None:
        raise ValueError("链运行结果为空，请确认 return_source_documents=True")

    row: dict[str, t.Any] = dict(run.outputs)
    if "query" in example.inputs:
        row["question"] = example.inputs["query"]
    elif "question" in example.inputs:
        row["question"] = example.inputs["question"]
    else:
        raise ValueError("样例 inputs 需包含 query 或 question")

    if "result" in row and "answer" not in row:
        row["answer"] = row["result"]

    if "source_documents" in row and "contexts" not in row:
        docs = row["source_documents"]
        if docs and isinstance(docs[0], Document):
            row["contexts"] = [d.page_content for d in docs]
        elif isinstance(docs, list):
            row["contexts"] = [str(d) for d in docs]

    return row


def _attach_ground_truth_if_needed(
    row: dict, example: Example, metric: Metric
) -> dict:
    required = get_required_columns_v1(metric)
    if "ground_truth" not in row and "ground_truth" in required:
        row = dict(row)
        row["ground_truth"] = _first_ground_truth(example.outputs)
    return row


class RagasMetricRunEvaluator(RunEvaluator):
    """将单个 RAGAS 指标包装为 LangSmith RunEvaluator。"""

    def __init__(
        self,
        metric: Metric,
        *,
        llm=None,
        embeddings=None,
        run_config: RunConfig | None = None,
    ):
        self.metric = metric
        self._run_config = run_config or RunConfig()
        if isinstance(metric, MetricWithLLM):
            metric.llm = LangchainLLMWrapper(llm or create_llm())
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = LangchainEmbeddingsWrapper(
                embeddings or create_embeddings()
            )
        metric.init(self._run_config)
        if not isinstance(metric, SingleTurnMetric):
            raise TypeError(f"{metric.name} 须为 SingleTurnMetric")

    def evaluate_run(
        self,
        run: Run,
        example: Example | None = None,
        evaluator_run_id=None,
    ) -> EvaluationResult:
        if example is None:
            raise ValueError("RAGAS 评估需要 LangSmith 样例（含 ground_truths）")
        raw = _build_chain_eval_row(run, example)
        raw = _attach_ground_truth_if_needed(raw, example, self.metric)
        row = convert_row_v1_to_v2(raw)
        sample = SingleTurnSample(**row)
        score = self.metric.single_turn_score(sample, callbacks=[])
        return EvaluationResult(key=self.metric.name, score=float(score))


def build_ragas_evaluators(
    metrics: list[Metric] | None = None,
    *,
    llm=None,
    embeddings=None,
) -> list[RagasMetricRunEvaluator]:
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = metrics or [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    llm = llm or create_llm()
    embeddings = embeddings or create_embeddings()
    return [
        RagasMetricRunEvaluator(m, llm=llm, embeddings=embeddings) for m in metrics
    ]
