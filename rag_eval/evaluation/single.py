"""单条 RAG 问答 + RAGAS evaluate()。"""

from __future__ import annotations

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_eval.config import create_embeddings, create_llm, print_langsmith_status
from rag_eval.config.cases import DEFAULT_SINGLE_CASE, EvalCase
from rag_eval.rag import build_qa_chain


def run_single_evaluation(case: EvalCase | None = None, *, doc_url: str | None = None):
    case = case or DEFAULT_SINGLE_CASE
    print_langsmith_status()

    print("构建 RAG 索引...")
    chain = build_qa_chain(url=doc_url) if doc_url else build_qa_chain()

    print("提问:", case.query)
    out = chain.invoke({"query": case.query})
    answer = out.get("result", "")
    docs = out.get("source_documents", [])
    contexts = [d.page_content for d in docs] if docs else []

    print("回答:", answer[:500] + ("..." if len(answer) > 500 else ""))
    print("检索片段数:", len(contexts))

    reference = case.ground_truths[0] if case.ground_truths else ""
    llm = create_llm()
    embeddings = create_embeddings()
    eval_dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": case.query,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        ]
    )

    print("RAGAS 评估中...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    print(result)
    print_langsmith_status()
    return result
