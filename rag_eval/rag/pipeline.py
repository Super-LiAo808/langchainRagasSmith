"""RAG 流水线：网页加载 → 向量索引 → RetrievalQA。"""

from __future__ import annotations

from langchain_classic.chains import RetrievalQA
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import WebBaseLoader

from rag_eval.config import create_embeddings, create_llm
from rag_eval.config.cases import DEFAULT_DOC_URL, DEFAULT_VERIFY_SSL


def build_index(
    url: str = DEFAULT_DOC_URL,
    *,
    verify_ssl: bool = DEFAULT_VERIFY_SSL,
):
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {"verify": verify_ssl}
    embedding = create_embeddings()
    return VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])


def build_qa_chain(
    index=None,
    *,
    url: str = DEFAULT_DOC_URL,
    verify_ssl: bool = DEFAULT_VERIFY_SSL,
):
    if index is None:
        index = build_index(url=url, verify_ssl=verify_ssl)
    llm = create_llm()
    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        return_source_documents=True,
    )
