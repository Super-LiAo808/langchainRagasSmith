from rag_eval.config.settings import (
    PROJECT_ROOT,
    create_embeddings,
    create_llm,
    get_llm_api_key,
    langsmith_project_url,
    print_langsmith_status,
    setup_langsmith_tracing,
)

__all__ = [
    "PROJECT_ROOT",
    "create_embeddings",
    "create_llm",
    "get_llm_api_key",
    "langsmith_project_url",
    "print_langsmith_status",
    "setup_langsmith_tracing",
]

# 加载 .env 并开启 LangSmith 追踪（须在其它 langchain 模块之前 import 本包）
setup_langsmith_tracing()
