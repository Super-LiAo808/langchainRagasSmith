"""环境变量、模型工厂与 LangSmith 追踪。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def langsmith_project_url() -> str | None:
    project = os.getenv("LANGCHAIN_PROJECT")
    if not project:
        return None
    return f"https://smith.langchain.com/o/-/projects/p/{project}"


def print_langsmith_status() -> None:
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    has_key = bool(os.getenv("LANGCHAIN_API_KEY"))
    project = os.getenv("LANGCHAIN_PROJECT", "(未设置)")
    print("========= LangSmith 追踪 ==========")
    print(f"  LANGCHAIN_TRACING_V2 = {tracing}")
    print(f"  LANGCHAIN_API_KEY    = {'已配置' if has_key else '未配置'}")
    print(f"  LANGCHAIN_PROJECT    = {project}")
    url = langsmith_project_url()
    if url and tracing and has_key:
        print(f"  控制台: {url}")
    elif not has_key:
        print("  提示: 在 .env 配置 LANGCHAIN_API_KEY 后才能在 LangSmith 看到运行过程")
    print("====================================")


def setup_langsmith_tracing() -> None:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "test-ragas2")
    for key in (
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_API_KEY",
    ):
        if os.getenv(key):
            os.environ[key] = os.environ[key]


def get_llm_api_key() -> str:
    key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "未读取到 LLM_API_KEY。请在项目根目录 .env 中配置（可复制 .env.example）。"
        )
    return key


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3"),
        temperature=0,
        openai_api_key=get_llm_api_key(),
        openai_api_base=os.getenv("LLM_API_BASE", "https://api.siliconflow.cn/v1"),
    )


def create_embeddings() -> OpenAIEmbeddings:
    api_key = get_llm_api_key()
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        openai_api_key=os.getenv("EMBEDDING_API_KEY", api_key),
        openai_api_base=os.getenv(
            "EMBEDDING_API_BASE",
            os.getenv("LLM_API_BASE", "https://api.siliconflow.cn/v1"),
        ),
        tiktoken_model_name="cl100k_base",
    )
