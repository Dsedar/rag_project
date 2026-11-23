"""
Централизованная конфигурация проекта Witcher 3 RAG.
Загружает переменные окружения из .env файла и предоставляет типизированные настройки.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Путь к корню репозитория
BASE_DIR = Path(__file__).resolve().parent

# Загружаем переменные из .env файла, если он существует
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()


class Settings:
    """Настройки подключения, моделей и базы знаний."""

    # --- Milvus Vector Database ---
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "root:Milvus")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "witcher3_rag")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "1024"))

    @property
    def milvus_uri(self) -> str:
        """Возвращает URI для подключения через MilvusClient."""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    # --- Embeddings & Reranker Models ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")

    # --- LLM Server (OpenAI-compatible: vLLM, Ollama, LM Studio, TGI, OpenAI) ---
    OPENAI_API_BASE: str = os.getenv(
        "OPENAI_API_BASE", "http://localhost:8000/v1"
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy-key")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # --- Retrieval & RAG Parameters ---
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    FETCH_K: int = int(os.getenv("FETCH_K", "20"))
    HNSW_EF: int = int(os.getenv("HNSW_EF", "200"))
    CHUNK_MAX_CHARS: int = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --- Data & Wiki Scraper ---
    DATA_FILE: Path = BASE_DIR / os.getenv(
        "DATA_FILE", "witcher3_knowledge_base.json"
    )
    WIKI_DOMAIN: str = os.getenv("WIKI_DOMAIN", "vedmak.fandom.com")
    ROOT_CATEGORY: str = os.getenv("ROOT_CATEGORY", "Категория:Ведьмак 3")
    SCRAPER_DELAY: float = float(os.getenv("SCRAPER_DELAY", "0.5"))


# Синглтон настроек для удобного импорта
settings = Settings()
