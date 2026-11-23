"""
Инструмент (Tool) для интеграции векторного поиска с агентным фреймворком Langroid.
Позволяет AI-агенту запрашивать релевантную информацию из базы знаний Ведьмака 3.
"""

import sys
from pathlib import Path
from typing import List

# Добавляем корень проекта в sys.path для импорта rag_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_config import settings

try:
    from langroid.agent.tool_message import ToolMessage
except ImportError:
    # Заглушка на случай, если langroid не установлен
    class ToolMessage:
        pass

# Ленивая инициализация клиентов
_milvus_client = None
_embedding_model = None


def get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        try:
            from pymilvus import MilvusClient
            _milvus_client = MilvusClient(
                uri=settings.milvus_uri, token=settings.MILVUS_TOKEN
            )
        except ImportError as e:
            raise ImportError(
                f"Не установлен пакет pymilvus ({e}). Установите: pip install pymilvus"
            )
    return _milvus_client


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except ImportError as e:
            raise ImportError(
                f"Не установлен пакет sentence-transformers ({e}). Установите: pip install sentence-transformers"
            )
    return _embedding_model


class RagSearchTool(ToolMessage):
    """
    Инструмент для поиска по базе знаний о вселенной Ведьмака 3 (Fandom Wiki).
    Используется агентом, когда пользователь задает вопросы о персонажах,
    квестах, монстрах, бестиарии, локациях или сюжете.
    """

    request: str = "rag_search"
    purpose: str = """
    Найти достоверную информацию в базе знаний о Ведьмаке 3,
    когда требуется вспомнить детали сюжета, характеристики персонажей,
    слабости чудовищ, рецепты зелий или описания локаций.
    """
    query: str = "Вопрос для поиска в базе знаний"

    @classmethod
    def response(cls, query: str) -> str:
        client = get_milvus_client()
        model = get_embedding_model()

        # Генерация нормализованного эмбеддинга
        query_emb = model.encode(query, normalize_embeddings=True).tolist()

        # Поиск по векторному полю
        results = client.search(
            collection_name=settings.COLLECTION_NAME,
            data=[query_emb],
            anns_field="vector",
            limit=3,
            output_fields=["title", "text", "url", "subcategory"],
            search_params={"metric_type": "COSINE", "params": {"ef": settings.HNSW_EF}},
        )

        if not results or not results[0]:
            return "В базе знаний Ведьмака ничего не найдено по данному запросу."

        # Формируем ответ для агента
        memories = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            title = entity.get("title", "Неизвестный источник")
            text = entity.get("text", "")
            url = entity.get("url", "")
            memories.append(f"📌 [Источник: '{title}' ({url})]\n{text}")

        return "\n\n---\n\n".join(memories)

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(query="Что такое Дикая Охота?"),
            cls(query="Кто такой Геральт из Ривии?"),
            cls(query="Уязвимости полуденницы и как её победить?"),
            cls(query="Где найти чертежи школы Кота?"),
        ]

    def handle(self) -> str:
        return self.response(self.query)
