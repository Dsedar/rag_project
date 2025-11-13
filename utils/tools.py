# tools.py
from langroid.agent.tool_message import ToolMessage
from langroid.utils.configuration import settings
import json
from typing import List
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# --- Настройки ---
MILVUS_URI = "порт"
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "witcher3_rag"

# --- Глобальные объекты ---
milvus_client = MilvusClient(uri=MILVUS_URI, token="root:Milvus")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

class RagSearchTool(ToolMessage):
    """
    Инструмент для поиска по базе знаний о Ведьмаке.
    """
    request: str = "rag_search"
    purpose: str = """
    Найти информацию в базе знаний о вселенной Ведьмака, 
    когда пользователь задаёт вопрос о локациях, персонажах, сюжете и т.д.
    """
    query: str = "Вопрос для поиска"

    @classmethod
    def response(cls, query: str) -> str:
        # Генерируем эмбеддинг
        query_emb = embedding_model.encode(query, normalize_embeddings=True).tolist()

        # Поиск
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_emb],
            anns_field="embedding",
            limit=3,
            output_fields=["title", "text", "url"],
            metric_type="COSINE"
        )

        # Формируем ответ
        if not results[0]:
            return "В базе знаний ничего не найдено."

        # Берём первый чанк как "воспоминание"
        hit = results[0][0]
        content = hit["entity"]["text"]
        title = hit["entity"]["title"]

        # Возвращаем как "всплыло в памяти"
        return f"""
        [Воспоминание легенд: '{title}']
        {content}
        """.strip()

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(query="Что такое Дикая Охота?"),
            cls(query="Кто такой Геральт из Ривии?"),
            cls(query="Что входит в состав Крови эльфов?"),
        ]

    def handle(self) -> str:
        return self.response(self.query)