"""
Утилита для быстрой проверки и тестирования векторного поиска в Milvus.
Выполняет семантический поиск по запросу и выводит найденные чанки с оценкой схожести.
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для импорта rag_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_config import settings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Диагностический поиск по векторной базе Milvus (Witcher 3 RAG)."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Кто такой Геральт из Ривии?",
        help="Текстовый запрос для поиска",
    )
    parser.add_argument(
        "--limit",
        "-k",
        type=int,
        default=3,
        help="Количество возвращаемых результатов (по умолчанию: 3)",
    )
    parser.add_argument(
        "--subcategory",
        "-s",
        type=str,
        default=None,
        help="Фильтр по подкатегории (например: 'Персонажи' или 'Квесты')",
    )
    return parser.parse_args()


def search_knowledge_base(
    query: str, limit: int = 3, subcategory: str = None
):
    try:
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(
            f"❌ Не установлены необходимые пакеты ({e}).\n"
            f"Установите зависимости: pip install -r requirements.txt"
        )
        sys.exit(1)

    print(f"🔍 Запрос: '{query}'")
    print(f"🔌 Подключение к Milvus по адресу: {settings.milvus_uri}")

    try:
        client = MilvusClient(uri=settings.milvus_uri, token=settings.MILVUS_TOKEN)
        model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Генерация нормализованного эмбеддинга запроса
        query_emb = model.encode(query, normalize_embeddings=True).tolist()

        filter_expr = f'subcategory == "{subcategory}"' if subcategory else ""

        results = client.search(
            collection_name=settings.COLLECTION_NAME,
            data=[query_emb],
            anns_field="vector",  # Соответствует полю в db_make.py
            limit=limit,
            output_fields=["title", "text", "url", "subcategory"],
            filter=filter_expr if filter_expr else None,
            search_params={"metric_type": "COSINE", "params": {"ef": settings.HNSW_EF}},
        )

        if not results or not results[0]:
            print("❌ По вашему запросу ничего не найдено.")
            return

        print(f"\nНайдено результатов: {len(results[0])}")
        print("=" * 70)

        for i, hit in enumerate(results[0], start=1):
            entity = hit.get("entity", {})
            title = entity.get("title", "Без названия")
            url = entity.get("url", "Нет ссылки")
            subcat = entity.get("subcategory", "Не указана")
            text = entity.get("text", "")
            distance = hit.get("distance", 0.0)

            print(f"[{i}] 🎯 Схожесть (Cosine): {distance:.4f}")
            print(f"    📘 Статья:     {title}")
            print(f"    🏷️ Категория:  {subcat}")
            print(f"    🔗 Ссылка:     {url}")
            print(f"    📝 Фрагмент:   {text[:350].strip()}...")
            print("-" * 70)

    except Exception as e:
        print(f"❌ Ошибка подключения или поиска в Milvus: {e}")
        print("💡 Убедитесь, что Milvus запущен (`docker compose up -d`) и коллекция создана (`python utils/db_make.py`).")


def main():
    args = parse_arguments()
    search_knowledge_base(
        query=args.query, limit=args.limit, subcategory=args.subcategory
    )


if __name__ == "__main__":
    main()
