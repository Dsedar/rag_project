"""
Индексация базы знаний Ведьмака 3 в векторную базу данных Milvus:
- Загрузка статей из JSON
- Умный чанкинг по смысловым абзацам
- Пакетная генерация эмбеддингов через BAAI/bge-m3
- Создание коллекции Milvus со схемой и HNSW индексом
- Запись данных и построение индекса
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Добавляем корень проекта в sys.path для импорта rag_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("db_make")


def chunk_by_paragraphs(
    text: str, max_chars: int = 2000, overlap: int = 100
) -> List[str]:
    """
    Разбивает текст на логические чанки по абзацам.
    Сохраняет целостность смысловых блоков, избегает разрыва предложений.
    """
    paragraphs = text.split("\n")
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Если абзац помещается в текущий чанк
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # Если отдельный абзац превышает max_chars, режем его с перекрытием
            if len(para) > max_chars:
                step = max(max_chars - overlap, 100)
                for i in range(0, len(para), step):
                    chunks.append(para[i : i + max_chars])
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Ограничение под максимальный размер VARCHAR в Milvus (65535 байт)
    sanitized_chunks = []
    for chunk in chunks:
        if len(chunk) > 65000:
            sanitized_chunks.append(chunk[:65000] + "... [обрезано]")
        else:
            sanitized_chunks.append(chunk)

    return sanitized_chunks


def setup_collection(
    host: str = settings.MILVUS_HOST,
    port: str = settings.MILVUS_PORT,
    collection_name: str = settings.COLLECTION_NAME,
    vector_dim: int = settings.VECTOR_DIM,
    recreate: bool = True,
):
    """Подключается к Milvus и инициализирует коллекцию со схемой и HNSW-индексом."""
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    logger.info(f"🔌 Подключение к Milvus ({host}:{port})...")
    connections.connect("default", host=host, port=port)
    logger.info("✅ Успешное подключение к Milvus.")

    if utility.has_collection(collection_name):
        if recreate:
            logger.info(f"🗑️ Удаление существующей коллекции: '{collection_name}'")
            utility.drop_collection(collection_name)
        else:
            logger.info(f"ℹ️ Коллекция '{collection_name}' уже существует. Загрузка...")
            collection = Collection(collection_name)
            collection.load()
            return collection

    # Описание полей коллекции
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="The Witcher 3 RAG Knowledge Base (Fandom Wiki)",
        enable_dynamic_field=False,
    )
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"✅ Создана коллекция: '{collection_name}'")

    # Создание индекса HNSW для быстрого и точного косинусного поиска
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    logger.info(f"🔨 Построение HNSW индекса для поля 'vector'...")
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info("✅ Индекс успешно инициализирован.")

    return collection


def load_and_index_data(
    collection,
    data_file: Path,
    batch_size: int = 64,
    embedding_model_name: str = settings.EMBEDDING_MODEL,
):
    """
    Считывает статьи из JSON, формирует чанки, выполняет пакетный расчет
    эмбеддингов и загружает данные в Milvus.
    """
    from pymilvus import utility
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    if not data_file.exists():
        raise FileNotFoundError(
            f"Файл с базой знаний '{data_file}' не найден. "
            "Запустите wiki_parser.py или укажите верный путь."
        )

    logger.info(f"📖 Чтение статей из '{data_file}'...")
    with open(data_file, "r", encoding="utf-8") as f:
        articles: List[Dict] = json.load(f)
    logger.info(f"✅ Загружено {len(articles)} статей.")

    logger.info(f"🧠 Загрузка модели эмбеддингов '{embedding_model_name}'...")
    model = SentenceTransformer(embedding_model_name)
    logger.info("✅ Модель эмбеддингов загружена.")

    # 1. Формирование всех чанков с метаданными
    logger.info("✂️ Нарезка статей на смысловые чанки...")
    raw_chunks: List[Dict[str, str]] = []

    for item in articles:
        title = item.get("title", "Без названия")
        url = item.get("url", "")
        subcategory = item.get("subcategory", "Категория:Ведьмак 3")
        text = item.get("text", "")

        if not text.strip():
            continue

        chunks = chunk_by_paragraphs(
            text,
            max_chars=settings.CHUNK_MAX_CHARS,
            overlap=settings.CHUNK_OVERLAP,
        )

        for chunk in chunks:
            raw_chunks.append(
                {
                    "title": title,
                    "text": chunk,
                    "url": url,
                    "subcategory": subcategory,
                }
            )

    total_chunks = len(raw_chunks)
    logger.info(f"📊 Итого сформировано чанков для индексации: {total_chunks}")

    # 2. Пакетная генерация эмбеддингов и вставка в Milvus
    insert_batch_size = 512
    logger.info(
        f"⚡ Генерация эмбеддингов (batch_size={batch_size}) и вставка в Milvus..."
    )

    start_time = time.time()
    for i in tqdm(
        range(0, total_chunks, insert_batch_size), desc="Индексация в Milvus"
    ):
        batch = raw_chunks[i : i + insert_batch_size]
        batch_texts = [item["text"] for item in batch]

        # Пакетное кодирование (быстрее в разы)
        batch_vectors = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        b_titles = [item["title"] for item in batch]
        b_urls = [item["url"] for item in batch]
        b_subcats = [item["subcategory"] for item in batch]

        collection.insert([b_titles, batch_texts, b_urls, b_subcats, batch_vectors])

    # Фиксация данных
    logger.info("💾 Сброс буферов (flush) в Milvus...")
    collection.flush()

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Вставка завершена за {elapsed:.1f} сек. ({total_chunks / max(elapsed, 0.01):.1f} чанков/сек)")

    logger.info("⏳ Ожидание завершения построения индекса...")
    utility.wait_for_index_building_complete(collection.name)

    logger.info("🔁 Загрузка коллекции в оперативную память Milvus...")
    collection.load()
    logger.info(f"🎉 Индексация полностью завершена! Всего векторов в базе: {collection.num_entities}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Создание и индексация коллекции Witcher 3 в Milvus."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(settings.DATA_FILE),
        help=f"Путь к JSON-файлу базы знаний (по умолчанию: '{settings.DATA_FILE}')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер пакета (batch size) для генерации эмбеддингов (по умолчанию: 64)",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Не удалять существующую коллекцию, если она уже создана",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        from pymilvus import connections
    except ImportError as e:
        logger.error(
            f"❌ Не установлены необходимые пакеты ({e}).\n"
            f"Установите зависимости: pip install -r requirements.txt"
        )
        sys.exit(1)

    try:
        collection = setup_collection(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.COLLECTION_NAME,
            vector_dim=settings.VECTOR_DIM,
            recreate=not args.no_recreate,
        )
        load_and_index_data(
            collection=collection,
            data_file=Path(args.data_file),
            batch_size=args.batch_size,
            embedding_model_name=settings.EMBEDDING_MODEL,
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при индексации: {e}", exc_info=True)
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()
