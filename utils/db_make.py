import json
import time
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility, MilvusClient
)
from tqdm import tqdm

# ----------------------------
# 1. НАСТРОЙКИ
# ----------------------------
DATA_FILE = "witcher3_knowledge_base.json"  # Выход парсера
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "witcher3_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024

# ----------------------------
# 2. УМНЫЙ ЧАНКИНГ ПО АБЗАЦАМ
# ----------------------------
def chunk_by_paragraphs(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки по абзацам, не разрывая логические блоки.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += (" " + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars - overlap):
                    chunk_part = para[i:i + max_chars - overlap]
                    chunks.append(chunk_part)
            else:
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 65535:
            final_chunks.append(chunk[:65530] + "... [обрезано]")
        else:
            final_chunks.append(chunk)
    return final_chunks

# ----------------------------
# 3. ПОДКЛЮЧЕНИЕ К MILVUS
# ----------------------------
def setup_collection():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✅ Подключено к Milvus (Docker)")
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"🗑️ Удалена старая коллекция: {COLLECTION_NAME}")

    # Добавляем поле subcategory
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    schema = CollectionSchema(fields, description="The Witcher 3 RAG")
    collection = Collection(COLLECTION_NAME, schema)

    # HNSW — хорош для точного поиска
    # M: Максимальное количество связей для каждого узла. Увеличение улучшает качество поиска, но замедляет индексацию.
    # efConstruction: Размер динамического списка кандидатов во время построения индекса. Чем больше, тем лучше качество, но медленнее построение.
    # ef: Размер динамического списка кандидатов во время поиска. Чем больше, тем точнее поиск, но медленнее.
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200, "ef": 100}
    }
    collection.create_index("vector", index_params)
    collection.load()
    print("✅ Коллекция создана и индексирована")
    return collection

# ----------------------------
# 4. ЗАГРУЗКА И ВСТАВКА
# ----------------------------
def load_and_insert(collection: Collection):
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"Нет файла: {DATA_FILE}")
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ Загружено {len(data)} статей")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Модель {EMBEDDING_MODEL} загружена")

    titles, texts, urls, subcategories, vectors = [], [], [], [], []

    # Создаем один индикатор прогресса для всего процесса
    with tqdm(total=len(data), desc="Обработка", unit="item") as pbar:
        for item in data:
            title = item.get('title', 'No Title')
            formatted_title = f"{title[:20]}..." if len(title) > 23 else title
            print(f"\r\033[94m{formatted_title}\033[0m", end="")
            
            # Обработка статьи
            chunks = chunk_by_paragraphs(item["text"], max_chars=3000)
            for chunk in chunks:
                if len(chunk) >= 65535:
                    chunk = chunk[:65000] + "..."
                titles.append(item["title"])
                texts.append(chunk)
                urls.append(item["url"])
                subcategories.append(item.get("subcategory", "unknown"))  # Добавляем подкатегорию
                emb = model.encode(chunk, normalize_embeddings=True)
                vectors.append(emb.tolist())
            
            # Обновляем индикатор прогресса
            pbar.update(1)

    print(f"🔄 Всего чанков: {len(texts)}")
    print("📥 Вставка в Milvus...")
    collection.insert([titles, texts, urls, subcategories, vectors])
    collection.flush()
    print(f"✅ Успешно вставлено {len(texts)} векторов")

    # Ждём, пока данные будут доступны
    print("⏳ Ожидание подтверждения вставки...")
    collection.num_entities

    # Ждём, пока индекс будет полностью построен
    print("⏳ Ожидание завершения построения индекса...")
    utility.wait_for_index_building_complete(COLLECTION_NAME)
    print("✅ Индекс построен")

    # Перезагружаем коллекцию
    print("🔁 Перезагрузка коллекции в память...")
    collection.load()
    print("✅ Коллекция готова к поиску")

# ----------------------------
# 5. ПРОВЕРКА ПОИСКА С ФИЛЬТРОМ ПО ПОДКАТЕГОРИИ
# ----------------------------
def test_search(collection: Collection):
    model = SentenceTransformer(EMBEDDING_MODEL)
    query = "Что такое Дикая Охота?"
    query_emb = model.encode(query, normalize_embeddings=True).tolist()

    # Поиск с фильтром по подкатегории
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 200}
    }
    try:
        # Пример фильтра: искать только в подкатегории "квесты"
        results = collection.search(
            data=[query_emb],
            anns_field="vector",
            param=search_params,
            limit=3,
            #expr='subcategory == "квесты"',  # Фильтр по подкатегории
            output_fields=["title", "text", "url", "subcategory"]
        )
        #print(f"\n🔍 Поиск: '{query}' (фильтр: subcategory == 'квесты')")
        for hits in results:
            for hit in hits:
                print(f"\n--- Результат {hit.rank} | Схожесть: {hit.distance:.4f} ---")
                print(f"📌 {hit.entity.get('title')}")
                print(f"🔗 {hit.entity.get('url')}")
                print(f"🏷️ {hit.entity.get('subcategory')}")
                print(f"💬 {hit.entity.get('text')[:400]}...")
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")

# ----------------------------
# 6. ЗАПУСК
# ----------------------------
if __name__ == "__main__":
    try:
        collection = setup_collection()
        load_and_insert(collection)
        test_search(collection)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        connections.disconnect("default")
