import os

os.environ["OPENAI_API_BASE"] = "http://192.168.2.87:8000/v1"
os.environ["OPENAI_API_KEY"] = "api-fake"

# === Настройки ===
MODEL_ID = "Tlite"
TOKEN = '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'
DATA_PATH = 'rp/witcher3_knowledge_base.json'

import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Загрузка данных
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Преобразование в документы
documents = [
    Document(
        page_content=f"[Заголовок: {item['title']}]\n\n{item['text']}",
        metadata={
            "title": item["title"],
            "url": item["url"],
            "category": item["source_category"],
            "exact_title": item["title"].strip()
        }
    )
    for item in data
]

# Разбиение на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Эмбеддинги
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# Подключение к Milvus и загрузка данных
vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="witcher3_rag",  # имя коллекции
    connection_args={"host": "localhost", "port": "19530"},
    drop_old=True  # удалить старую коллекцию, если есть
)

print(f"Загружено {len(chunks)} чанков в Milvus.")