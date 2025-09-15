import os
import logging
import requests
from typing import Dict, List, Any, ClassVar
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from sentence_transformers import CrossEncoder
from pymilvus import Collection, connections

# Конфигурация
MODEL_ID = "Tlite"
TOKEN = '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'
DATA_PATH = 'rp/witcher3_knowledge_base.json'
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "witcher3_rag"

os.environ["OPENAI_API_BASE"] = "http://192.168.2.87:8000/v1"
os.environ["OPENAI_API_KEY"] = "api-fake"

class HybridMilvusRetriever(BaseRetriever):
    """
    Гибридный ретривер: сначала векторный поиск, затем rerank кросс-энкодером.
    Совместим с LangChain.
    """
    vectorstore: Any
    collection: Any  # ← ДОБАВЛЯЕМ! Нативный pymilvus.Collection
    embedding_function: Any
    reranker_model_name: str = "BAAI/bge-reranker-large"
    k: int = 4
    fetch_k: int = 20
    ef: int = 40  # ← будем использовать в expanded_search!
    all_categories: ClassVar[List[str]] = []
    _reranker: CrossEncoder = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reranker = CrossEncoder(self.reranker_model_name)
        logger.info(f"✅ Загружен reranker: {self.reranker_model_name}")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            # Передаём collection, а не vectorstore!
            top_hits = expanded_search(
                collection=self.collection,  # ← ВАЖНО!
                all_categories=self.all_categories,
                original_query=query,
                model=self.embedding_function,
                reranker=self._reranker,  # ← ПЕРЕДАЁМ, чтобы не создавать заново
                k=self.fetch_k,
                ef=self.ef  # ← ПЕРЕДАЁМ ef!
            )

            docs = [
                Document(
                    page_content=hit.entity.get("text"),
                    metadata={
                        "title": hit.entity.get("title"),
                        "url": hit.entity.get("url"),
                        "subcategory": hit.entity.get("subcategory"),
                        "distance": hit.distance
                    }
                )
                for hit in top_hits
            ]
            logger.info(f"🔍 Найдено {len(docs)} документов на этапе dense поиска.")

            # Используем уже загруженный reranker для финального ранжирования
            if self._reranker and len(docs) > self.k:
                pairs = [(query, doc.page_content) for doc in docs]
                scores = self._reranker.predict(pairs)
                scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                docs = [doc for doc, score in scored_docs[:self.k]]

            return docs

        except Exception as e:
            logger.error(f"❌ Ошибка при векторном поиске: {e}")
            return []

def connect_to_milvus():
    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            primary_field="id",
            text_field="text",
            vector_field="vector"
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Ошибка при подключении к Milvus: {e}")
        raise

def get_unique_subcategories(collection: Collection, limit: int = None) -> list:
    """
    Возвращает список уникальных значений столбца `subcategory`.
    """
    # Определяем, какие поля нужно извлечь
    output_fields = ["subcategory"]

    # Извлекаем все записи (или ограниченное количество, если данных много)
    if limit:
        results = collection.query(
            expr="",  # Пустой expr означает "все записи"
            output_fields=output_fields,
            limit=limit
        )
    else:
        # Если данных много, лучше извлекать порциями
        offset = 0
        batch_size = 1000
        results = []
        while True:
            batch = collection.query(
                expr="",
                output_fields=output_fields,
                offset=offset,
                limit=batch_size
            )
            if not batch:
                break
            results.extend(batch)
            offset += batch_size

    # Выделяем уникальные значения
    unique_subcategories = set()
    for record in results:
        subcategory = record.get("subcategory")
        if subcategory:
            unique_subcategories.add(subcategory)

    return list(unique_subcategories)

def format_docs_with_sources(docs):
    result = ""
    sources = set()
    categories = set()
    for doc in docs:
        result += f"[Начало документа]\n{doc.page_content}\n[Конец документа]\n"
        sources.add(doc.metadata.get("url", "Источник не указан"))
        categories.add(doc.metadata.get("subcategory", "Категории нет"))
    return result.strip(), list(sources), list(categories)

def retrieve_with_sources(question: str) -> Dict:
    try:
        docs = retriever.invoke(question)
        context, sources , categories = format_docs_with_sources(docs)

        logger.info(f"Найден контекст для вопроса: {question}")
        logger.info(f"Контекст: {context}")  # Логирование первых 200 символов
        logger.info(f"Источники: {sources}")
        logger.info(f"Категории: {categories}")

        return {"context": context, "sources": sources, "categories": categories, "question": question}
    except Exception as e:
        logger.error(f"Ошибка при получении контекста: {e}")
        return {"context": "", "sources": [],  "categories": [], "question": question}

def generate_paraphrases(query: str, num_paraphrases: int = 3) -> List[str]:
    """
    Генерирует несколько переформулировок запроса с помощью LLM.
    """
    url = "http://192.168.2.87:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Переформулируй следующий запрос {num_paraphrases} разными способами:
    Запрос: {query}
    Ответь только списком переформулированных запросов, без дополнительных комментариев.
    """
    data = {
        "model": "Tlite",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        paraphrases = response.json()["choices"][0]["message"]["content"].strip().split("\n")
        print(f"Переформулированные запросы: {[p.strip() for p in paraphrases if p.strip()]}")
        return [p.strip() for p in paraphrases if p.strip()]
    else:
        return [query]  # Возвращаем оригинальный запрос, если что-то пошло не так

def generate_related_queries(query: str, categories: List[str]) -> List[str]:
    """
    Генерирует связанные категории для поиска.
    """
    url = "http://192.168.2.87:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Выбери из списка {categories} 3 категории которые помогут в поиске данных по этому запросу:
    Запрос: {query}
    Ответь только списком этих категорий, так как они записаны, без дополнительных комментариев.
    """
    data = {
        "model": "Tlite",
        "messages": [{"role": "user", "content": prompt}],
        #"temperature": 0.7,
        "max_tokens": 400
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        relevant_categories = response.json()["choices"][0]["message"]["content"].strip().split("\n")
        print(f"Дополнительные связанные запросы: {[c.strip() for c in relevant_categories if c.strip()]}")
        return [c.strip() for c in relevant_categories if c.strip()]
    else:
        return [query] 

def expanded_search(
    collection,
    all_categories: List[str],
    original_query: str,
    model,
    reranker: CrossEncoder,  # ← ПРИНИМАЕМ ГОТОВЫЙ!
    k: int = 5,
    ef: int = 200  # ← ПАРАМЕТР ИЗ РЕТРИВЕРА
    ) -> List[Any]:  # ← Возвращаем хиты Milvus, не Document
    """
    Выполняет расширенный поиск по переформулировкам + rerank.
    """
    # 1. Генерация вариаций запроса
    paraphrases = generate_paraphrases(original_query)
    related_queries = generate_related_queries(original_query, all_categories)
    all_queries = [original_query] + paraphrases + related_queries

    # 2. Эмбеддинги
    query_embs = [model.embed_query(q) for q in all_queries]

    # 3. Поиск по каждому эмбеддингу
    all_results = []
    for emb in query_embs:
        results = collection.search(
            data=[emb],
            anns_field="vector",
            param={"ef": ef},  # ← ИСПОЛЬЗУЕМ ПЕРЕДАННЫЙ ef!
            limit=k,
            output_fields=["title", "text", "url", "subcategory"]
        )
        for hits in results:
            all_results.extend(hits)

    # 4. Убираем дубли
    seen_ids = set()
    unique_results = []
    for hit in all_results:
        if hit.id not in seen_ids:
            seen_ids.add(hit.id)
            unique_results.append(hit)

    # 5. Rerank одним кросс-энкодером (переданным извне!)
    if reranker and len(unique_results) > k:
        pairs = [(original_query, hit.entity.get("text")) for hit in unique_results]
        scores = reranker.predict(pairs)
        scored = sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)
        unique_results = [hit for hit, _ in scored[:k]]

    return unique_results  # top_hits

def get_rag_response(question: str) -> str:
    try:
        result = rag_chain_with_sources.invoke(question)
        return result
    except Exception as e:
        logger.error(f"Ошибка при получении ответа от модели: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = connect_to_milvus()
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)
unique_subcategories = get_unique_subcategories(collection)

retriever = HybridMilvusRetriever(
    vectorstore=vectorstore,
    collection=collection,
    embedding_function=embeddings,
    all_categories=unique_subcategories,
    k=4,
    fetch_k=20,
    ef=200 
)

llm = ChatOpenAI(
    model_name=MODEL_ID,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    #temperature=0.6,
)

template_with_sources = """Ответь на вопрос, используя только приведённый ниже контекст.
Если в контексте нет ответа, скажи: "Я не знаю на основании предоставленных данных".
Контекст:
{context}
Вопрос: {question}
Ответ:"""

prompt = ChatPromptTemplate.from_template(template_with_sources)

rag_chain_with_sources = (
    retrieve_with_sources
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print(">>> Начинаем...")

    while True:
        user_input = input("\n[Ты] ").strip()
        if user_input.lower() in ["выход", "quit", "exit"]:
            break

        # Явно указываем: сначала проверь знания
        result = get_rag_response(user_input)
        print(f"[СИСТЕМА] {result}")