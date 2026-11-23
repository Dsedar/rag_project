"""
Основной модуль инференса RAG-системы по Ведьмаку 3 (Witcher 3 RAG).

Архитектура пайплайна:
1. Query Expansion & Category Guidance: Генерация переформулировок запроса через LLM.
2. Dense Multi-Query Retrieval: Поиск в Milvus (индекс HNSW) по нормализованным эмбеддингам BGE-M3.
3. Deduplication: Устранение дубликатов фрагментов из разных вариаций запроса.
4. Cross-Encoder Reranking: Точная пересортировка кандидатов моделью BAAI/bge-reranker-large.
5. Context Generation: Формирование структурированного промпта со ссылками на первоисточники.
6. LLM Answering: Ответ с использованием локальной или облачной OpenAI-совместимой модели (vLLM, Ollama, LM Studio).
"""

import argparse
import logging
import sys
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import requests

from rag_config import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_inference")

# Безопасный импорт тяжелых библиотек
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from langchain_openai import ChatOpenAI
    from pymilvus import Collection, connections
    from sentence_transformers import CrossEncoder

    HAS_DEPENDENCIES = True
    _IMPORT_ERROR = None
except ImportError as e:
    HAS_DEPENDENCIES = False
    _IMPORT_ERROR = e
    BaseRetriever = object  # Заглушка для определения класса


# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ГЕНЕРАЦИИ ВАРИАЦИЙ ЗАПРОСОВ
# ==============================================================================

def generate_paraphrases(
    query: str,
    base_url: str = settings.OPENAI_API_BASE,
    api_key: str = settings.OPENAI_API_KEY,
    model: str = settings.LLM_MODEL,
    num_paraphrases: int = 3,
    timeout: float = 8.0,
) -> List[str]:
    """
    Генерирует несколько переформулировок поискового запроса с помощью LLM
    для повышения полноты (recall) семантического поиска.
    """
    prompt = (
        f"Переформулируй следующий поисковый запрос {num_paraphrases} разными способами "
        f"для базы знаний по игре 'Ведьмак 3'.\n"
        f"Запрос: {query}\n"
        f"Ответь исключительно списком переформулировок (каждая с новой строки), без вводных слов и номеров."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 200,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            paraphrases = [
                line.strip().lstrip("-•*0123456789. ")
                for line in content.strip().split("\n")
                if line.strip()
            ]
            logger.info(f"Сгенерированы переформулировки: {paraphrases}")
            return paraphrases[:num_paraphrases]
    except Exception as e:
        logger.warning(f"Не удалось сгенерировать переформулировки (fallback): {e}")

    return []


def generate_related_categories(
    query: str,
    categories: List[str],
    base_url: str = settings.OPENAI_API_BASE,
    api_key: str = settings.OPENAI_API_KEY,
    model: str = settings.LLM_MODEL,
    num_categories: int = 3,
    timeout: float = 8.0,
) -> List[str]:
    """
    Выбирает наиболее релевантные тематические категории для запроса
    из доступного списка рубрик Вики.
    """
    if not categories:
        return []

    sample_categories = categories[:50]
    prompt = (
        f"Выбери из списка категорий не более {num_categories}, наиболее релевантных вопросу: '{query}'.\n"
        f"Список категорий: {sample_categories}\n"
        f"Ответь только выбранными категориями через новую строку, без комментариев."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 150,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            selected = [
                c.strip().lstrip("-•*0123456789. ")
                for c in content.strip().split("\n")
                if c.strip()
            ]
            logger.info(f"Выбраны релевантные категории: {selected}")
            return selected[:num_categories]
    except Exception as e:
        logger.warning(f"Не удалось подобрать категории (fallback): {e}")

    return []


# ==============================================================================
# 2. ГИБРИДНЫЙ РЕТРИВЕР (MILVUS HNSW + BGE CROSS-ENCODER)
# ==============================================================================

class HybridMilvusRetriever(BaseRetriever):
    """
    Продвинутый ретривер для LangChain:
    - Query Expansion (переформулировки)
    - Multi-Query плотный поиск в Milvus
    - Дедупликация кандидатов
    - Финальное переранжирование через CrossEncoder (bge-reranker-large)
    """

    collection: Any = None
    embedding_function: Any = None
    reranker_model_name: str = settings.RERANKER_MODEL
    k: int = settings.TOP_K
    fetch_k: int = settings.FETCH_K
    ef: int = settings.HNSW_EF
    all_categories: ClassVar[List[str]] = []
    _reranker: Optional[Any] = None

    def __init__(self, **kwargs):
        if not HAS_DEPENDENCIES:
            raise ImportError(f"Отсутствуют необходимые библиотеки: {_IMPORT_ERROR}")
        super().__init__(**kwargs)
        try:
            self._reranker = CrossEncoder(self.reranker_model_name)
            logger.info(f"✅ Reranker успешно инициализирован: '{self.reranker_model_name}'")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить CrossEncoder ({e}). Rerank будет пропущен.")
            self._reranker = None

    def _expanded_search(self, original_query: str) -> List[Any]:
        """Выполняет расширенный поиск: генерация вариаций + векторный опрос + дедупликация."""
        paraphrases = generate_paraphrases(original_query)
        related_cats = generate_related_categories(original_query, self.all_categories)
        all_queries = [original_query] + paraphrases + related_cats

        # Генерация эмбеддингов для всех вариаций
        query_embs = [
            self.embedding_function.embed_query(q) for q in all_queries if q.strip()
        ]

        all_hits = []
        for emb in query_embs:
            results = self.collection.search(
                data=[emb],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": self.ef}},
                limit=self.fetch_k,
                output_fields=["title", "text", "url", "subcategory"],
            )
            for hit_list in results:
                all_hits.extend(hit_list)

        # Дедупликация по ID записи
        seen_ids = set()
        unique_hits = []
        for hit in all_hits:
            if hit.id not in seen_ids:
                seen_ids.add(hit.id)
                unique_hits.append(hit)

        # Переранжирование Cross-Encoder'ом
        if self._reranker and len(unique_hits) > self.k:
            pairs = [
                (original_query, hit.entity.get("text", "")) for hit in unique_hits
            ]
            scores = self._reranker.predict(pairs)
            scored_hits = sorted(
                zip(unique_hits, scores), key=lambda x: x[1], reverse=True
            )
            unique_hits = [hit for hit, _ in scored_hits[: self.k]]
        else:
            unique_hits = unique_hits[: self.k]

        return unique_hits

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Any]:
        try:
            top_hits = self._expanded_search(query)
            docs = [
                Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "title": hit.entity.get("title", ""),
                        "url": hit.entity.get("url", ""),
                        "subcategory": hit.entity.get("subcategory", ""),
                        "distance": getattr(hit, "distance", 0.0),
                    },
                )
                for hit in top_hits
            ]
            logger.info(f"🔎 Отобрано {len(docs)} релевантных фрагментов для контекста.")
            return docs
        except Exception as e:
            logger.error(f"❌ Ошибка в HybridMilvusRetriever: {e}", exc_info=True)
            return []


# ==============================================================================
# 3. ПОЛНЫЙ RAG ПАЙПЛАЙН (WitcherRAGPipeline)
# ==============================================================================

class WitcherRAGPipeline:
    """
    Класс управления RAG-пайплайном:
    - Подключение к Milvus
    - Инициализация моделей эмбеддингов и LLM
    - Сборка LCEL-цепочки
    - Выполнение запросов
    """

    def __init__(
        self,
        milvus_host: str = settings.MILVUS_HOST,
        milvus_port: str = settings.MILVUS_PORT,
        collection_name: str = settings.COLLECTION_NAME,
        llm_model: str = settings.LLM_MODEL,
        openai_api_base: str = settings.OPENAI_API_BASE,
        openai_api_key: str = settings.OPENAI_API_KEY,
    ):
        if not HAS_DEPENDENCIES:
            raise ImportError(
                f"Не установлены необходимые пакеты для RAG пайплайна ({_IMPORT_ERROR}).\n"
                f"Установите зависимости: pip install -r requirements.txt"
            )

        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key

        logger.info("⚙️ Инициализация HuggingFace BGE-M3 эмбеддингов...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"🔌 Подключение к коллекции Milvus '{collection_name}'...")
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

        # Получаем уникальные категории для категориального наведения
        self.unique_subcategories = self._fetch_subcategories()
        HybridMilvusRetriever.all_categories = self.unique_subcategories

        # Инициализация ретривера
        self.retriever = HybridMilvusRetriever(
            collection=self.collection,
            embedding_function=self.embeddings,
            k=settings.TOP_K,
            fetch_k=settings.FETCH_K,
            ef=settings.HNSW_EF,
        )

        # Инициализация LLM
        logger.info(f"🤖 Подключение к LLM ({self.llm_model} @ {self.openai_api_base})...")
        self.llm = ChatOpenAI(
            model_name=self.llm_model,
            openai_api_base=self.openai_api_base,
            openai_api_key=self.openai_api_key,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        # Промпт
        template = (
            "Ты — летописец и эксперт по вселенной 'Ведьмак 3: Дикая Охота'.\n"
            "Ответь на вопрос пользователя, опираясь исключительно на предоставленный контекст из базы знаний.\n"
            "Если в контексте нет прямого ответа, честно скажи: 'В базе знаний нет достоверных сведений по этому вопросу'.\n"
            "Не придумывай факты, которых нет в тексте.\n\n"
            "Контекст из архивов:\n"
            "{context}\n\n"
            "Вопрос: {question}\n\n"
            "Подробный и точный ответ:"
        )
        self.prompt = ChatPromptTemplate.from_template(template)

        # Сборка цепочки LCEL
        self.chain = (
            self._prepare_context_and_sources
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _fetch_subcategories(self, limit: int = 200) -> List[str]:
        """Извлекает уникальные подкатегории из коллекции для подсказок поиску."""
        try:
            results = self.collection.query(
                expr="", output_fields=["subcategory"], limit=limit
            )
            cats = {
                r.get("subcategory") for r in results if r.get("subcategory")
            }
            return list(cats)
        except Exception as e:
            logger.warning(f"Не удалось извлечь подкатегории: {e}")
            return []

    def _prepare_context_and_sources(self, question: str) -> Dict[str, Any]:
        """Извлекает документы и форматирует их в строку контекста."""
        docs = self.retriever.invoke(question)
        context_parts = []
        sources = set()
        categories = set()

        for doc in docs:
            title = doc.metadata.get("title", "Источник")
            url = doc.metadata.get("url", "")
            subcat = doc.metadata.get("subcategory", "")

            context_parts.append(f"--- [Документ: {title}] ---\n{doc.page_content}")
            if url:
                sources.add(url)
            if subcat:
                categories.add(subcat)

        return {
            "context": "\n\n".join(context_parts),
            "question": question,
            "sources": list(sources),
            "categories": list(categories),
            "docs": docs,
        }

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Основной метод запроса: возвращает ответ модели, список ссылок на первоисточники
        и найденные рубрики.
        """
        prepared = self._prepare_context_and_sources(question)
        if not prepared["context"].strip():
            return {
                "answer": "К сожалению, в базе знаний не нашлось информации по данному запросу.",
                "sources": [],
                "categories": [],
                "question": question,
            }

        prompt_val = self.prompt.format(
            context=prepared["context"], question=question
        )
        llm_response = self.llm.invoke(prompt_val)
        answer_text = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )

        return {
            "answer": answer_text.strip(),
            "sources": prepared["sources"],
            "categories": prepared["categories"],
            "question": question,
        }


# Глобальный ленивый экземпляр для обратной совместимости
_default_pipeline: Optional[WitcherRAGPipeline] = None


def get_pipeline() -> WitcherRAGPipeline:
    """Возвращает или создает экземпляр WitcherRAGPipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = WitcherRAGPipeline()
    return _default_pipeline


def get_rag_response(question: str) -> str:
    """Простая функция для обратной совместимости со старыми скриптами/ноутбуками."""
    pipeline = get_pipeline()
    res = pipeline.ask(question)
    return res["answer"]


# ==============================================================================
# 4. ИНТЕРАКТИВНЫЙ РЕЖИМ КОМАНДНОЙ СТРОКИ (CLI)
# ==============================================================================

def run_interactive_cli():
    print("\n" + "=" * 70)
    print("🐺  WITCHER 3 RAG ASSISTANT  🐺")
    print("База знаний: 'Ведьмак 3: Дикая Охота' (Fandom Wiki)")
    print(f"Векторная база: Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})")
    print(f"Модель:         {settings.LLM_MODEL}")
    print("Для выхода введите 'выход', 'quit' или 'exit'")
    print("=" * 70 + "\n")

    try:
        pipeline = get_pipeline()
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("💡 Убедитесь, что Milvus запущен (`docker compose up -d`) и сервер LLM доступен.")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n[Ты] 🧙 > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["выход", "quit", "exit", "q"]:
                print("До встречи на большаке, ведьмак!")
                break

            print("\n[Поиск в архивах...]")
            result = pipeline.ask(user_input)

            print(f"\n[Ответ]\n{result['answer']}\n")

            if result["sources"]:
                print("🔗 Источники:")
                for src in result["sources"]:
                    print(f"   • {src}")

            if result["categories"]:
                print(f"🏷️  Категории: {', '.join(result['categories'])}")

        except (KeyboardInterrupt, EOFError):
            print("\nВыход из программы.")
            break
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG инференс по базе знаний Ведьмака 3."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Разовый вопрос к базе знаний (без входа в интерактивный режим)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.question:
        try:
            pipeline = get_pipeline()
            res = pipeline.ask(args.question)
            print(f"\nВопрос: {args.question}")
            print(f"Ответ:  {res['answer']}")
            if res["sources"]:
                print("Источники:", ", ".join(res["sources"]))
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            sys.exit(1)
    else:
        run_interactive_cli()
