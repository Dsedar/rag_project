import os

os.environ["OPENAI_API_BASE"] = "http://192.168.2.87:8000/v1"
os.environ["OPENAI_API_KEY"] = "api-fake"

# === Настройки ===
MODEL_ID = "Tlite"
TOKEN = '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'
DATA_PATH = 'rp/witcher3_knowledge_base.json'

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Подключение к существующей коллекции Milvus
vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name="witcher_3rag",
    connection_args={"host": "localhost", "port": "19530"}
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# LLM через vLLM
llm = ChatOpenAI(
    model_name=MODEL_ID,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.6,
    #max_tokens=1024,
)


def format_docs_with_sources(docs):
    result = ""
    sources = set()
    for doc in docs:
        result += doc.page_content + "\n\n"
        sources.add(doc.metadata["url"])
    return result.strip(), list(sources)

# Модифицированный retriever
def retrieve_with_sources(question):
    docs = retriever.invoke(question)
    context, sources = format_docs_with_sources(docs)
    
    # 🔍 ОТЛАДКА: выводим контекст и источники
    print("\n" + "="*80)
    print("🔍 НАЙДЕННЫЙ КОНТЕКСТ:")
    print("="*80)
    print(context)
    print("\n📚 ИСТОЧНИКИ:")
    for src in sources:
        print(f"  - {src}")
    print("="*80 + "\n")
    
    return {"context": context, "sources": sources, "question": question}

# Новый промпт
template_with_sources = """Ответь на вопрос, используя только приведённый ниже контекст.
Если в контексте нет ответа, скажи: "Я не знаю на основании предоставленных данных".

Контекст:
{context}

Вопрос: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(template_with_sources)

# Цепочка с источниками
rag_chain_with_sources = (
    retrieve_with_sources
    | prompt
    | llm
    | StrOutputParser()
)

# Тестовый запрос
result = rag_chain_with_sources.invoke("Кто такой Ольгерд?")
#print(rag_chain_with_sources)
print(result)