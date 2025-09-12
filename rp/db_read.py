from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
model = SentenceTransformer("BAAI/bge-m3")

query = "Кто такой Геральт"
query_emb = model.encode(query, normalize_embeddings=True).tolist()

res = client.search(
    collection_name="witcher3_rag",
    data=[query_emb],
    anns_field="embedding",
    limit=3,
    output_fields=["title", "text", "url"],
    metric_type="COSINE"  # лучше явно указать
)

# Красивый вывод
for hit in res[0]:
    print(f"\n🎯 Релевантность: {hit['distance']:.4f}")
    print(f"📘 {hit['entity']['title']}")
    print(f"🔗 {hit['entity']['url']}")
    print(f"📝 {hit['entity']['text'][:500]}...")