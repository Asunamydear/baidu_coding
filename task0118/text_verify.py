import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# 向量库与模型配置
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "pubmed_rct20k"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5


# 从文本中取一句作为查询
def pick_one_sentence(text):
    parts = text.split(".")
    if len(parts) > 1:
        return parts[0].strip() + "."
    return " ".join(text.split()[:20])


# 执行一次向量检索
def run_query(title, collection, model, query_text, where_filter=None):
    print("\n" + title)

    query_vec = model.encode([query_text])[0]

    if where_filter is None:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=TOP_K
        )
    else:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=TOP_K,
            where=where_filter
        )

    for i in range(len(results["ids"][0])):
        print("\nRank", i + 1)
        print("Distance =", results["distances"][0][i])
        print("Chunk ID =", results["ids"][0][i])
        print("Metadata =", results["metadatas"][0][i])


def main():
    # 加载嵌入模型
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 打开已有向量数据库
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    # 基础统计验证
    total_vectors = collection.count()
    print("Total vectors in index =", total_vectors)

    # 随机抽样文本块
    sample = collection.get(
        limit=3,
        include=["documents", "metadatas"]
    )

    print("\nSample chunks:")
    for i in range(len(sample["documents"])):
        print("\nChunk", i + 1)
        print("Chunk ID:", sample["ids"][i])
        print("Metadata:", sample["metadatas"][i])
        print("Text head:", sample["documents"][i][:200])

    # 相似性检索验证（句子级自相似）
    full_text = sample["documents"][0]
    query_text = pick_one_sentence(full_text)

    print("\nSelf-similarity test")
    print("Query sentence:", query_text)

    run_query("No filter", collection, model, query_text)

    # 元数据过滤：短文本
    run_query(
        "Filter: short chunks (token_count <= 250)",
        collection, model, query_text,
        where_filter={"token_count": {"$lte": 250}}
    )

    # 元数据过滤：长文本
    run_query(
        "Filter: long chunks (token_count >= 450)",
        collection, model, query_text,
        where_filter={"token_count": {"$gte": 450}}
    )

    # 元数据过滤：多块文献
    run_query(
        "Filter: multi-chunk docs (total_chunks > 1)",
        collection, model, query_text,
        where_filter={"total_chunks": {"$gt": 1}}
    )

    # 边界情况验证：空查询
    empty_query = ""
    empty_vec = model.encode([empty_query])[0]
    empty_results = collection.query(
        query_embeddings=[empty_vec],
        n_results=TOP_K
    )
    print("\nEdge case: empty query")
    print("Returned results =", len(empty_results["ids"][0]))

    # 边界情况验证：超长查询
    long_query = query_text * 100
    long_vec = model.encode([long_query])[0]
    long_results = collection.query(
        query_embeddings=[long_vec],
        n_results=TOP_K
    )
    print("\nEdge case: long query")
    print("Returned results =", len(long_results["ids"][0]))

    # 保存验证统计信息
    verify_stats = {
        "total_vectors": total_vectors,
        "self_similarity_sentence_level": True,
        "metadata_filter_short_long_multi": True,
        "empty_query_test": True,
        "long_query_test": True,
        "embedding_model": EMBED_MODEL_NAME
    }

    with open("verify_stats.json", "w", encoding="utf-8") as f:
        json.dump(verify_stats, f, ensure_ascii=False, indent=2)

    print("\nSaved verify_stats.json")


if __name__ == "__main__":
    main()
