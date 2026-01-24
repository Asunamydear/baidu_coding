# build_index_and_verify.py
# 作用：将文本块向量化，构建向量索引，并进行质量验证

import os
import json
import random
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# 1 嵌入模型选择与加载
CHUNKS_FILE = "chunks.jsonl"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "pubmed_rct20k"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
TOP_K = 5


def load_chunks_jsonl(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    # 加载文本块数据
    chunks = load_chunks_jsonl(CHUNKS_FILE)
    print("Loaded chunks =", len(chunks))

    # 初始化嵌入模型
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 初始化持久化向量数据库
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # 创建或加载向量集合并使用余弦相似度
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 批量生成向量并写入索引
    ids_batch = []
    docs_batch = []
    metas_batch = []
    total_added = 0

    for i, c in enumerate(chunks):
        ids_batch.append(str(c["chunk_id"]))
        docs_batch.append(c["text"])
        metas_batch.append({
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "token_count": c["token_count"]
        })

        if len(ids_batch) == BATCH_SIZE or i == len(chunks) - 1:
            embeddings = model.encode(docs_batch, show_progress_bar=False)
            collection.add(
                ids=ids_batch,
                documents=docs_batch,
                metadatas=metas_batch,
                embeddings=embeddings
            )
            total_added += len(ids_batch)
            print("Added:", total_added)
            ids_batch = []
            docs_batch = []
            metas_batch = []

    print("Done. Total added to collection =", total_added)

    # 2 向量数据库配置与索引构建完成后的统计信息保存
    index_stats = {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total_added,
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "index_built_at": datetime.now().isoformat(),
        "metadata_fields": ["doc_id", "chunk_index", "total_chunks", "token_count"]
    }

    with open("index_stats.json", "w", encoding="utf-8") as f:
        json.dump(index_stats, f, ensure_ascii=False, indent=2)

    # # 3 质量验证工作：基础检索测试//这个功能做的不好，后续单独做一个文件运行，
    # query_text = "treatment effect on endothelial function in COPD"
    # query_vec = model.encode([query_text])[0]
    #
    # results = collection.query(
    #     query_embeddings=[query_vec],
    #     n_results=TOP_K
    # )
    #
    # print("\nTEST QUERY:", query_text)
    # for i in range(TOP_K):
    #     print("\nRank", i + 1)
    #     print("Distance =", results["distances"][0][i])
    #     print("Chunk ID =", results["ids"][0][i])
    #     print("Metadata =", results["metadatas"][0][i])
    #     print("Text head =", results["documents"][0][i][:200])
    #
    # # 质量验证：自相似性与边界情况
    # verify_stats = {
    #     "total_vectors": collection.count(),
    #     "test_query": query_text,
    #     "returned_results": TOP_K,
    #     "empty_query_handled": True,
    #     "long_query_handled": True
    # }
    # # 元数据过滤检索（示例：只要 token_count < 400 的 chunk）//后续单独做一个过滤检索，这里 做个示范
    # filtered_results = collection.query(
    #     query_embeddings=[query_vec],
    #     n_results=TOP_K,
    #     where={"token_count": {"$lt": 400}}
    # )
    #
    # print("\nFILTERED QUERY (token_count < 400):", query_text)
    # for i in range(len(filtered_results["ids"][0])):
    #     print("\nRank", i + 1)
    #     print("Distance =", filtered_results["distances"][0][i])
    #     print("Chunk ID =", filtered_results["ids"][0][i])
    #     print("Metadata =", filtered_results["metadatas"][0][i])
    #     print("Text head =", filtered_results["documents"][0][i][:200])
    #
    # with open("verify_stats.json", "w", encoding="utf-8") as f:
    #     json.dump(verify_stats, f, ensure_ascii=False, indent=2)
    #
    # print("\nSaved verify_stats.json")


if __name__ == "__main__":
    main()
