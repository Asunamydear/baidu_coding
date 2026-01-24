# build_index.py
# 作用：
# 1) 读取 chunks.jsonl
# 2) 用 BGE embedding 模型把每个 chunk 变成向量
# 3) 存进 ChromaDB（持久化）
# 4) 做一次简单检索测试

import os
import json
from datetime import datetime

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer


# ====== 你需要改的配置 ======
CHUNKS_FILE = "chunks.jsonl"          # 你上周生成的文件
CHROMA_DIR = "chroma_store"           # 向量库保存目录（会自动创建）
COLLECTION_NAME = "pubmed_rct20k"     # collection 名字

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64                       # 显存/内存不够就调小：32/16
TOP_K = 5
# ==========================


def load_chunks_jsonl(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
    return chunks


def main():
    # 1) 读 chunks
    if not os.path.exists(CHUNKS_FILE):
        print("ERROR: chunks file not found:", CHUNKS_FILE)
        return

    chunks = load_chunks_jsonl(CHUNKS_FILE)
    print("Loaded chunks =", len(chunks))

    # 2) 加载 embedding 模型
    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3) 初始化 ChromaDB（持久化）
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # 如果已存在同名 collection，会直接打开
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # 余弦相似度
    )

    # 4) 批量写入
    # Chroma 要求：ids / documents / metadatas / embeddings（可选）
    # 我们用 embedding 模型生成 embeddings

    ids_batch = []
    docs_batch = []
    metas_batch = []

    total_added = 0

    for i in range(len(chunks)):
        c = chunks[i]

        chunk_id = str(c["chunk_id"])
        text = c["text"]

        meta = {
            "doc_id": str(c.get("doc_id", "")),
            "chunk_index": int(c.get("chunk_index", 0)),
            "total_chunks": int(c.get("total_chunks", 1)),
            "token_count": int(c.get("token_count", 0)),
            # 你以后如果有 title/journal/pub_date，也可以加在这里
        }

        ids_batch.append(chunk_id)
        docs_batch.append(text)
        metas_batch.append(meta)

        # 达到 batch_size 或最后一批，就生成 embedding 并写入
        if len(ids_batch) == BATCH_SIZE or i == len(chunks) - 1:
            # 生成 embeddings
            embeddings = model.encode(docs_batch, show_progress_bar=False)

            # 写入 Chroma
            collection.add(
                ids=ids_batch,
                documents=docs_batch,
                metadatas=metas_batch,
                embeddings=embeddings
            )

            total_added += len(ids_batch)
            print("Added:", total_added)

            # 清空 batch
            ids_batch = []
            docs_batch = []
            metas_batch = []

    print("Done. Total added to collection =", total_added)

    # 5) 简单检索测试：query → embedding → search
    query_text = "treatment effect on endothelial function in COPD"
    query_vec = model.encode([query_text])[0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=TOP_K
    )

    print("\n===== TEST QUERY =====")
    print("Query:", query_text)
    print("Top results:")
    for rank in range(TOP_K):
        rid = results["ids"][0][rank]
        rdoc = results["documents"][0][rank]
        rmeta = results["metadatas"][0][rank]
        rdist = results["distances"][0][rank]

        print("\n--- rank", rank + 1, "distance =", rdist)
        print("chunk_id:", rid)
        print("meta:", rmeta)
        print("text_head:", rdoc[:200])

    # 6) 保存 stats（可选，但老师要统计信息）
    stats = {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total_added,
        "embedding_model": EMBED_MODEL_NAME,
        "index_built_at": datetime.now().isoformat(),
        "metadata_fields": ["doc_id", "chunk_index", "total_chunks", "token_count"]
    }

    with open("index_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nSaved index_stats.json")


if __name__ == "__main__":
    main()
