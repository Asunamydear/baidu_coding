# task0202/multipath_retriever.py
# 作用：MultiPathRetriever（向量检索 + BM25 检索）+ 融合策略（simple / rrf / weighted）
# 特点：
# - 统一主键 uid = doc_id_chunk_index，便于去重/融合
# - short 等 filters 会先应用到 BM25；如果过滤后为空 -> fallback 回不过滤（改法1）
# - weighted 融合：先归一化再加权（向量权重更高）

import os
import sys
import textwrap

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from bm25_index import BM25Index

# ====== 引入 task0125 的查询处理器 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # task0202
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))   # 项目根目录
sys.path.append(PROJECT_DIR)

from task0125.rag_retrieval import process_medical_query, filters_to_where


# ====== 路径：task0202 和 task0118 同级 ======
TASK0118_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "task0118"))
CHROMA_DIR = os.path.join(TASK0118_DIR, "chroma_store")

COLLECTION_NAME = "pubmed_rct20k"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def pick_one_sentence(text):
    parts = text.split(".")
    if len(parts) > 1:
        return parts[0].strip() + "."
    return " ".join(text.split()[:20])


def shorten_text(text, width=220):
    if not isinstance(text, str):
        text = str(text)
    one = pick_one_sentence(text)
    return textwrap.shorten(one, width=width, placeholder="...")


def make_uid(doc_id, chunk_index):
    if doc_id is None or doc_id == "":
        return ""
    if chunk_index is None:
        chunk_index = 0
    return str(doc_id) + "_" + str(chunk_index)


def match_filters(meta, filters):
    """
    BM25 的过滤（尽量和 Chroma where 对齐）
    支持：
      - doc_id 精确匹配
      - token_count 范围
      - total_chunks 范围
    """
    if not filters:
        return True

    # doc_id
    if "doc_id" in filters:
        if str(meta.get("doc_id", "")) != str(filters["doc_id"]):
            return False

    # token_count
    if "token_count" in filters:
        op, val = filters["token_count"]
        tc = meta.get("token_count", None)
        if tc is None:
            return False
        tc = int(tc)
        val = int(val)

        if op == "<" and not (tc < val):
            return False
        if op == "<=" and not (tc <= val):
            return False
        if op == ">" and not (tc > val):
            return False
        if op == ">=" and not (tc >= val):
            return False

    # total_chunks
    if "total_chunks" in filters:
        op, val = filters["total_chunks"]
        t = meta.get("total_chunks", None)
        if t is None:
            return False
        t = int(t)
        val = int(val)

        if op == "<" and not (t < val):
            return False
        if op == "<=" and not (t <= val):
            return False
        if op == ">" and not (t > val):
            return False
        if op == ">=" and not (t >= val):
            return False

    return True


class MultiPathRetriever:
    def __init__(self):
        # 向量模型
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

        # Chroma collection
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError("Chroma dir not found: " + CHROMA_DIR)

        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = client.get_collection(name=COLLECTION_NAME)

        # BM25
        self.bm25 = BM25Index()
        self.bm25.build_index(os.path.join(TASK0118_DIR, "chunks.jsonl"))

    # ------------------------
    # 向量检索
    # ------------------------
    def vector_search(self, query_text, where_filter=None, top_k=5):
        query_vec = self.model.encode([query_text])[0]

        if where_filter is None:
            results = self.collection.query(query_embeddings=[query_vec], n_results=top_k)
        else:
            results = self.collection.query(query_embeddings=[query_vec], n_results=top_k, where=where_filter)

        out = []
        ids = results["ids"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]
        docs = results["documents"][0]

        for i in range(len(ids)):
            distance = float(dists[i])
            vector_score = 1.0 - distance  # 越大越好

            meta = metas[i] if metas[i] else {}
            text = docs[i] if docs[i] else ""

            doc_id = str(meta.get("doc_id", ""))
            chunk_index = meta.get("chunk_index", 0)
            uid = make_uid(doc_id, chunk_index)

            out.append({
                "source": "vector",
                "uid": uid,
                "raw_id": str(ids[i]),
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "total_chunks": meta.get("total_chunks", None),
                "token_count": meta.get("token_count", None),
                "vector_score": vector_score,
                "bm25_score": 0.0,
                "fusion_score": 0.0,
                "text": text,
                "text_head": shorten_text(text),
            })

        return out

    # ------------------------
    # BM25 检索（改法1：先过滤，空了就 fallback）
    # ------------------------
    def bm25_search(self, query_text, filters=None, top_k=5):
        hits_all = self.bm25.search(query_text, top_k=top_k)

        # 没有 filters：直接用不过滤结果
        if not filters:
            return self._format_bm25_hits(hits_all)

        # 有 filters：先过滤
        filtered = []
        for h in hits_all:
            meta = {
                "doc_id": h["doc_id"],
                "chunk_index": h["chunk_index"],
                "total_chunks": h["total_chunks"],
                "token_count": h["token_count"],
            }
            if match_filters(meta, filters):
                filtered.append(h)

        # 如果过滤后为空 -> fallback 回 hits_all
        use_hits = filtered if len(filtered) > 0 else hits_all

        return self._format_bm25_hits(use_hits)

    def _format_bm25_hits(self, hits):
        out = []
        for h in hits:
            uid = make_uid(h["doc_id"], h["chunk_index"])
            out.append({
                "source": "bm25",
                "uid": uid,
                "raw_id": str(h["chunk_id"]),
                "doc_id": str(h["doc_id"]),
                "chunk_index": h["chunk_index"],
                "total_chunks": h["total_chunks"],
                "token_count": h["token_count"],
                "vector_score": 0.0,
                "bm25_score": float(h["bm25_score"]),
                "fusion_score": 0.0,
                "text": "",
                "text_head": h["text_head"],
            })
        return out

    # ------------------------
    # simple 融合：合并去重（向量优先）
    # ------------------------
    def fuse_simple(self, vec_results, bm25_results):
        seen = set()
        merged = []

        for r in vec_results:
            uid = r["uid"]
            if uid and uid not in seen:
                seen.add(uid)
                merged.append(r)

        for r in bm25_results:
            uid = r["uid"]
            if uid and uid not in seen:
                seen.add(uid)
                merged.append(r)

        return merged

    # ------------------------
    # RRF 融合
    # ------------------------
    def fuse_rrf(self, vec_results, bm25_results, k=60):
        score_map = {}
        item_map = {}

        for rank, r in enumerate(vec_results, start=1):
            uid = r["uid"]
            if not uid:
                continue
            score_map[uid] = score_map.get(uid, 0.0) + 1.0 / (k + rank)
            item_map[uid] = r

        for rank, r in enumerate(bm25_results, start=1):
            uid = r["uid"]
            if not uid:
                continue
            score_map[uid] = score_map.get(uid, 0.0) + 1.0 / (k + rank)
            if uid not in item_map:
                item_map[uid] = r

        merged = []
        for uid, s in score_map.items():
            item = dict(item_map[uid])
            item["fusion_score"] = float(s)
            merged.append(item)

        merged.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        return merged

    # ------------------------
    # weighted 融合（归一化 + 加权）
    # ------------------------
    def fuse_weighted(self, vec_results, bm25_results, alpha=0.7):
        all_items = {}

        # 先放向量
        for r in vec_results:
            uid = r["uid"]
            if uid:
                all_items[uid] = dict(r)

        # 再把 bm25 分数合并进去
        for r in bm25_results:
            uid = r["uid"]
            if not uid:
                continue
            if uid not in all_items:
                all_items[uid] = dict(r)
            else:
                all_items[uid]["bm25_score"] = r.get("bm25_score", 0.0)

        vec_scores = []
        bm25_scores = []
        for item in all_items.values():
            vec_scores.append(float(item.get("vector_score", 0.0)))
            bm25_scores.append(float(item.get("bm25_score", 0.0)))

        vmin, vmax = min(vec_scores), max(vec_scores)
        bmin, bmax = min(bm25_scores), max(bm25_scores)

        def norm(x, mn, mx):
            if mx - mn < 1e-12:
                return 0.0
            return (x - mn) / (mx - mn)

        merged = []
        for uid, item in all_items.items():
            v = float(item.get("vector_score", 0.0))
            b = float(item.get("bm25_score", 0.0))
            v_norm = norm(v, vmin, vmax)
            b_norm = norm(b, bmin, bmax)

            fusion = alpha * v_norm + (1.0 - alpha) * b_norm
            item["fusion_score"] = float(fusion)
            merged.append(item)

        merged.sort(key=lambda x: x.get("fusion_score", 0.0), reverse=True)
        return merged

    # ------------------------
    # 对外统一接口
    # ------------------------
    def retrieve(self, query_info, top_k_vector=5, top_k_keyword=5, fusion_strategy="rrf"):
        filters = query_info.get("filters", {})
        where_filter = filters_to_where(filters)

        vec_results = self.vector_search(
            query_text=query_info["vector_query"],
            where_filter=where_filter,
            top_k=top_k_vector
        )

        # 改法1：BM25 先按 filters 过滤，空则 fallback
        bm25_results = self.bm25_search(
            query_text=query_info["keyword_query"],
            filters=filters,
            top_k=top_k_keyword
        )

        if fusion_strategy == "simple":
            fused = self.fuse_simple(vec_results, bm25_results)
        elif fusion_strategy == "weighted":
            fused = self.fuse_weighted(vec_results, bm25_results, alpha=0.7)
        else:
            fused = self.fuse_rrf(vec_results, bm25_results, k=60)

        return {
            "vector_results": vec_results,
            "bm25_results": bm25_results,
            "fused_results": fused
        }


def main():
    print("== MultiPathRetriever Demo ==")
    retriever = MultiPathRetriever()
    print("Vector index size =", retriever.collection.count())

    q = input("\nEnter your query: ").strip()
    query_info = process_medical_query(q)

    if not query_info.get("ok"):
        print("Query rejected:", query_info.get("reason"))
        return

    print("\n--- Query Info ---")
    print("clean_query =", query_info["clean_query"])
    print("filters     =", query_info["filters"])
    print("vector_query=", query_info["vector_query"])
    print("keyword_query=", query_info["keyword_query"])

    fusion = input("\nFusion strategy (rrf/simple/weighted, default rrf): ").strip().lower()
    if fusion not in ["rrf", "simple", "weighted"]:
        fusion = "rrf"

    out = retriever.retrieve(
        query_info=query_info,
        top_k_vector=5,
        top_k_keyword=5,
        fusion_strategy=fusion
    )

    print("\n=== Vector Results ===")
    for i, r in enumerate(out["vector_results"], start=1):
        print(f"\nRank {i}")
        print("vector_score =", round(r["vector_score"], 4))
        print("uid          =", r["uid"])
        print("doc_id       =", r["doc_id"])
        print("text_head    =", r["text_head"])

    print("\n=== BM25 Results (after filters + fallback) ===")
    for i, r in enumerate(out["bm25_results"], start=1):
        print(f"\nRank {i}")
        print("bm25_score =", round(r["bm25_score"], 4))
        print("uid        =", r["uid"])
        print("doc_id     =", r["doc_id"])
        print("token_count=", r["token_count"])
        print("text_head  =", r["text_head"])

    print("\n=== Fused Results ===")
    for i, r in enumerate(out["fused_results"][:10], start=1):
        print(f"\nRank {i}")
        print("fusion_score =", round(r.get("fusion_score", 0.0), 6))
        print("source       =", r.get("source"))
        print("uid          =", r["uid"])
        print("doc_id       =", r["doc_id"])
        print("text_head    =", r["text_head"])


if __name__ == "__main__":
    main()
