# task0202/pipeline_retrieve.py
# 作用：完整检索流水线（Query增强 -> MultiPath检索 -> 融合 -> Reranker重排序 -> 输出证据文档列表）
#
# 目录假设：
#   task0118/  (chroma_store, chunks.jsonl)
#   task0125/  (rag_retrieval.py)
#   task0202/  (bm25_index.py, multipath_retriever.py, reranker.py, pipeline_retrieve.py)
#
# 用法（最简单）：
#   python task0202/pipeline_retrieve.py
#

#   query
#   fusion策略（rrf/simple/weighted）
#   topN（先拿多少候选去rerank）
#   topK（最后输出多少条）
#
# 输出：
#   1) 终端打印 TopK reranked results（带分数、doc_id、uid、来源）
#   2) 自动保存 JSON：task0202/retrieval_output.json


import os
import sys
import json
import textwrap
from datetime import datetime

# 保证能 import 到同目录模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # task0202
sys.path.append(BASE_DIR)

from multipath_retriever import MultiPathRetriever

# 引入 task0125 的查询增强
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_DIR)
from task0125.rag_retrieval import process_medical_query

# 引入 reranker（你之前写的）
from reranker import SimpleReranker, RERANK_MODEL


DEFAULT_FUSION = "rrf"      # rrf / simple / weighted
DEFAULT_TOP_N = 20          # 从 fused 里取前 N 条去 rerank
DEFAULT_TOP_K = 10          # 最终输出前 K
DEFAULT_SAVE_NAME = "retrieval_output.json"


def shorten_text(text, width=240):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # 取第一句方便看
    parts = text.split(".")
    if len(parts) > 1:
        text = parts[0].strip() + "."
    return textwrap.shorten(text, width=width, placeholder="...")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def build_reference_list(reranked_items):
    """
    把 reranked_items 变成“证据文档列表”结构，方便后续 RAG 生成使用
    """
    refs = []
    for i, r in enumerate(reranked_items, start=1):
        refs.append({
            "rank": i,
            "doc_id": r.get("doc_id"),
            "uid": r.get("uid"),
            "source": r.get("source"),  # vector / bm25
            "rerank_score": safe_float(r.get("rerank_score", 0.0)),
            "fusion_score": safe_float(r.get("fusion_score", 0.0)),
            "vector_score": safe_float(r.get("vector_score", 0.0)),
            "bm25_score": safe_float(r.get("bm25_score", 0.0)),
            "text_head": r.get("text_head", ""),
            # 如果你想给后续生成用全文，可以加 text（向量结果通常有 text，bm25 可能只有 head）
            "text": r.get("text", "") or r.get("text_head", ""),
            "metadata": {
                "chunk_index": r.get("chunk_index"),
                "total_chunks": r.get("total_chunks"),
                "token_count": r.get("token_count"),
                "raw_id": r.get("raw_id"),
            }
        })
    return refs


def main():
    print("== Retrieval Pipeline ==")
    print("Rerank model =", RERANK_MODEL)

    # 1) 输入 query
    query = input("\nEnter your query (or 'q' to quit): ").strip()
    if not query or query.lower() == "q":
        print("Exit.")
        return

    # 2) fusion / topN / topK
    fusion = input(f"Fusion strategy (rrf/simple/weighted, default {DEFAULT_FUSION}): ").strip().lower()
    if fusion not in ["rrf", "simple", "weighted"]:
        fusion = DEFAULT_FUSION

    top_n_str = input(f"Top N candidates to rerank (default {DEFAULT_TOP_N}): ").strip()
    top_k_str = input(f"Final Top K after rerank (default {DEFAULT_TOP_K}): ").strip()

    top_n = DEFAULT_TOP_N
    top_k = DEFAULT_TOP_K
    if top_n_str.isdigit():
        top_n = int(top_n_str)
    if top_k_str.isdigit():
        top_k = int(top_k_str)

    # 3) Query 增强
    query_info = process_medical_query(query)
    if not query_info.get("ok"):
        print("Query rejected:", query_info.get("reason"))
        return

    print("\n--- Query Info ---")
    print("clean_query   =", query_info["clean_query"])
    print("filters       =", query_info["filters"])
    print("vector_query  =", query_info["vector_query"])
    print("keyword_query =", query_info["keyword_query"])

    # 4) MultiPath 检索 + 融合
    retriever = MultiPathRetriever()

    # 这里 top_k_vector/top_k_keyword 用 top_n，保证 fused 里候选够多
    result = retriever.retrieve(
        query_info=query_info,
        top_k_vector=max(5, top_n),
        top_k_keyword=max(5, top_n),
        fusion_strategy=fusion
    )

    fused = result["fused_results"][:top_n]

    print("\n=== Fused candidates (before rerank) ===")
    for i, r in enumerate(fused, start=1):
        print(f"\nRank {i}")
        print("source     =", r.get("source"))
        print("uid        =", r.get("uid"))
        print("doc_id     =", r.get("doc_id"))
        print("text_head  =", shorten_text(r.get("text_head", "")))

    # 5) Reranker 重排序（relevance）
    reranker = SimpleReranker(RERANK_MODEL)

    # 用 clean_query rerank（你也可以换成 keyword_query 试试）
    reranked = reranker.rerank(
        query=query_info["clean_query"],
        candidates=fused,
        top_k=top_k
    )

    print("\n=== Reranked results (after rerank) ===")
    for i, r in enumerate(reranked, start=1):
        print(f"\nRank {i}")
        print("rerank_score =", round(safe_float(r.get("rerank_score", 0.0)), 6))
        print("source       =", r.get("source"))
        print("uid          =", r.get("uid"))
        print("doc_id       =", r.get("doc_id"))
        print("text_head    =", shorten_text(r.get("text_head", "")))

    # 6) 生成“证据文档列表” + 保存
    references = build_reference_list(reranked)

    output = {
        "created_at": datetime.now().isoformat(),
        "query": query,
        "query_info": {
            "clean_query": query_info.get("clean_query"),
            "filters": query_info.get("filters"),
            "vector_query": query_info.get("vector_query"),
            "keyword_query": query_info.get("keyword_query"),
        },
        "settings": {
            "fusion": fusion,
            "top_n_candidates": top_n,
            "top_k_final": top_k,
            "rerank_model": RERANK_MODEL
        },
        "references": references
    }

    save_path = os.path.join(BASE_DIR, DEFAULT_SAVE_NAME)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nSaved:", save_path)
    print("Done.")


if __name__ == "__main__":
    main()
