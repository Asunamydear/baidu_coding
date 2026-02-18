# task0202/reranker.py
# 作用：用 BAAI/bge-reranker-base 对 multipath 的 fused_results 做重排序（relevance）
# 用法：
#   python task0202/reranker.py
# 然后输入 query（例如：metformin cardiovascular disease）
# 先跑 multipath 得到 fused_results，再对 topN 做 rerank，输出最终 topK。
#
# 依赖：
#   pip install transformers torch
#   （你已经有 sentence-transformers/chromadb 等）


import os
import sys
import textwrap

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 让 Python 能 import 到同目录的 multipath_retriever
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # task0202
sys.path.append(BASE_DIR)

from multipath_retriever import MultiPathRetriever


RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TOP_N = 20   # 先从 fused_results 里取前 N 条来 rerank
DEFAULT_TOP_K = 10   # 最终输出前 K 条


def shorten_text(text, width=220):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # 取第一句做展示
    parts = text.split(".")
    if len(parts) > 1:
        text = parts[0].strip() + "."
    return textwrap.shorten(text, width=width, placeholder="...")


class SimpleReranker:
    def __init__(self, model_name=RERANK_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # 用 GPU 就用 GPU（没有就 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query, passages):
        """
        给一批 passages 打分（越大越相关）
        返回：scores 列表，长度与 passages 相同
        """
        if len(passages) == 0:
            return []

        # tokenizer 支持成对输入：(query, passage)
        pairs = [(query, p) for p in passages]

        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            out = self.model(**encoded)

            # 一般 bge-reranker-base 是单输出 logit
            logits = out.logits.squeeze(-1)  # shape: (batch,)
            scores = logits.detach().cpu().tolist()

        # 如果只有一个元素，tolist() 可能返回 float
        if isinstance(scores, float):
            scores = [scores]

        return scores

    def rerank(self, query, candidates, top_k=DEFAULT_TOP_K):
        """
        candidates：来自 multipath 的 fused_results（list of dict）
        需要每条里有 text 或 text_head
        """
        # 取出 passages（尽量用全文 text，没有就用 text_head）
        passages = []
        for c in candidates:
            t = c.get("text", "")
            if not t:
                t = c.get("text_head", "")
            passages.append(t)

        scores = self.score_pairs(query, passages)

        # 把分数写回去
        reranked = []
        for i in range(len(candidates)):
            item = dict(candidates[i])
            item["rerank_score"] = float(scores[i])
            reranked.append(item)

        # 按 rerank_score 从高到低排序
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return reranked[:top_k]


def main():
    print("== Reranker Demo ==")
    print("Rerank model =", RERANK_MODEL)

    # 1) 先跑 multipath 取 fused_results
    retriever = MultiPathRetriever()

    query = input("\nEnter your query: ").strip()
    if not query:
        print("Empty query, exit.")
        return

    fusion = input("Fusion strategy (rrf/simple/weighted, default rrf): ").strip().lower()
    if fusion not in ["rrf", "simple", "weighted"]:
        fusion = "rrf"

    top_n_str = input(f"Top N candidates to rerank (default {DEFAULT_TOP_N}): ").strip()
    top_k_str = input(f"Final Top K after rerank (default {DEFAULT_TOP_K}): ").strip()

    top_n = DEFAULT_TOP_N
    top_k = DEFAULT_TOP_K
    if top_n_str.isdigit():
        top_n = int(top_n_str)
    if top_k_str.isdigit():
        top_k = int(top_k_str)

    # multipath 会内部做 query enhancement
    query_info = retriever  # 只是占位避免误解

    # retriever 里自己会 process_medical_query，所以这里直接用 retrieve 的入口：
    # 注意：MultiPathRetriever.main() 里是交互式的，我们这里直接调用内部逻辑更清晰：
    from task0125.rag_retrieval import process_medical_query
    query_info = process_medical_query(query)

    if not query_info.get("ok"):
        print("Query rejected:", query_info.get("reason"))
        return

    out = retriever.retrieve(
        query_info=query_info,
        top_k_vector=max(5, top_n),
        top_k_keyword=max(5, top_n),
        fusion_strategy=fusion
    )

    fused = out["fused_results"][:top_n]

    print("\n=== Fused candidates (before rerank) ===")
    for i, r in enumerate(fused, start=1):
        print(f"\nRank {i}")
        print("source     =", r.get("source"))
        print("uid        =", r.get("uid"))
        print("doc_id     =", r.get("doc_id"))
        print("text_head  =", shorten_text(r.get("text_head", "")))

    # 2) rerank
    reranker = SimpleReranker(RERANK_MODEL)
    reranked = reranker.rerank(
        query=query_info["clean_query"],   # 用清洗后的 query
        candidates=fused,
        top_k=top_k
    )

    print("\n=== Reranked results (after rerank) ===")
    for i, r in enumerate(reranked, start=1):
        print(f"\nRank {i}")
        print("rerank_score =", round(r.get("rerank_score", 0.0), 6))
        print("source       =", r.get("source"))
        print("uid          =", r.get("uid"))
        print("doc_id       =", r.get("doc_id"))
        print("text_head    =", shorten_text(r.get("text_head", "")))


if __name__ == "__main__":
    main()
