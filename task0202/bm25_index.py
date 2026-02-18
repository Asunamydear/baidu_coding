# task0202/bm25_index.py
# 作用：读取 ../task0118/chunks.jsonl，建立 BM25 关键词索引，并做 top_k 检索

import os
import re
import json
import math

# ====== 路径：task0202 和 task0118 同级 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../task0202
TASK0118_DIR = os.path.join(BASE_DIR, "..", "task0118")
CHUNKS_FILE = os.path.join(TASK0118_DIR, "chunks.jsonl")

# ====== BM25 参数 ======
K1 = 1.5
B = 0.75


def tokenize(text):
    """把文本切成小写 token（字母数字）"""
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(path):
    """读取 chunks.jsonl"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class BM25Index:
    def __init__(self, k1=K1, b=B):
        self.k1 = k1
        self.b = b

        self.docs = []        # 原始 chunk dict 列表
        self.tfs = []         # 每个文档的词频 dict
        self.dl = []          # 每个文档的长度（token 数）
        self.df = {}          # term -> 出现过的文档数
        self.idf = {}         # term -> idf

        self.N = 0
        self.avgdl = 0.0

    def build_index(self, jsonl_path):
        """构建 BM25 索引"""
        self.docs = load_chunks(jsonl_path)
        self.N = len(self.docs)
        if self.N == 0:
            print("No chunks found in", jsonl_path)
            return

        total_len = 0
        self.tfs = []
        self.dl = []
        self.df = {}
        self.idf = {}

        # 1) 统计 tf / df / 文档长度
        for d in self.docs:
            tokens = tokenize(d.get("text", ""))
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            self.tfs.append(tf)
            doc_len = len(tokens)
            self.dl.append(doc_len)
            total_len += doc_len

            # df：同一个 term 在同一个文档只算一次
            for term in tf.keys():
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_len / self.N

        # 2) 计算 idf（加平滑）
        for term, df_val in self.df.items():
            self.idf[term] = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

        print("BM25 index built.")
        print("Total docs =", self.N)
        print("Average doc length =", round(self.avgdl, 2))
        print("Vocab size =", len(self.df))

    def score_one_doc(self, query_terms, i):
        """计算 query 对第 i 个文档的 BM25 分数"""
        score = 0.0
        tf = self.tfs[i]
        dl = self.dl[i]

        for term in query_terms:
            if term not in tf:
                continue

            freq = tf[term]
            idf = self.idf.get(term, 0.0)

            denom = freq + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-12)))
            score += idf * (freq * (self.k1 + 1)) / (denom + 1e-12)

        return score

    def search(self, query, top_k=5):
        """BM25 检索：返回 top_k 结果"""
        query = (query or "").strip()
        if not query:
            return []

        # 用 set 去重，避免 query 某词重复影响太大
        query_terms = list(set(tokenize(query)))
        if not query_terms:
            return []

        scored = []
        for i in range(self.N):
            s = self.score_one_doc(query_terms, i)
            if s > 0:
                scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        results = []
        for idx, s in scored:
            d = self.docs[idx]
            text = d.get("text", "")
            text_head = " ".join(text.split())[:220]
            results.append({
                "chunk_id": str(d.get("chunk_id", "")),
                "doc_id": str(d.get("doc_id", "")),
                "chunk_index": d.get("chunk_index", None),
                "total_chunks": d.get("total_chunks", None),
                "token_count": d.get("token_count", None),
                "bm25_score": float(s),
                "text_head": text_head
            })
        return results


def main():
    print("== BM25 Build & Search ==")
    print("Reading:", CHUNKS_FILE)

    bm25 = BM25Index()
    bm25.build_index(CHUNKS_FILE)

    while True:
        q = input("\nInput query (or type 'q' to quit): ").strip()
        if q.lower() == "q":
            break

        top_k_str = input("Top K (default 5): ").strip()
        if top_k_str == "":
            top_k = 5
        else:
            top_k = int(top_k_str)

        hits = bm25.search(q, top_k=top_k)

        print("\nTop results:")
        for rank, h in enumerate(hits, 1):
            print("\nRank", rank)
            print("bm25_score =", round(h["bm25_score"], 4))
            print("chunk_id   =", h["chunk_id"])
            print("doc_id     =", h["doc_id"])
            print("meta       =", {
                "chunk_index": h["chunk_index"],
                "total_chunks": h["total_chunks"],
                "token_count": h["token_count"],
            })
            print("text_head  =", h["text_head"])


if __name__ == "__main__":
    main()
