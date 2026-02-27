import json
import random
from transformers import AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MAX_TOKENS = 512
CHUNK_OVERLAP = 80
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    use_fast=True
)

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

# 1) 读 chunks.jsonl
chunks = []
with open(BASE_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print("Total chunks =", len(chunks))

# 2) 检查是否有超限 chunk
over_limit = []
for c in chunks:
    tlen = c.get("token_count")
    if tlen is None:
        tlen = count_tokens(c["text"])
    if tlen > MAX_TOKENS:
        over_limit.append((c["chunk_id"], tlen))

print(">512 chunks =", len(over_limit))
if len(over_limit) > 0:
    print("Examples of >512:")
    for x in over_limit[:5]:
        print(x)

# 3) 抽样检查文本质量（随机 10 块）
print("\n===== Random 10 chunk samples =====")
samples = random.sample(chunks, 10)
for c in samples:
    text = c["text"].replace("\n", " ")
    print("\n---", c["chunk_id"], "| tokens =", c.get("token_count", count_tokens(c["text"])))
    print(text[:300])  # 只看前 300 字符

# 4) 找出多块文献（total_chunks > 1），抽 3 篇重点检查
multi_docs = {}
for c in chunks:
    if c["total_chunks"] > 1:
        doc_id = c["doc_id"]
        if doc_id not in multi_docs:
            multi_docs[doc_id] = []
        multi_docs[doc_id].append(c)

print("\nMulti-chunk docs =", len(multi_docs))

if len(multi_docs) > 0:
    picked_ids = random.sample(list(multi_docs.keys()), min(3, len(multi_docs)))
    for doc_id in picked_ids:
        print("\n===============================")
        print("DOC_ID:", doc_id, "| total_chunks =", len(multi_docs[doc_id]))
        print("===============================")

        # 按 chunk_index 排序
        doc_chunks = sorted(multi_docs[doc_id], key=lambda x: x["chunk_index"])

        # 打印一下
        for c in doc_chunks:
            text = c["text"].replace("\n", " ")
            print("\n[chunk", c["chunk_index"], "] tokens =", c.get("token_count", count_tokens(c["text"])))
            print("HEAD:", text[:200])
            print("TAIL:", text[-200:])



        # ===== overlap 检查（token 级，检查相邻 chunk）=====
        if len(doc_chunks) >= 2:
            tokens0 = tokenizer.encode(doc_chunks[0]["text"], add_special_tokens=False)
            tokens1 = tokenizer.encode(doc_chunks[1]["text"], add_special_tokens=False)

            overlap0 = tokens0[-CHUNK_OVERLAP:]
            overlap1 = tokens1[:CHUNK_OVERLAP]

            exact_match = (overlap0 == overlap1)#从第一个token的80前80个，比较到后80个
            common = set(overlap0) & set(overlap1)#看看有多少tokenid 重复

            print("\nOverlap check:")
            print("Exact overlap match:", exact_match)
            print("Common overlap tokens count:", len(common))

