from datasets import load_dataset
from transformers import AutoTokenizer
import json


tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    use_fast=True
)


ds = load_dataset("armanc/pubmed-rct20k")
df = ds["train"].to_pandas()


df = df.sort_values(["abstract_id", "sentence_id"])

docs = []
grouped = df.groupby("abstract_id")["text"]
for abstract_id, sentences in grouped:
    full_text = " ".join(list(sentences))
    docs.append({
        "doc_id": str(abstract_id),
        "text": full_text
    })

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_by_tokens(text, chunk_size, chunk_overlap):
    ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    while start < len(ids):
        end = start + chunk_size
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)

        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= len(ids):
            break

    return chunks

# params
MAX_TOKENS = 512
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

all_chunks = []

# stats
total_docs = len(docs)
split_docs = 0
split_chunk_counts = []

for doc in docs:
    doc_id = doc["doc_id"]
    full_text = doc["text"]
    tok_len = count_tokens(full_text)

    # <=512: no split
    if tok_len <= MAX_TOKENS:
        chunk_data = {
            "chunk_id": doc_id,
            "text": full_text,
            "doc_id": doc_id,
            "chunk_index": 0,
            "total_chunks": 1,
            "source_title": "",
            "token_count": tok_len
        }
        all_chunks.append(chunk_data)

    # >512: split
    else:
        split_docs += 1
        texts = split_by_tokens(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        total = len(texts)
        split_chunk_counts.append(total)

        for i in range(total):
            t = texts[i]
            chunk_id = doc_id + "_" + str(i)

            chunk_data = {
                "chunk_id": chunk_id,
                "text": t,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": total,
                "source_title": "",
                "token_count": count_tokens(t)
            }
            all_chunks.append(chunk_data)


out_path = "chunks.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for item in all_chunks:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print stats
print("done. saved to:", out_path)
print("total docs =", total_docs)
print("docs split (>512) =", split_docs)
if split_docs > 0:
    avg_chunks = sum(split_chunk_counts) / len(split_chunk_counts)
    print("avg chunks per split doc =", round(avg_chunks, 2))
print("total chunks written =", len(all_chunks))
