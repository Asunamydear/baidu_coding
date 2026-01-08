from datasets import load_dataset
import pandas as pd

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    use_fast=True
)
## 这个查询之后tokenizer 和deepseek-r1:8b 的内容是一样的
ds = load_dataset("armanc/pubmed-rct20k")
df = ds["train"].to_pandas()

# 保证句子顺序
df = df.sort_values(["abstract_id", "sentence_id"])

# 合并成摘要
abstract_df = (
    df.groupby("abstract_id")["text"]
      .apply(lambda x: " ".join(x))
      .reset_index()
)

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

abstract_df["token_len"] = abstract_df["text"].apply(count_tokens)

print(abstract_df["token_len"].describe())

shortest5 = abstract_df.sort_values("token_len", ascending=True).head(5)
print(shortest5[["abstract_id", "token_len"]])

longest5 = abstract_df.sort_values("token_len", ascending=False).head(5)
print(longest5[["abstract_id", "token_len"]])

median_len = abstract_df["token_len"].median()

median5 = (
    abstract_df
    .assign(dist=lambda d: (d["token_len"] - median_len).abs())
    .sort_values("dist")
    .head(5)
)

print(median5[["abstract_id", "token_len"]])