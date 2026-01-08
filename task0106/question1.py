# from datasets import load_dataset
#
# ds = load_dataset("armanc/pubmed-rct20k")
# print(ds["train"].column_names)

import pandas as pd
from datasets import load_dataset

ds = load_dataset("armanc/pubmed-rct20k")
df = ds["train"].to_pandas()
print(ds["train"].features)

missing_count = df["text"].isna().sum()
total_count = len(df)
missing_rate = missing_count / total_count

print("text 缺失数量:", missing_count)
print("text 缺失率:", missing_rate)
for i in range(20):
    print(ds["train"][i])
# 查看字段
# print(df.columns)

df["text_len"] = df["text"].str.len()

short_text_count = (df["text_len"] < 20).sum()
short_text_ratio = short_text_count / len(df)

print("极短文本数量:", short_text_count)
print("极短文本比例:", short_text_ratio)
