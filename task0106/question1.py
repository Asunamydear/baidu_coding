# from datasets import load_dataset
#
# ds = load_dataset("armanc/pubmed-rct20k")
# print(ds["train"].column_names)

import pandas as pd
from datasets import load_dataset

ds = load_dataset("armanc/pubmed-rct20k")
df = ds["train"].to_pandas()
print(ds["train"].features)
for i in range(20):
    print(ds["train"][i])
# 查看字段
# print(df.columns)
