import pandas as pd
pd.set_option("display.max_columns", None)
splits = {
    "train": "train.jsonl",
    "validation": "dev.jsonl",
    "test": "test.jsonl"
}

df = pd.read_json(
    "hf://datasets/armanc/pubmed-rct20k/" + splits["train"],
    lines=True
)
print(df.head())
print(df[["abstract_id", "label", "text", "sentence_id"]].head())

