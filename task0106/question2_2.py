from datasets import load_dataset


ds = load_dataset("armanc/pubmed-rct20k")
df = ds["train"].to_pandas()


abstract_ids = [
    "25752109", "24473376", "25407377", "24387919", "24636143",  # 最短
    "25130995", "26144908", "25481791", "25795409", "24717919",  # 最长
    "25679343", "24844551", "24655865", "26016823", "25965710"   # 中位数
]


for aid in abstract_ids:
    print("\n" + "=" * 80)
    print(f"ABSTRACT ID: {aid}")
    print("=" * 80)

    abstract = (
        df[df["abstract_id"] == aid]
        .sort_values("sentence_id")
    )

    for _, row in abstract.iterrows():
        print(f"[{row['sentence_id']}] {row['label']}: {row['text']}")
