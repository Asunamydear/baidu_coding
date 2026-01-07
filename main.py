from datasets import load_dataset
dataset = load_dataset("armanc/pubmed-rct20k")
# print(dataset)
print(dataset["train"])
print(dataset["train"][0])