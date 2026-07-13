from datasets import load_dataset
import json
import random
import os

print("Downloading SQuAD v2...")

dataset = load_dataset("rajpurkar/squad_v2", split="train[:20000]")

os.makedirs("data", exist_ok=True)

triplets = []
contexts = [row["context"] for row in dataset]

print("Creating triplets...")

for row in dataset:
    question = row["question"]
    positive = row["context"]

    negative = random.choice(contexts)
    while negative == positive:
        negative = random.choice(contexts)

    triplets.append({
        "anchor": question,
        "positive": positive,
        "negative": negative
    })

print("Triplets created:", len(triplets))

with open("data/training_triplets.json", "w", encoding="utf-8") as f:
    json.dump(triplets, f, indent=2)

print("Saved to data/training_triplets.json")