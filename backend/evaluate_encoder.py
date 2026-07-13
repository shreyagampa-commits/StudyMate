import json
import pickle
import torch
from pathlib import Path
from model import TransformerEncoderModel


def sim(a, b):
    return torch.cosine_similarity(a, b).item()


def main():
    artifact_dir = Path("artifacts")
    with open(artifact_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = TransformerEncoderModel(len(tokenizer.word2idx), max_len=tokenizer.max_len)
    model.load_state_dict(torch.load(artifact_dir / "encoder_model.pth", map_location="cpu"))
    model.eval()

    with open("data/training_triplets.json", "r", encoding="utf-8") as f:
        rows = json.load(f)

    correct = 0
    with torch.no_grad():
        for row in rows:
            a = model(tokenizer.encode(row["anchor"]).unsqueeze(0))
            p = model(tokenizer.encode(row["positive"]).unsqueeze(0))
            n = model(tokenizer.encode(row["negative"]).unsqueeze(0))
            correct += int(sim(a, p) > sim(a, n))

    print(f"Retrieval pairwise accuracy: {correct / len(rows):.4f}")

    examples = [
        "What is RAG?",
        "Why are indexes used in database systems?",
        "How does TCP provide reliability?",
    ]
    candidates = list({row["positive"] for row in rows})
    cand_embs = []
    with torch.no_grad():
        for text in candidates:
            cand_embs.append(model(tokenizer.encode(text).unsqueeze(0)))

    for q in examples:
        with torch.no_grad():
            q_emb = model(tokenizer.encode(q).unsqueeze(0))
        scores = [(sim(q_emb, emb), text) for emb, text in zip(cand_embs, candidates)]
        scores.sort(reverse=True, key=lambda x: x[0])
        print("\nQuestion:", q)
        print("Top match:", scores[0][1])
        print("Score:", round(scores[0][0], 4))


if __name__ == "__main__":
    main()
