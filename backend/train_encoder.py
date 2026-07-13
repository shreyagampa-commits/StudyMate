import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer
from model import TransformerEncoderModel


class TripletTextDataset(Dataset):
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        row = self.triplets[idx]
        return (
            self.tokenizer.encode(row["anchor"]),
            self.tokenizer.encode(row["positive"]),
            self.tokenizer.encode(row["negative"]),
        )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_retrieval(model, tokenizer, triplets, device):
    model.eval()
    correct = 0
    distances = []

    with torch.no_grad():
        for row in triplets:
            a = tokenizer.encode(row["anchor"]).unsqueeze(0).to(device)
            p = tokenizer.encode(row["positive"]).unsqueeze(0).to(device)
            n = tokenizer.encode(row["negative"]).unsqueeze(0).to(device)

            ae, pe, ne = model(a), model(p), model(n)

            pos_sim = torch.cosine_similarity(ae, pe).item()
            neg_sim = torch.cosine_similarity(ae, ne).item()

            correct += int(pos_sim > neg_sim)
            distances.append(pos_sim - neg_sim)

    return {
        "pairwise_accuracy": round(correct / max(1, len(triplets)), 4),
        "avg_similarity_margin": round(float(np.mean(distances)), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train custom Transformer encoder for StudyMate AI retrieval"
    )
    parser.add_argument("--data", default="data/training_triplets.json")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    print("=" * 70)
    print("STUDYMATE AI - TRANSFORMER ENCODER TRAINING")
    print("=" * 70)

    set_seed(42)
    print("[1] Random seed set to 42")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    print(f"[2] Artifact folder ready: {out_dir}")

    print(f"[3] Loading dataset from: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        triplets = json.load(f)

    print("=" * 70)
    print("DATASET LOADED")
    print(f"Total Triplets: {len(triplets)}")
    print("Sample Training Row:")
    print(json.dumps(triplets[0], indent=2) if triplets else "No data found")
    print("=" * 70)

    random.shuffle(triplets)
    split = int(0.85 * len(triplets))
    train_rows = triplets[:split]
    val_rows = triplets[split:]

    print("[4] Dataset split completed")
    print(f"Training samples: {len(train_rows)}")
    print(f"Validation samples: {len(val_rows)}")

    print("[5] Building tokenizer vocabulary...")
    all_texts = []
    for row in train_rows:
        all_texts.extend([row["anchor"], row["positive"], row["negative"]])

    tokenizer = SimpleTokenizer(
        min_freq=1,
        max_vocab_size=30000,
        max_len=args.max_len
    )
    tokenizer.build_vocab(all_texts)

    print("=" * 70)
    print("TOKENIZER BUILT")
    print(f"Vocabulary Size: {len(tokenizer.word2idx)}")
    print("Sample Tokens:")
    print(list(tokenizer.word2idx.items())[:20])
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[6] Device selected: {device}")

    print("[7] Creating Transformer Encoder model...")
    model = TransformerEncoderModel(
        vocab_size=len(tokenizer.word2idx),
        max_len=args.max_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 70)
    print("MODEL CREATED")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Max Sequence Length: {args.max_len}")
    print(f"Running On: {device}")
    print("=" * 70)

    train_loader = DataLoader(
        TripletTextDataset(train_rows, tokenizer),
        batch_size=args.batch_size,
        shuffle=True
    )

    criterion = nn.TripletMarginLoss(margin=0.35)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("TRAINING CONFIGURATION")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Loss Function: TripletMarginLoss")
    print(f"Optimizer: AdamW")
    print(f"Number of Batches per Epoch: {len(train_loader)}")
    print("=" * 70)

    best_acc = -1
    history = []

    print("STARTING TRAINING...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} started")

        model.train()
        total_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(train_loader, start=1):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(train_loader):
                print(
                    f"  Batch {batch_idx}/{len(train_loader)} "
                    f"| Current Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(1, len(train_loader))

        print("  Running validation...")
        metrics = evaluate_retrieval(model, tokenizer, val_rows, device)

        history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            **metrics
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} completed "
            f"| Avg Loss: {avg_loss:.4f} "
            f"| Val Accuracy: {metrics['pairwise_accuracy']:.4f} "
            f"| Similarity Margin: {metrics['avg_similarity_margin']:.4f}"
        )

        if metrics["pairwise_accuracy"] > best_acc:
            best_acc = metrics["pairwise_accuracy"]

            torch.save(model.state_dict(), out_dir / "encoder_model.pth")
            with open(out_dir / "tokenizer.pkl", "wb") as f:
                pickle.dump(tokenizer, f)

            print("  Best model updated and saved")

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print("Saved best model to artifacts/encoder_model.pth")
    print("Saved tokenizer to artifacts/tokenizer.pkl")
    print("Saved training history to artifacts/training_history.json")
    print("=" * 70)


if __name__ == "__main__":
    main()