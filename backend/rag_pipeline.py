import os
import pickle
from pathlib import Path

import numpy as np
import torch

from model import TransformerEncoderModel

try:
    import faiss
except Exception:
    faiss = None


class RAGPipeline:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = Path(artifact_dir)
        tokenizer_path = self.artifact_dir / "tokenizer.pkl"
        model_path = self.artifact_dir / "encoder_model.pth"

        if not tokenizer_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run: python train_encoder.py --epochs 18"
            )

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerEncoderModel(
            vocab_size=len(self.tokenizer.word2idx),
            max_len=self.tokenizer.max_len,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.chunks = []
        self.embeddings = None
        self.index = None

    def get_embedding(self, text: str):
        input_ids = self.tokenizer.encode(text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(input_ids).cpu().numpy().astype("float32")
        return emb[0]

    def add_documents(self, chunks):
        self.chunks = chunks
        if not chunks:
            self.embeddings = np.empty((0, 128), dtype="float32")
            self.index = None
            return

        self.embeddings = np.vstack([self.get_embedding(chunk["text"]) for chunk in chunks]).astype("float32")

        if faiss is not None:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None

    def retrieve(self, query, top_k=4):
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = self.get_embedding(query).astype("float32").reshape(1, -1)
        top_k = min(top_k, len(self.chunks))

        if self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            return [(float(scores[0][i]), self.chunks[int(indices[0][i])]) for i in range(top_k)]

        # fallback if faiss-cpu installation is unavailable
        scores = np.dot(self.embeddings, query_embedding[0])
        indices = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.chunks[int(i)]) for i in indices]
