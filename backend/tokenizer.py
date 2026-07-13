import re
import torch
from collections import Counter


class SimpleTokenizer:
    def __init__(self, min_freq: int = 1, max_vocab_size: int = 30000, max_len: int = 96):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9#+.\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self.clean_text(text).split())

        most_common = [item for item in counter.most_common() if item[1] >= self.min_freq]
        for word, _ in most_common[: self.max_vocab_size - 2]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = None) -> torch.Tensor:
        max_len = max_len or self.max_len
        words = self.clean_text(text).split()
        ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words[:max_len]]
        ids += [self.word2idx["<PAD>"]] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
