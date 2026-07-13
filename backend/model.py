import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderModel(nn.Module):
    """Small custom Transformer encoder for semantic document retrieval."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 96,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        padding_mask = input_ids.eq(0)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling over non-padding tokens
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        embeddings = self.projection(pooled)
        return F.normalize(embeddings, p=2, dim=1)
