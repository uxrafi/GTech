"""1-D CNN sequence classifier for HDFS anomaly detection.

architecture:  embedding -> [conv1d + relu + maxpool] x 3 -> fc -> sigmoid
consumes the same tokenized sequences as the LSTM (X_*.npy / len_*.npy / y_*.npy)
so results are directly comparable across models.
"""
import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 5, 7),
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x)          # [B, T, E]
        emb = emb.permute(0, 2, 1)       # [B, E, T]

        pooled = []
        for conv in self.convs:
            out = conv(emb)               # [B, F, T]
            out = out.max(dim=2).values   # [B, F]
            pooled.append(out)

        cat = torch.cat(pooled, dim=1)    # [B, F * n_kernels]
        cat = self.dropout(cat)
        return self.fc(cat).squeeze(1)    # [B] logits