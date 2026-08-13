import math

import torch
import torch.nn as nn


class LogAnomalyTransformer(nn.Module):
    """Event-ID sequence -> single anomaly logit.

    forward(x, lengths) contract matches evaluation.validate.infer_scores.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 50,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.max_len}")

        pad_mask = x == self.pad_id
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        h = self.embedding(x) * math.sqrt(self.d_model)
        h = h + self.pos_embedding(positions)
        h = self.dropout(h)
        h = self.transformer(h, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))
