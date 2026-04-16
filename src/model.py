"""
model.py — Stacked LSTM model (PyTorch) for SELFIES token generation.

Improvements vs original notebook:
  - PyTorch (works on Python 3.14; TensorFlow requires ≤3.12)
  - 2 stacked LSTM layers (256 units each) vs 1 × 128
  - Dropout + recurrent dropout for regularisation
  - Clean save/load interface
"""

import os
import torch
import torch.nn as nn
from config import CFG


class SELFIESLSTMModel(nn.Module):
    """
    Character-level LSTM for SELFIES token generation.

    Architecture:
      Embedding -> LSTM-1 -> Dropout -> LSTM-2 -> Dropout -> Linear (logits)
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, CFG.EMBED_DIM, padding_idx=0)
        self.lstm   = nn.LSTM(
            CFG.EMBED_DIM, CFG.LSTM_UNITS_1,
            num_layers=2,
            batch_first=True,
            dropout=CFG.DROPOUT,
        )
        self.drop   = nn.Dropout(CFG.DROPOUT)
        self.linear = nn.Linear(CFG.LSTM_UNITS_1, vocab_size)

    def forward(self, x, hidden1=None, hidden2=None):
        """
        Args:
            x       : (batch, seq_len) int64 token ids
            hidden1 : optional LSTM-1 hidden state tuple
            hidden2 : optional LSTM-2 hidden state tuple
        Returns:
            logits  : (batch, seq_len, vocab_size) float32
            hidden1, hidden2 : updated hidden states
        """
        emb = self.embed(x)                         # (B, T, E)
        out, hidden = self.lstm(emb, hidden1)
        out  = self.drop(out)
        logits = self.linear(out)                   # (B, T, V)
        return logits, hidden, None


def build_model(vocab_size: int) -> SELFIESLSTMModel:
    torch.manual_seed(CFG.SEED)
    model = SELFIESLSTMModel(vocab_size)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def save_model(model: SELFIESLSTMModel, path: str):
    torch.save(model.state_dict(), path)
    print(f"[Model] Saved -> {path}")


def load_model(vocab_size: int, path: str) -> SELFIESLSTMModel:
    model = build_model(vocab_size)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[Model] Loaded weights from {path}")
    return model


if __name__ == "__main__":
    m = build_model(100)
    print(m)
