"""
train.py — Train the SELFIES LSTM (PyTorch) with proper callbacks & loss logging.

Saves:
  models/selfies_lstm_best.pt   — best model weights
  results/training_history.csv  — per-epoch loss/acc for the loss-curve figure
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from config import CFG
from data_loader import load_and_prepare
from model import build_model, save_model


def train(data: dict | None = None) -> dict:
    """Train and return history dict."""
    if data is None:
        data = load_and_prepare()

    X     = torch.tensor(data["X_train"], dtype=torch.long)
    y     = torch.tensor(data["y_train"], dtype=torch.long)
    vocab = data["vocab"]

    print(f"[Trainer] Samples: {len(X):,}  |  Vocab: {len(vocab)}")

    # ── Dataset & loaders ─────────────────────────────────────────────────
    dataset   = TensorDataset(X, y)
    n_val     = int(len(dataset) * 0.10)
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(CFG.SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Device: {device}")
    model  = build_model(len(vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=CFG.LR_PATIENCE
    )
    criterion = nn.CrossEntropyLoss(ignore_index=data["pad_idx"])

    best_val_loss = float("inf")
    patience_cnt  = 0
    weights_path  = os.path.join(CFG.MODEL_DIR, "selfies_lstm_best.pt")
    history       = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    print(f"[Trainer] Training for up to {CFG.EPOCHS} epochs …")
    for epoch in range(1, CFG.EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(xb)              # (B, T, V)
            # Reshape for loss: (B*T, V) vs (B*T,)
            loss = criterion(logits.reshape(-1, len(vocab)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(-1)
            mask  = yb != data["pad_idx"]
            total_correct += (preds[mask] == yb[mask]).sum().item()
            total_tokens  += mask.sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = total_correct / max(total_tokens, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_tokens = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _, _ = model(xb)
                loss  = criterion(logits.reshape(-1, len(vocab)), yb.reshape(-1))
                v_loss += loss.item() * xb.size(0)
                preds  = logits.argmax(-1)
                mask   = yb != data["pad_idx"]
                v_correct += (preds[mask] == yb[mask]).sum().item()
                v_tokens  += mask.sum().item()

        val_loss = v_loss / len(val_loader.dataset)
        val_acc  = v_correct / max(v_tokens, 1)

        scheduler.step(val_loss)

        history["loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["accuracy"].append(round(train_acc, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:03d}/{CFG.EPOCHS} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | lr={lr_now:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, weights_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CFG.EARLY_STOP:
                print(f"[Trainer] Early stopping at epoch {epoch}.")
                break

    # ── Save history ──────────────────────────────────────────────────────
    hist_df   = pd.DataFrame(history)
    hist_path = os.path.join(CFG.RESULTS_DIR, "training_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[Trainer] History → {hist_path}")

    # Save vocab reference
    vocab_path = os.path.join(CFG.MODEL_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"vocab": data["vocab"], "max_len": data["max_len"]}, f)
    print(f"[Trainer] Vocab  → {vocab_path}")

    return history


if __name__ == "__main__":
    train()
