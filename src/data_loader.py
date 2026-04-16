"""
data_loader.py — Load ZINC250K, convert SMILES -> SELFIES, build vocabulary.

KEY FIX vs original notebook:
  - Uses SELFIES encoding -> ANY token sequence decodes to a valid molecule
    (solves the 7.3% validity problem at its root)
  - Proper train/test split stored for novelty evaluation later
"""

import os, random, pickle
import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
from config import CFG


# ----------------------------------------------------------------------------─
# Helpers
# ----------------------------------------------------------------------------─

def smiles_to_selfies(smiles: str) -> str | None:
    """Convert a SMILES string to SELFIES; return None if conversion fails."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            clean_smi = Chem.MolToSmiles(mol)
            return sf.encoder(clean_smi)
        return None
    except Exception:
        return None


def tokenise_selfies(selfies_str: str) -> list[str]:
    """Split a SELFIES string into its constituent tokens, e.g. [C][=C][N]."""
    return list(sf.split_selfies(selfies_str))


def pad_tokens(tokens: list[str], max_len: int, pad: str) -> list[str]:
    """Truncate or right-pad a token list to exactly max_len."""
    tokens = tokens[:max_len]
    tokens += [pad] * (max_len - len(tokens))
    return tokens


# ----------------------------------------------------------------------------─
# Main loader
# ----------------------------------------------------------------------------─

def load_and_prepare(
    data_path: str  = CFG.DATA_PATH,
    max_molecules   = CFG.MAX_MOLECULES,
    max_len: int    = CFG.MAX_LEN,
    train_split     = CFG.TRAIN_SPLIT,
    seed: int       = CFG.SEED,
    cache_dir: str  = CFG.RESULTS_DIR,
) -> dict:
    """
    Full data pipeline:
      SMILES -> SELFIES -> tokenise -> pad -> integer-encode -> train/test split.

    Returns a dict with keys:
      X_train, X_test  (np arrays, shape [N, max_len-1])
      y_train, y_test  (np arrays, shape [N, max_len-1])
      vocab, token2idx, idx2token
      train_smiles, test_smiles  (raw SMILES for novelty evaluation)
    """
    cache_file = os.path.join(cache_dir, "data_cache.pkl")

    if os.path.exists(cache_file):
        print("[DataLoader] Loading from cache ...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("[DataLoader] Reading CSV ...")
    df = pd.read_csv(data_path)
    smiles_col = CFG.SMILES_COL
    df = df[[smiles_col]].dropna()

    if max_molecules:
        df = df.sample(min(max_molecules, len(df)), random_state=seed)

    smiles_list = df[smiles_col].tolist()
    print(f"[DataLoader] {len(smiles_list):,} SMILES loaded.")

    # -- Convert to SELFIES ------------------------------------------------
    print("[DataLoader] Converting SMILES -> SELFIES ...")
    selfies_list, valid_smiles = [], []
    for smi in tqdm(smiles_list):
        sel = smiles_to_selfies(smi)
        if sel:
            selfies_list.append(sel)
            valid_smiles.append(smi)

    print(f"[DataLoader] {len(selfies_list):,} valid SELFIES obtained.")

    # -- Build vocabulary --------------------------------------------------
    all_tokens: set[str] = set()
    for sel in selfies_list:
        all_tokens.update(tokenise_selfies(sel))
    vocab = sorted(all_tokens) + [CFG.PAD_TOKEN]
    token2idx = {t: i for i, t in enumerate(vocab)}
    idx2token = {i: t for t, i in token2idx.items()}
    pad_idx = token2idx[CFG.PAD_TOKEN]
    print(f"[DataLoader] Vocabulary size: {len(vocab)}")

    # -- Encode sequences --------------------------------------------------
    print("[DataLoader] Encoding sequences ...")
    X_list, y_list = [], []
    for sel in tqdm(selfies_list):
        tokens = tokenise_selfies(sel)
        padded = pad_tokens(tokens, max_len, CFG.PAD_TOKEN)
        ids    = [token2idx[t] for t in padded]
        X_list.append(ids[:-1])   # input: all tokens except last
        y_list.append(ids[1:])    # target: all tokens except first (next-token)

    X = np.array(X_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)

    # -- Train / test split ------------------------------------------------
    np.random.seed(seed)
    idx       = np.random.permutation(len(X))
    n_train   = int(len(X) * train_split)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    data = dict(
        X_train      = X[train_idx],
        y_train      = y[train_idx],
        X_test       = X[test_idx],
        y_test       = y[test_idx],
        vocab        = vocab,
        token2idx    = token2idx,
        idx2token    = idx2token,
        pad_idx      = pad_idx,
        train_smiles = [valid_smiles[i] for i in train_idx],
        test_smiles  = [valid_smiles[i] for i in test_idx],
        max_len      = max_len,
    )

    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print("[DataLoader] Cached to disk.")
    return data


if __name__ == "__main__":
    d = load_and_prepare()
    print("X_train shape:", d["X_train"].shape)
    print("Vocab size:   ", len(d["vocab"]))
    print("Sample SELFIES token sequence:", d["idx2token"])
