"""
evaluate.py — Full multi-metric evaluation suite for generated molecules.

Implements:
  1. Validity      — % that parse as valid RDKit molecules (should be ~100% with SELFIES)
  2. Uniqueness    — % distinct molecules among candidates
  3. Novelty       — % not in training set (Tanimoto similarity < 0.4 threshold)
  4. Drug-Likeness — Lipinski Rule of 5 compliance (strict)
  5. QED           — Quantitative Estimate of Drug-likeness (RDKit built-in)
  6. REAL Toxicity — ADMET proxy: TPSA + PAINS + logP + MW (NOT length-based!)
  7. Internal Diversity — avg pairwise Tanimoto distance (MOSES metric)
  8. Scaffold Diversity — unique Murcko scaffolds / total valid
  9. Final Score   — composite (drug_score + qed) / 2 − toxicity

CRITICAL FIX vs original notebook:
  - Toxicity is now based on validated ADMET descriptors, not string length
  - Minimum MW/atom filter applied so trivial molecules never top the ranking
"""

import os, json, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, QED,
    FilterCatalog, rdMolDescriptors as rmd,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from tqdm import tqdm
from config import CFG

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------─
# ADMET / Toxicity — REPLACES the naive len(smiles)/50 proxy
# ----------------------------------------------------------------------------─

# Build PAINS filter catalog once
_PAINS_PARAMS = FilterCatalog.FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog.FilterCatalog(_PAINS_PARAMS)
# Brenk structural alerts
_BRENK_PARAMS = FilterCatalog.FilterCatalogParams()
_BRENK_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK_CATALOG = FilterCatalog.FilterCatalog(_BRENK_PARAMS)


def admet_toxicity(mol) -> float:
    """
    Evidence-based ADMET toxicity / undesirability proxy.
    Returns a score in [0, 1] where:
        0.0 = excellent ADMET profile (low toxicity)
        1.0 = poor ADMET profile (high toxicity)

    Components:
      - TPSA penalty  : TPSA > 140 Å² -> poor oral absorption (Veber rule)
      - logP penalty  : logP > 5      -> poor solubility
      - MW penalty    : MW > 500      -> poor absorption (Lipinski)
      - PAINS penalty : Pan-Assay Interference — known false-positive alerts
      - Brenk penalty : Structural alerts for toxic motifs
    """
    tpsa  = rmd.CalcTPSA(mol)
    logp  = Descriptors.MolLogP(mol)
    mw    = Descriptors.MolWt(mol)

    # Continuous penalties normalised to [0,1]
    tpsa_pen  = min(1.0, max(0.0, (tpsa  - 140) / 60))   # 0 if ≤140, 1 if ≥200
    logp_pen  = min(1.0, max(0.0, (logp  -   5) / 3))    # 0 if ≤5,   1 if ≥8
    mw_pen    = min(1.0, max(0.0, (mw    - 500) / 200))  # 0 if ≤500, 1 if ≥700

    pains_pen = 1.0 if _PAINS_CATALOG.HasMatch(mol) else 0.0
    brenk_pen = 0.5 if _BRENK_CATALOG.HasMatch(mol) else 0.0

    score = (
        CFG.TOX_TPSA_WEIGHT  * tpsa_pen  +
        CFG.TOX_LOGP_WEIGHT  * logp_pen  +
        CFG.TOX_MW_WEIGHT    * mw_pen    +
        CFG.TOX_PAINS_WEIGHT * min(1.0, pains_pen + brenk_pen)
    )
    return round(float(score), 4)


# ----------------------------------------------------------------------------─
# Drug-likeness
# ----------------------------------------------------------------------------─

def lipinski_score(mol) -> float:
    """
    Lipinski Rule of 5 compliance score ∈ {0.0, 0.25, 0.50, 0.75, 1.0}.
    Each criterion passed = 0.25 added.
    """
    mw  = Descriptors.MolWt(mol)
    lp  = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return round(
        sum([mw <= 500, lp <= 5, hbd <= 5, hba <= 10]) / 4, 4
    )


# ----------------------------------------------------------------------------─
# Filters
# ----------------------------------------------------------------------------─

def passes_basic_filters(mol) -> bool:
    """
    Hard filters applied BEFORE scoring.
    FIX: minimum MW enforced -> trivial molecules (isobutane etc.) excluded.
    """
    mw          = Descriptors.MolWt(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    logp        = Descriptors.MolLogP(mol)
    tpsa        = rmd.CalcTPSA(mol)
    rot         = rmd.CalcNumRotatableBonds(mol)
    hbd         = Descriptors.NumHDonors(mol)
    hba         = Descriptors.NumHAcceptors(mol)

    return (
        CFG.MIN_MW         <= mw          <= CFG.MAX_MW  and
        heavy_atoms        >= CFG.MIN_HEAVY_ATOMS        and
        logp               <= CFG.MAX_LOGP               and
        hbd                <= CFG.MAX_HBD                and
        hba                <= CFG.MAX_HBA                and
        tpsa               <= CFG.MAX_TPSA               and
        rot                <= CFG.MAX_ROT_BONDS
    )


# ----------------------------------------------------------------------------─
# Tanimoto diversity
# ----------------------------------------------------------------------------─

def morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def internal_diversity(mols: list, sample: int = 200) -> float:
    """
    Average pairwise Tanimoto distance among generated molecules.
    1.0 = maximally diverse, 0.0 = all identical.
    Sampled for speed when > `sample` molecules.
    """
    if len(mols) < 2:
        return 0.0
    fps = [morgan_fp(m) for m in mols]
    if len(fps) > sample:
        rng = np.random.default_rng(CFG.SEED)
        idx = rng.choice(len(fps), sample, replace=False)
        fps = [fps[i] for i in idx]
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return round(float(np.mean(dists)), 4) if dists else 0.0


def novelty_tanimoto(gen_mol, train_fps: list, threshold: float = 0.4) -> bool:
    """
    A generated molecule is novel if its max Tanimoto similarity to any
    training molecule is below `threshold` (i.e. not memorised).
    """
    fp = morgan_fp(gen_mol)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return max(sims) < threshold if sims else True


def scaffold_diversity(mols: list) -> float:
    """Fraction of unique Murcko scaffolds among valid molecules."""
    scaffolds = set()
    for mol in mols:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(sc))
        except Exception:
            pass
    return round(len(scaffolds) / len(mols), 4) if mols else 0.0


# ----------------------------------------------------------------------------─
# Full evaluation pipeline
# ----------------------------------------------------------------------------─

def evaluate(
    smiles_list: list[str],
    train_smiles: list[str],
    label: str = "model",
    save_csv: bool = True,
) -> dict:
    """
    Complete evaluation of a list of generated SMILES strings.

    Args:
        smiles_list  : list of generated SMILES (may include invalids)
        train_smiles : SMILES from training set (for novelty check)
        label        : identifier for saving results (e.g. "temp_0.7")
        save_csv     : whether to save per-molecule CSV

    Returns:
        summary dict with all metrics
    """
    print(f"\n[Evaluator] Evaluating {len(smiles_list):,} candidates (label={label}) ...")

    # -- Pre-compute training fingerprints (sample 5K for speed) ----------
    print("[Evaluator] Building training fingerprints ...")
    train_sample = train_smiles[:5_000]
    train_fps = []
    for smi in tqdm(train_sample, desc="train fps"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_fps.append(morgan_fp(mol))

    # -- Parse + filter ----------------------------------------------------
    rows = []
    valid_mols = []

    for smi in tqdm(smiles_list, desc="evaluating"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue                          # invalid SMILES

        if not passes_basic_filters(mol):
            continue                          # fails MW/atom/ADMET pre-filters

        # Compute all properties
        mw     = round(Descriptors.MolWt(mol), 2)
        logp   = round(Descriptors.MolLogP(mol), 2)
        tpsa   = round(rmd.CalcTPSA(mol), 2)
        hbd    = Descriptors.NumHDonors(mol)
        hba    = Descriptors.NumHAcceptors(mol)
        rot    = rmd.CalcNumRotatableBonds(mol)
        arom   = rmd.CalcNumAromaticRings(mol)
        ha     = mol.GetNumHeavyAtoms()
        qed_sc = round(QED.qed(mol), 4)
        lip    = lipinski_score(mol)
        tox    = admet_toxicity(mol)
        novel  = novelty_tanimoto(mol, train_fps) if train_fps else True
        final  = round((lip + qed_sc) / 2 - tox, 4)

        valid_mols.append(mol)
        rows.append({
            "SMILES"       : smi,
            "MolWeight"    : mw,
            "LogP"         : logp,
            "TPSA"         : tpsa,
            "HBD"          : hbd,
            "HBA"          : hba,
            "RotBonds"     : rot,
            "AromaticRings": arom,
            "HeavyAtoms"   : ha,
            "QED"          : qed_sc,
            "DrugScore"    : lip,
            "Toxicity"     : tox,
            "FinalScore"   : final,
            "IsNovel"      : novel,
            "IsDrugLike"   : lip >= 0.75,
            "PassesPAINS"  : not _PAINS_CATALOG.HasMatch(mol),
        })

    df = pd.DataFrame(rows).drop_duplicates("SMILES").sort_values(
        "FinalScore", ascending=False
    ).reset_index(drop=True)

    # -- Summary metrics --------------------------------------------------─
    n_gen      = len(smiles_list)
    n_valid    = len(rows)             # after RDKit parse (invalids dropped)
    n_filtered = len(df)              # after uniqueness + filter
    validity   = round(n_valid / n_gen, 4)        if n_gen    else 0
    uniqueness = round(len(df["SMILES"].unique()) / n_valid, 4) if n_valid else 0
    novelty    = round(df["IsNovel"].mean(), 4)   if len(df)  else 0
    drug_like  = round(df["IsDrugLike"].mean(), 4)if len(df)  else 0
    avg_qed    = round(df["QED"].mean(), 4)        if len(df)  else 0
    avg_tox    = round(df["Toxicity"].mean(), 4)   if len(df)  else 0
    avg_final  = round(df["FinalScore"].mean(), 4) if len(df)  else 0
    pains_ok   = round(df["PassesPAINS"].mean(), 4)if len(df)  else 0
    int_div    = internal_diversity(valid_mols)
    scaf_div   = scaffold_diversity(valid_mols)

    summary = {
        "label"            : label,
        "n_generated"      : n_gen,
        "n_valid"          : n_valid,
        "validity_%"       : f"{validity*100:.1f}",
        "uniqueness_%"     : f"{uniqueness*100:.1f}",
        "novelty_%"        : f"{novelty*100:.1f}",
        "drug_like_%"      : f"{drug_like*100:.1f}",
        "pains_clean_%"    : f"{pains_ok*100:.1f}",
        "avg_QED"          : avg_qed,
        "avg_toxicity"     : avg_tox,
        "avg_final_score"  : avg_final,
        "internal_diversity": int_div,
        "scaffold_diversity": scaf_div,
    }

    print("\n-- Evaluation Summary ------------------------------")
    for k, v in summary.items():
        print(f"   {k:<25}: {v}")
    print("----------------------------------------------------")

    if save_csv and len(df) > 0:
        out_path = os.path.join(CFG.RESULTS_DIR, f"molecules_{label}.csv")
        df.to_csv(out_path, index=False)
        print(f"[Evaluator] Per-molecule results -> {out_path}")

    # Save summary
    summ_path = os.path.join(CFG.RESULTS_DIR, f"summary_{label}.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {"summary": summary, "df": df}


# ----------------------------------------------------------------------------─
# MOSES-style benchmark table
# ----------------------------------------------------------------------------─

def build_benchmark_table(results: list[dict]) -> pd.DataFrame:
    """
    Combine multiple evaluation summaries into a single comparison table
    (e.g. model at different temperatures vs. random baseline).
    """
    rows = [r["summary"] for r in results]
    df = pd.DataFrame(rows).set_index("label")
    return df
