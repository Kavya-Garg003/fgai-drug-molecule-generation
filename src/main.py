"""
main.py — Full pipeline orchestrator.

Run this single file to reproduce all results:
    python main.py

Stages:
  1. Data loading & SELFIES preprocessing
  2. Model training (or load if already trained)
  3. Temperature sweep generation
  4. Random baseline generation
  5. Full evaluation (model + baseline)
  6. All publication figures
  7. Summary CSV & benchmark table
"""

import os, json, argparse
import pandas as pd
from config import CFG


def parse_args():
    p = argparse.ArgumentParser(description="FGAI Drug Molecule Generation Pipeline")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training and load existing weights")
    p.add_argument("--quick",       action="store_true",
                   help="Use reduced dataset (5K molecules, 5 epochs) for quick test")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        CFG.MAX_MOLECULES = 5_000
        CFG.EPOCHS        = 5
        CFG.N_GENERATE    = 200
        CFG.TEMPERATURES  = [0.5, 0.7]
        print("[Main] QUICK MODE: reduced dataset + epochs for testing.")

    # -- 1. Load & preprocess data ----------------------------------------─
    print("\n" + "="*60)
    print("STAGE 1: Data Loading & SELFIES Preprocessing")
    print("="*60)
    from data_loader import load_and_prepare
    data = load_and_prepare()
    vocab     = data["vocab"]
    token2idx = data["token2idx"]
    idx2token = data["idx2token"]
    pad_idx   = data["pad_idx"]

    # -- 2. Build / train model --------------------------------------------
    print("\n" + "="*60)
    print("STAGE 2: Model Training")
    print("="*60)
    from model import build_model, load_model

    weights_path = os.path.join(CFG.MODEL_DIR, "selfies_lstm_best.pt")

    if args.skip_train and os.path.exists(weights_path):
        print(f"[Main] Loading existing weights from {weights_path}")
        model = load_model(len(vocab), weights_path)
        history_csv = os.path.join(CFG.RESULTS_DIR, "training_history.csv")
        if not os.path.exists(history_csv):
            # Create a stub history if file missing
            pd.DataFrame({"loss":[0], "val_loss":[0]}).to_csv(history_csv, index=False)
    else:
        from train import train
        history = train(data)
        model   = load_model(len(vocab), weights_path)
        history_csv = os.path.join(CFG.RESULTS_DIR, "training_history.csv")

    # -- 3. Generate molecules at each temperature ------------------------─
    print("\n" + "="*60)
    print("STAGE 3: Molecule Generation (Temperature Sweep)")
    print("="*60)
    from generate import generate_sweep, selfies_to_smiles_list, random_baseline

    sweep_smiles = generate_sweep(
        model, token2idx, idx2token, pad_idx,
        temperatures = CFG.TEMPERATURES,
        n_per_temp   = CFG.N_GENERATE,
    )

    # -- 4. Random baseline ------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 4: Random Baseline Generation")
    print("="*60)
    baseline_smiles = random_baseline(data["train_smiles"], n=CFG.N_GENERATE)
    baseline_path   = os.path.join(CFG.RESULTS_DIR, "generated_baseline.txt")
    with open(baseline_path, "w") as f:
        f.write("\n".join(baseline_smiles))

    # -- 5. Evaluate ------------------------------------------------------─
    print("\n" + "="*60)
    print("STAGE 5: Evaluation")
    print("="*60)
    from evaluate import evaluate, build_benchmark_table

    all_results   = []
    temp_summaries= []

    for temp, smiles in sweep_smiles.items():
        res = evaluate(
            smiles_list  = smiles,
            train_smiles = data["train_smiles"],
            label        = f"T{temp}",
            save_csv     = True,
        )
        all_results.append(res)
        s = res["summary"]
        s["label"] = f"T{temp}"
        temp_summaries.append(s)

    # Best temperature results (for figures)
    best_label = max(temp_summaries, key=lambda s: float(s.get("avg_QED", 0)))["label"]
    best_df    = pd.read_csv(os.path.join(CFG.RESULTS_DIR, f"molecules_{best_label}.csv"))
    best_summary = next(s for s in temp_summaries if s["label"] == best_label)

    # Baseline evaluation
    base_res = evaluate(
        smiles_list  = baseline_smiles,
        train_smiles = data["train_smiles"],
        label        = "baseline",
        save_csv     = True,
    )
    base_summary = base_res["summary"]
    base_df      = base_res["df"]

    # Benchmark table  
    benchmark_df = build_benchmark_table(all_results + [base_res])
    bench_path   = os.path.join(CFG.RESULTS_DIR, "benchmark_table.csv")
    benchmark_df.to_csv(bench_path)
    print(f"\n[Main] Benchmark table saved -> {bench_path}")
    print(benchmark_df.to_string())

    # -- 6. All figures ----------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 6: Figure Generation")
    print("="*60)
    from visualize import generate_all_figures
    generate_all_figures(
        history_csv        = history_csv,
        df_model           = best_df,
        df_baseline        = base_df,
        summaries_by_temp  = temp_summaries,
        model_summary      = best_summary,
        baseline_summary   = base_summary,
        train_smiles       = data["train_smiles"],
    )

    # -- 7. Final summary --------------------------------------------------
    print("\n" + "="*60)
    print("PIPELINE COMPLETE ✓")
    print("="*60)
    print(f"  Results dir : {CFG.RESULTS_DIR}")
    print(f"  Figures dir : {CFG.FIGURES_DIR}")
    print(f"  Model dir   : {CFG.MODEL_DIR}")
    print(f"\n  Best config : {best_label}")
    for k, v in best_summary.items():
        print(f"    {k:<25}: {v}")

    # Print top-5 molecules
    if len(best_df) > 0:
        print("\n  [Top] Top 5 Generated Molecules:")
        cols = ["SMILES", "MolWeight", "QED", "DrugScore", "Toxicity", "FinalScore"]
        print(best_df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
