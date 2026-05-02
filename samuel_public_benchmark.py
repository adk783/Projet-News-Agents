"""
samuel_public_benchmark.py — Benchmark public des modeles Samuel sur datasets standards
----------------------------------------------------------------------------------------
Wrapper autour de benchmark_public_models.py de Samuel.
Teste PolarityAgent + UncertaintyAgent sur des datasets HuggingFace publics.

Datasets utilises :
  - financial_phrasebank       : 4840 phrases financieres labelisees (pos/neg/neu)
  - TheFinAI/fiqa-sentiment    : Q&A financier avec scores de sentiment
  - NickyNicky/Finance_sentim  : classification sentiment + topics finance EN

Usage :
    python samuel_public_benchmark.py
    python samuel_public_benchmark.py --only polarity   (polarity seul)
    python samuel_public_benchmark.py --only uncertainty

Prerequis :
    - pip install datasets transformers torch
    - Connexion internet (telechargement HuggingFace ~100MB)

Resultats sauvegardes dans :
    resultats/samuel_polarity_benchmarks.json
    resultats/samuel_uncertainty_benchmarks.json
"""

import sys
import os
import json
import argparse

# Pointer vers le dossier de Samuel
SAMUEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'samuel')
sys.path.insert(0, os.path.abspath(SAMUEL_PATH))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'resultats')
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--only', choices=['polarity', 'uncertainty'], default=None)
args = parser.parse_args()

print(f"\n{'='*65}")
print("BENCHMARK PUBLIC SAMUEL — datasets HuggingFace standard")
print(f"{'='*65}\n")

# Importation du module benchmark de Samuel (sans le modifier)
from benchmark_public_models import (
    run_polarity_benchmarks,
    run_uncertainty_benchmarks,
)

if args.only != 'uncertainty':
    print("[1/2] Benchmark Polarity (financial_phrasebank + FiQA + NickyNicky)...")
    polarity_results = run_polarity_benchmarks()
    out_path = os.path.join(OUTPUT_DIR, 'samuel_polarity_benchmarks.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(polarity_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultats -> resultats/samuel_polarity_benchmarks.json")

    # Affichage rapide
    print("\n  Resume Polarity :")
    for dataset_name, metrics in polarity_results.items():
        acc  = metrics.get('accuracy', 'N/A')
        mf1  = metrics.get('macro_f1', 'N/A')
        n    = metrics.get('dataset_size', '?')
        acc_str = f"{acc:.1%}" if isinstance(acc, float) else str(acc)
        mf1_str = f"{mf1:.3f}" if isinstance(mf1, float) else str(mf1)
        print(f"    {dataset_name:<40} Acc={acc_str:>7} MacroF1={mf1_str}")

if args.only != 'polarity':
    print("\n[2/2] Benchmark Uncertainty (FiQA proxy)...")
    uncertainty_results = run_uncertainty_benchmarks()
    out_path = os.path.join(OUTPUT_DIR, 'samuel_uncertainty_benchmarks.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(uncertainty_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultats -> resultats/samuel_uncertainty_benchmarks.json")

    print("\n  Resume Uncertainty :")
    for dataset_name, metrics in uncertainty_results.items():
        auc = metrics.get('roc_auc', 'N/A')
        n   = metrics.get('dataset_size', '?')
        auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
        print(f"    {dataset_name:<40} ROC-AUC={auc_str} (n={n})")

print(f"\n{'='*65}")
print("Benchmark public Samuel termine.")
print(f"{'='*65}\n")
