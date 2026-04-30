"""
compare.py — Comparaison des 3 approches sur le benchmark d'Adam
-----------------------------------------------------------------
Charge les resultats d'Antoine, de Samuel et d'Adam (tous dans resultats/),
puis compare accuracy, F1 par classe et vitesse d'analyse.

Usage :
    python compare.py

Prerequis :
    - Avoir lance antoine_benchmark.py  -> resultats/benchmark_antoine.csv
    - Avoir lance samuel_benchmark.py   -> resultats/benchmark_samuel.csv
    - Avoir lance adam_benchmark.py     -> resultats/benchmark_adam.csv
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict

DATASET      = os.path.join(os.path.dirname(__file__), 'benchmark_dataset.json')
ANTOINE_CSV  = os.path.join(os.path.dirname(__file__), 'resultats', 'benchmark_antoine.csv')
SAMUEL_CSV   = os.path.join(os.path.dirname(__file__), 'resultats', 'benchmark_samuel.csv')
ADAM_CSV     = os.path.join(os.path.dirname(__file__), 'resultats', 'benchmark_adam.csv')

CLASSES = ['Achat', 'Vente', 'Neutre']


# -- Helpers ---------------------------------------------------------------------

def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def classification_metrics(rows, pred_col='prediction', gt_col='ground_truth_signal'):
    """
    Calcule accuracy et F1 par classe sur les articles pertinents seulement.
    Ignore les lignes avec gt_signal == '' ou None (hors_scope).
    """
    pertinent = [r for r in rows
                 if r.get('ground_truth_filtrage') == 'pertinent'
                 and r.get(gt_col) and r.get(pred_col)]

    if not pertinent:
        return None

    correct = sum(1 for r in pertinent if r[pred_col] == r[gt_col])
    accuracy = round(correct / len(pertinent), 3)

    per_class = {}
    for cls in CLASSES:
        tp = sum(1 for r in pertinent if r[gt_col] == cls and r[pred_col] == cls)
        fp = sum(1 for r in pertinent if r[gt_col] != cls and r[pred_col] == cls)
        fn = sum(1 for r in pertinent if r[gt_col] == cls and r[pred_col] != cls)
        support = sum(1 for r in pertinent if r[gt_col] == cls)
        p, rec, f1 = prf(tp, fp, fn)
        per_class[cls] = {'p': p, 'r': rec, 'f1': f1, 'support': support}

    macro_f1 = round(sum(per_class[c]['f1'] for c in CLASSES) / len(CLASSES), 3)

    return {
        'n': len(pertinent),
        'correct': correct,
        'accuracy': accuracy,
        'per_class': per_class,
        'macro_f1': macro_f1,
    }


def latency_stats(rows, lat_col='latency_s'):
    """Calcule latence totale et moyenne."""
    lats = [float(r[lat_col]) for r in rows if r.get(lat_col)]
    if not lats:
        return None, None
    return round(sum(lats), 1), round(sum(lats) / len(lats), 2)


def load_adam_results():
    """Charge benchmark_adam.csv depuis resultats/ (genere par adam_benchmark.py)."""
    rows = load_csv(ADAM_CSV)
    if rows is None:
        return None, None, None
    metrics = classification_metrics(rows)
    lat_total, lat_avg = latency_stats(rows)
    return metrics, lat_total, lat_avg


# -- Affichage --------------------------------------------------------------------

def print_metrics_row(name, m, lat_total, lat_avg):
    if m is None:
        print(f"  {name:<12} | DONNEES MANQUANTES")
        return
    acc_str = f"{m['accuracy']:.0%}" if m['accuracy'] is not None else 'N/A'
    mf1_str = f"{m['macro_f1']:.3f}" if m['macro_f1'] is not None else 'N/A'
    lat_str = f"{lat_avg:.2f}s/art" if lat_avg else 'N/A'
    tot_str = f"({lat_total:.0f}s total)" if lat_total else ''
    print(f"  {name:<12} | Acc={acc_str:>6} | MacroF1={mf1_str:>6} | Vitesse={lat_str} {tot_str}")


def print_per_class(name, m):
    if m is None:
        return
    for cls in CLASSES:
        pc = m['per_class'].get(cls, {})
        f1  = pc.get('f1')
        sup = pc.get('support', '?')
        f1_str = f"{f1:.3f}" if f1 is not None else ' N/A '
        print(f"    {name:<12} | {cls:<7} | F1={f1_str} | support={sup}")


def confusion_block(rows, name, pred_col='prediction', gt_col='ground_truth_signal'):
    pertinent = [r for r in rows
                 if r.get('ground_truth_filtrage') == 'pertinent'
                 and r.get(gt_col) and r.get(pred_col)]
    if not pertinent:
        return

    print(f"\n  {name} — Matrice de confusion (lignes=GT, cols=Prédit)")
    print(f"  {'':>10} | {'Achat':>7} | {'Vente':>7} | {'Neutre':>7}")
    print(f"  {'-'*40}")
    for gt_cls in CLASSES:
        row_str = f"  {gt_cls:<10} |"
        for pred_cls in CLASSES:
            count = sum(1 for r in pertinent
                        if r[gt_col] == gt_cls and r[pred_col] == pred_cls)
            marker = ' <-' if gt_cls == pred_cls else ''
            row_str += f" {count:>5}{marker if marker else '   '} |"
        print(row_str)


def detail_errors(rows, name, pred_col='prediction', gt_col='ground_truth_signal'):
    errors = [r for r in rows
              if r.get('ground_truth_filtrage') == 'pertinent'
              and r.get(gt_col) and r.get(pred_col)
              and r[gt_col] != r[pred_col]]
    if not errors:
        print(f"  {name} : aucune erreur !")
        return
    print(f"  {name} — {len(errors)} erreur(s) :")
    for r in errors:
        print(f"    [{r['id']:12s}] GT={r[gt_col]:7s} | Prédit={r[pred_col]:7s} | {r['title'][:50]}")


# -- Main -------------------------------------------------------------------------

def main():
    # Chargement des données
    with open(DATASET, encoding='utf-8') as f:
        dataset = json.load(f)

    antoine_rows = load_csv(ANTOINE_CSV)
    samuel_rows  = load_csv(SAMUEL_CSV)
    adam_rows    = load_csv(ADAM_CSV)

    antoine_metrics = classification_metrics(antoine_rows) if antoine_rows else None
    samuel_metrics  = classification_metrics(samuel_rows)  if samuel_rows  else None
    adam_metrics    = classification_metrics(adam_rows)    if adam_rows    else None

    antoine_lat_total, antoine_lat_avg = latency_stats(antoine_rows) if antoine_rows else (None, None)
    samuel_lat_total,  samuel_lat_avg  = latency_stats(samuel_rows)  if samuel_rows  else (None, None)
    adam_lat_total,    adam_lat_avg    = latency_stats(adam_rows)    if adam_rows    else (None, None)

    n_total     = len(dataset['articles'])
    n_pertinent = sum(1 for a in dataset['articles'] if a['ground_truth']['filtrage'] == 'pertinent')
    n_hors      = n_total - n_pertinent

    # -- En-tête --------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  BENCHMARK COMPARATIF — 3 approches vs vérité terrain d'Adam")
    print(f"{'='*70}")
    print(f"  Dataset : {n_total} articles ({n_pertinent} pertinents, {n_hors} hors_scope)")
    print(f"  Ground truth : cours réel J+5 (Achat >+1.5%, Vente <-1.5%, Neutre sinon)")
    print()

    # -- Tableau récapitulatif ------------------------------------------------------
    print(f"  {'APPROCHE':<12} | {'ACCURACY':>8} | {'MACRO F1':>8} | {'VITESSE':>14}")
    print(f"  {'-'*60}")

    def summary_row(name, m, lat_avg):
        acc = f"{m['accuracy']:.0%}" if m and m['accuracy'] is not None else 'N/A'
        mf1 = f"{m['macro_f1']:.3f}" if m and m['macro_f1'] is not None else 'N/A'
        lat = f"{lat_avg:.2f}s/art" if lat_avg else 'N/A'
        missing = '' if m else '  [manquant]'
        print(f"  {name:<12} | {acc:>8} | {mf1:>8} | {lat:>14}{missing}")

    summary_row("Antoine",  antoine_metrics, antoine_lat_avg)
    summary_row("Samuel",   samuel_metrics,  samuel_lat_avg)
    summary_row("Adam",     adam_metrics,    adam_lat_avg)

    # -- F1 par classe --------------------------------------------------------------
    print(f"\n  {'-'*70}")
    print(f"  F1 PAR CLASSE")
    print(f"  {'-'*70}")
    print(f"  {'APPROCHE':<12} | {'CLASSE':>7} | {'F1':>7} | {'SUPPORT':>8}")
    print(f"  {'-'*45}")
    for name, m in [("Antoine", antoine_metrics), ("Samuel", samuel_metrics), ("Adam", adam_metrics)]:
        print_per_class(name, m)
        print(f"  {'-'*45}")

    # -- Matrices de confusion -----------------------------------------------------
    print(f"\n  {'-'*70}")
    print(f"  MATRICES DE CONFUSION")
    if antoine_rows:
        confusion_block(antoine_rows, "Antoine")
    if samuel_rows:
        confusion_block(samuel_rows,  "Samuel")
    if adam_rows:
        confusion_block(adam_rows,    "Adam")

    # -- Détail des erreurs --------------------------------------------------------
    print(f"\n  {'-'*70}")
    print(f"  DETAIL DES ERREURS")
    if antoine_rows:
        print()
        detail_errors(antoine_rows, "Antoine")
    if samuel_rows:
        print()
        detail_errors(samuel_rows, "Samuel")
    if adam_rows:
        print()
        detail_errors(adam_rows, "Adam")

    # -- Comparaison vitesse -------------------------------------------------------
    print(f"\n  {'-'*70}")
    print(f"  VITESSE D'ANALYSE")
    print(f"  {'-'*70}")
    speeds = []
    for name, lat_avg, lat_total in [
        ("Antoine (Ollama phi4-mini)", antoine_lat_avg, antoine_lat_total),
        ("Samuel  (FinBERT + LogReg)", samuel_lat_avg,  samuel_lat_total),
        ("Adam    (ABSA + débat LLM)", adam_lat_avg,    adam_lat_total),
    ]:
        if lat_avg:
            speeds.append((name, lat_avg, lat_total))
            print(f"  {name:<30} : {lat_avg:>6.2f}s/article  ({lat_total:.0f}s sur {n_total} art.)")
        else:
            print(f"  {name:<30} : N/A")

    if len(speeds) >= 2:
        fastest = min(speeds, key=lambda x: x[1])
        slowest = max(speeds, key=lambda x: x[1])
        ratio   = round(slowest[1] / fastest[1], 1)
        print(f"\n  Fastest : {fastest[0].strip()} ({fastest[1]:.2f}s)")
        print(f"  Slowest : {slowest[0].strip()} ({slowest[1]:.2f}s) — {ratio}x plus lent")

    # -- Conclusion ----------------------------------------------------------------
    print(f"\n  {'-'*70}")
    print(f"  CONCLUSION")
    print(f"  {'-'*70}")

    ranked = []
    for name, m in [("Antoine", antoine_metrics), ("Samuel", samuel_metrics), ("Adam", adam_metrics)]:
        if m and m.get('macro_f1') is not None:
            ranked.append((name, m['macro_f1'], m['accuracy']))

    if ranked:
        ranked.sort(key=lambda x: x[1], reverse=True)
        print()
        for rank, (name, f1, acc) in enumerate(ranked, 1):
            medal = ['1er', '2eme', '3eme'][rank-1]
            print(f"  {medal} : {name:<10} (MacroF1={f1:.3f}, Acc={acc:.0%})")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
