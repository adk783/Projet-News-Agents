"""
evaluate_calibration.py -- Calibration décisionnelle : Brier Score, ECE, Self-Consistency
============================================================================================
OBJECTIF :
  Un bon système décisionnel ne doit pas seulement être précis — il doit
  aussi être *calibré* : quand il dit "confiance = 0.85", il doit réallement
  avoir raison 85% du temps sur les cas similaires.

MÉTRIQUES IMPLÉMENTÉES :

  1. BRIER SCORE (Brier, 1950)
     BS = mean((confidence - outcome)²)
     0 = calibration parfaite, 1 = calibration inverse
     → Utilise impact_strength comme proxy de confiance
     → outcome = 1 si signal correct dans le marché, 0 sinon

  2. ECE — Expected Calibration Error
     Groupe les prédictions par décile de confiance.
     Dans chaque groupe : |précision réelle - confiance déclarée|
     ECE = moyenne pondérée de ces écarts
     → 0% = parfaitement calibré, 20%+ = très mal calibré

  3. SELF-CONSISTENCY (inter-run)
     Proportion de runs (sur N exécutions du même article) qui produisent
     le même signal. Mesure si les agents sont stables ou erratiques.
     → 1.0 = unanimité absolue, 0.33 = équivalent à un tirage au sort à 3 classes

  4. VARIANCE INTER-RUN
     Écart-type des signaux encodés numériquement (Achat=1, Neutre=0, Vente=-1)
     → 0 = tous les runs donnent le même signal
     → 1+ = les runs donnent des signaux opposés

NOTE SUR IMPACT_STRENGTH COMME PROXY DE CONFIANCE :
  impact_strength est calculé de façon déterministe (FinBERT + ABSA), pas
  par le LLM. C'est un proxy imparfait : il mesure la "force du signal"
  plutôt que la "confiance probabiliste" du modèle.
  L'ECE et le Brier Score restent informatifs mais avec ce caveat en tête.

Lancé via : python eval/run_eval.py --layer 5 --sub calibration
"""

import asyncio
import json
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Calibration")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"

# Encodage numérique des signaux pour la variance inter-run
SIGNAL_TO_NUM = {"Achat": 1, "Neutre": 0, "Vente": -1}


# ---------------------------------------------------------------------------
# Chargement des données combinées L2+L3
# ---------------------------------------------------------------------------


def _load_l2_l3_combined() -> list[dict]:
    """
    Charge les prédictions de L2 avec les outcomes de marché de L3.
    Seuls les articles présents dans les deux sources sont retenus.
    """
    # Charger L2 (prédictions)
    l2_runs = (
        sorted([d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir() and "pipeline_layer2" in d.name], reverse=True)
        if EVAL_RESULTS_DIR.exists()
        else []
    )

    if not l2_runs:
        return []

    l2_file = l2_runs[0] / "details.json"
    if not l2_file.exists():
        return []

    with open(l2_file, encoding="utf-8") as f:
        l2_data = json.load(f)

    # Charger L3 (outcomes de marché)
    l3_runs = (
        sorted([d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir() and "market_layer3" in d.name], reverse=True)
        if EVAL_RESULTS_DIR.exists()
        else []
    )

    l3_outcomes = {}
    if l3_runs:
        l3_file = l3_runs[0] / "trades.json"
        if l3_file.exists():
            with open(l3_file, encoding="utf-8") as f:
                trades = json.load(f)
            for t in trades:
                key = t.get("id", "")
                if key:
                    l3_outcomes[key] = {
                        "signal_correct": t.get("signal_correct"),
                        "return_pct": t.get("return_pct"),
                        "change_pct": t.get("change_pct"),
                    }

    combined = []
    for r in l2_data:
        if r.get("filtrage_pred") != "pertinent" or r.get("signal_pred") is None or r.get("impact_strength") is None:
            continue

        outcome = l3_outcomes.get(r["id"], {})
        signal_correct = outcome.get("signal_correct")  # None si pas de données marché

        combined.append(
            {
                "id": r["id"],
                "ticker": r["ticker"],
                "signal_pred": r["signal_pred"],
                "impact_strength": r.get("impact_strength", 0.5),
                "consensus_rate": r.get("consensus_rate", 0.5),
                "risk_level": r.get("risk_level", "INCONNU"),
                "gt_signal": r.get("ground_truth", {}).get("signal"),
                # Outcome marché (None si pas de données)
                "signal_correct_market": signal_correct,
                # Outcome benchmark (comparaison avec label annoté)
                "signal_correct_benchmark": (
                    r["signal_pred"] == r.get("ground_truth", {}).get("signal")
                    if r.get("ground_truth", {}).get("signal")
                    else None
                ),
            }
        )

    return combined


# ---------------------------------------------------------------------------
# 1. BRIER SCORE
# ---------------------------------------------------------------------------


def compute_brier_score(predictions: list[dict], use_market: bool = True) -> dict:
    """
    BS = mean((confidence - outcome)²)

    confidence = impact_strength (proxy déterministe)
    outcome    = 1 si signal correct, 0 sinon
    """
    valid = []
    source = "marche" if use_market else "benchmark"

    for p in predictions:
        if use_market:
            correct = p.get("signal_correct_market")
        else:
            correct = p.get("signal_correct_benchmark")

        if correct is None:
            continue

        confidence = p["impact_strength"]
        outcome = 1.0 if correct else 0.0
        bs_i = (confidence - outcome) ** 2
        valid.append((confidence, outcome, bs_i, p["id"]))

    if not valid:
        return {
            "brier_score": None,
            "n_samples": 0,
            "source": source,
            "note": "Aucune donnee disponible — lancez L2 + L3 d'abord.",
        }

    bs = sum(v[2] for v in valid) / len(valid)
    bs_by_bucket = {
        "low_confidence": [v[2] for v in valid if v[0] < 0.50],
        "mid_confidence": [v[2] for v in valid if 0.50 <= v[0] < 0.75],
        "high_confidence": [v[2] for v in valid if v[0] >= 0.75],
    }

    return {
        "brier_score": round(bs, 4),
        "n_samples": len(valid),
        "source": source,
        "interpretation": (
            "Calibration excellente (BS < 0.10)"
            if bs < 0.10
            else "Calibration bonne (BS 0.10-0.20)"
            if bs < 0.20
            else "Calibration moderate (BS 0.20-0.33)"
            if bs < 0.33
            else "Calibration mauvaise (BS > 0.33) — confidence_impact desalignnee avec precision"
        ),
        "bs_by_bucket": {k: (sum(v) / len(v) if v else None) for k, v in bs_by_bucket.items()},
    }


# ---------------------------------------------------------------------------
# 2. ECE — Expected Calibration Error
# ---------------------------------------------------------------------------


def compute_ece(predictions: list[dict], n_bins: int = 5, use_market: bool = True) -> dict:
    """
    Groupe par décile de confiance (impact_strength), compare accuracy réelle vs confiance.
    ECE = Σ (|bin| / N) * |accuracy_bin - mean_confidence_bin|
    """
    valid = []
    for p in predictions:
        correct = p.get("signal_correct_market") if use_market else p.get("signal_correct_benchmark")
        if correct is None:
            continue
        valid.append((p["impact_strength"], 1.0 if correct else 0.0))

    if len(valid) < 5:
        return {
            "ece": None,
            "n_samples": len(valid),
            "note": "Echantillon insuffisant (< 5) pour calculer l'ECE.",
        }

    # Construction des bins
    bins = defaultdict(list)
    bin_size = 1.0 / n_bins
    for conf, outcome in valid:
        bin_idx = min(int(conf / bin_size), n_bins - 1)
        bins[bin_idx].append((conf, outcome))

    ece = 0.0
    bin_details = []
    n_total = len(valid)

    for i in range(n_bins):
        if not bins[i]:
            continue
        mean_conf = sum(c for c, o in bins[i]) / len(bins[i])
        accuracy = sum(o for c, o in bins[i]) / len(bins[i])
        weight = len(bins[i]) / n_total
        ece += weight * abs(accuracy - mean_conf)
        bin_details.append(
            {
                "bin": f"[{i * bin_size:.1f}, {(i + 1) * bin_size:.1f}]",
                "n": len(bins[i]),
                "mean_conf": round(mean_conf, 3),
                "accuracy": round(accuracy, 3),
                "gap": round(abs(accuracy - mean_conf), 3),
            }
        )

    return {
        "ece": round(ece, 4),
        "n_samples": n_total,
        "bins": bin_details,
        "interpretation": (
            "Tres bien calibre (ECE < 5%)"
            if ece < 0.05
            else "Bien calibre (ECE 5-10%)"
            if ece < 0.10
            else "Modere (ECE 10-20%)"
            if ece < 0.20
            else "Mal calibre (ECE > 20%) — revoir le calcul d'impact_strength"
        ),
    }


# ---------------------------------------------------------------------------
# 3. SELF-CONSISTENCY (inter-run)
# ---------------------------------------------------------------------------


def compute_self_consistency(
    articles_subset: list[dict],
    n_runs: int = 3,
) -> dict:
    """
    Lance le pipeline N fois sur chaque article et mesure la cohérence des signaux.
    Utilise le module consensus_voter existant.
    """
    from src.utils.consensus_voter import SEUIL_INSTABILITE, run_with_majority_vote

    all_variances = []
    all_consistent = []

    for article in articles_subset:
        ticker = article["ticker"]
        gt = article.get("ground_truth", {})
        content = f"{article.get('title', '')}\n\n{article.get('content', '')}"
        date = gt.get("date", "")

        logger.info("[Calib] Self-consistency %s %s (%d runs)...", article.get("id"), ticker, n_runs)

        vote = run_with_majority_vote(
            article_text=content,
            ticker=ticker,
            contexte_marche={},
            absa_result={"aspects": []},
            n_runs=n_runs,
        )

        all_variances.append(vote.variance)
        all_consistent.append(vote.variance <= SEUIL_INSTABILITE)

    n_consistent = sum(all_consistent)
    avg_variance = sum(all_variances) / len(all_variances) if all_variances else 1.0
    consistency_rate = n_consistent / len(all_consistent) if all_consistent else 0.0

    return {
        "n_articles": len(articles_subset),
        "n_runs_per_art": n_runs,
        "consistency_rate": round(consistency_rate, 3),
        "avg_variance": round(avg_variance, 3),
        "n_stable": n_consistent,
        "per_article": [
            {"id": articles_subset[i].get("id"), "variance": round(all_variances[i], 3), "stable": all_consistent[i]}
            for i in range(len(articles_subset))
        ],
        "interpretation": (
            f"Tres stable ({consistency_rate:.0%} des signaux coherents sur {n_runs} runs)"
            if consistency_rate >= 0.8
            else f"Instabilite moderee ({consistency_rate:.0%} stables) — "
            f"variance moyenne {avg_variance:.0%}. Envisager temperature=0 ou plus de runs."
            if consistency_rate >= 0.5
            else f"Instabilite severe ({consistency_rate:.0%} stables) — "
            f"Les agents sont erratiques. Temperature trop elevee ou modele pas adapte."
        ),
    }


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_calibration_analysis(
    n_runs_consistency: int = 3,
    n_articles_consistency: int = 2,
) -> dict:
    """Lance l'analyse de calibration complète."""
    print(f"\n{'=' * 70}")
    print("COUCHE 5b : Calibration Decisionnelle (Brier Score, ECE, Self-Consistency)")
    print(f"{'=' * 70}")

    predictions = _load_l2_l3_combined()

    if not predictions:
        print("\n  [INFO] Aucune donnee. Lancez L2 puis L3 d'abord:")
        print("  python eval/run_eval.py --layer 2 --limit 10")
        print("  python eval/run_eval.py --layer 3")
        return {}

    print(f"\n  {len(predictions)} predictions chargees (L2+L3)")

    # - Brier Score (sur benchmark annoté car c'est celui qu'on a toujours)
    print("\n  [1/3] Brier Score...")
    bs_benchmark = compute_brier_score(predictions, use_market=False)
    bs_market = compute_brier_score(predictions, use_market=True)

    print(
        f"    Brier Score (benchmark) : {bs_benchmark['brier_score']} "
        f"({bs_benchmark['n_samples']} samples) — {bs_benchmark.get('interpretation', 'N/A')}"
    )
    print(
        f"    Brier Score (marche)    : {bs_market['brier_score']} "
        f"({bs_market['n_samples']} samples) — {bs_market.get('interpretation', 'N/A')}"
    )

    # ECE
    print("\n  [2/3] Expected Calibration Error...")
    ece_bench = compute_ece(predictions, use_market=False)
    ece_mkt = compute_ece(predictions, use_market=True)

    if ece_bench.get("ece") is not None:
        print(f"    ECE (benchmark) : {ece_bench['ece']:.2%} — {ece_bench['interpretation']}")
        if ece_bench.get("bins"):
            print(f"    {'Bin':<16} {'N':>4} {'Conf':>8} {'Acc':>8} {'Gap':>8}")
            print(f"    {'-' * 48}")
            for b in ece_bench["bins"]:
                print(f"    {b['bin']:<16} {b['n']:>4} {b['mean_conf']:>8.0%} {b['accuracy']:>8.0%} {b['gap']:>8.0%}")
    else:
        print(f"    ECE : {ece_bench.get('note', 'N/A')}")

    if ece_mkt.get("ece") is not None:
        print(f"    ECE (marche)    : {ece_mkt['ece']:.2%} — {ece_mkt['interpretation']}")

    # Self-consistency (nécessite des appels LLM)
    print(f"\n  [3/3] Self-Consistency ({n_runs_consistency} runs × {n_articles_consistency} articles)...")
    print("    Note: Cette etape appelle les LLMs. Attendez...\n")

    dataset_path = Path(__file__).parent / "benchmark_dataset.json"
    consistency_result = {}
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8") as f:
            bench_data = json.load(f)
        test_articles = [a for a in bench_data["articles"] if a["ground_truth"]["filtrage"] == "pertinent"][
            :n_articles_consistency
        ]

        consistency_result = compute_self_consistency(test_articles, n_runs=n_runs_consistency)
        print(f"    {consistency_result['interpretation']}")
        print(f"    Variance moyenne : {consistency_result['avg_variance']:.0%}")
        if consistency_result.get("per_article"):
            for pa in consistency_result["per_article"]:
                status = "STABLE" if pa["stable"] else "INSTABLE"
                print(f"    {pa['id']}: variance={pa['variance']:.0%} [{status}]")
    else:
        print("    benchmark_dataset.json introuvable.")

    print(f"\n{'=' * 70}")

    return {
        "sub": "calibration",
        "brier_benchmark": bs_benchmark,
        "brier_market": bs_market,
        "ece_benchmark": ece_bench,
        "ece_market": ece_mkt,
        "self_consistency": consistency_result,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--n-articles", type=int, default=2)
    args = p.parse_args()
    run_calibration_analysis(n_runs_consistency=args.n_runs, n_articles_consistency=args.n_articles)
