"""
evaluate_error_propagation.py -- Couche 6b : Analyse Causale des Erreurs
========================================================================
Utilise les logs structurés de la Couche 2 (details_*.json) pour attribuer la cause
racine exacte d'un échec de bout-en-bout aux différents sous-composants, et
calcule la probabilité conditionnelle d'échec.

Définition mathématique des FAILS :
  1. Filtrage : filtrage_pred != ground_truth["filtrage"]
  2. ABSA     : GT=pertinent MAIS (0 aspects ET finbert < 0.5)
  3. Débat    : auc_agreement <= 0.20 ET confidence_delta == 0
  4. Consensus: Si aucun des 3 fails précédents, mais signal_pred != GT["signal"]
"""

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

# On réutilise notre script de dynamique pour évaluer 'live' si le débat a fail
try:
    from eval.evaluate_debate_dynamics import analyse_scratchpad_live
except ImportError:
    analyse_scratchpad_live = None


def _get_latest_details_json() -> str:
    """Trouve le fichier details_*.json le plus récent."""
    search_dir = Path("eval/eval_results")
    files = list(search_dir.glob("*/details.json"))
    if not files:
        return ""
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])


def analyze_causality(limit_files: int = 1) -> dict:
    """Analyse la propagation des erreurs."""
    print(f"\n{'=' * 70}")
    print("COUCHE 6b : Analyse Causale & Propagation des Erreurs")
    print(f"{'=' * 70}")

    latest_file = _get_latest_details_json()
    if not latest_file:
        print("  [ERREUR] Aucun fichier 'details.json' trouvé dans eval/eval_results/.")
        print("           Lancez d'abord la Couche 2 (evaluate_pipeline.py).")
        return {}

    try:
        with open(latest_file, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                articles = data
            else:
                articles = data.get("results", [])
    except Exception as e:
        print(f"  [ERREUR] Impossible de lire {latest_file} : {e}")
        return {}

    if not articles:
        print("  Le fichier JSON est vide.")
        return {}

    print(f"  Fichier audit : {latest_file}")
    print(f"  Articles à analyser : {len(articles)}\n")

    # Compteurs
    metrics = {
        "total": len(articles),
        "total_errors": 0,
        "cause_filtrage": 0,
        "cause_absa": 0,
        "cause_debate": 0,
        "cause_consensus": 0,
        "absa_evaluations": 0,
        "debate_evaluations": 0,
        "absa_fails": 0,
        "debate_fails": 0,
    }

    for item in articles:
        gt = item.get("ground_truth", {})
        ticker = item.get("ticker", "UNKNOWN")

        # S'il y a eu une erreur de code pure (exception)
        if item.get("pipeline_error") or item.get("parsing_error"):
            metrics["total_errors"] += 1
            # On ne calcule pas la causalité LLM sur les crash de code.
            continue

        gt_filtrage = gt.get("filtrage")
        gt_signal = gt.get("signal")
        pred_filtrage = item.get("filtrage_pred")
        pred_signal = item.get("signal_pred")

        # Vrai succès complet ?
        if gt_filtrage == "hors_scope":
            if pred_filtrage == "hors_scope":
                continue  # Vrai Négatif (Succès)
            else:
                metrics["total_errors"] += 1
                metrics["cause_filtrage"] += 1
                continue

        if pred_filtrage != "pertinent":
            # Le GT dit pertinent, mais le prédict a dit hors scope.
            metrics["total_errors"] += 1
            metrics["cause_filtrage"] += 1
            continue

        # ---> A partir d'ici, GT = pertinent ET Pred = pertinent
        is_end_to_end_error = pred_signal != gt_signal

        # 2. Vérification ABSA
        absa = item.get("absa_result", {})
        aspects = absa.get("aspects", [])
        finbert = item.get("score_finbert", 0.0)

        metrics["absa_evaluations"] += 1
        absa_failed = False
        if len(aspects) == 0 and finbert < 0.5:
            absa_failed = True
            metrics["absa_fails"] += 1

        # 3. Vérification Débat
        metrics["debate_evaluations"] += 1
        debate_failed = False
        scratchpad = item.get("scratchpad_xml", "")
        if scratchpad and analyse_scratchpad_live:
            dyn = analyse_scratchpad_live(scratchpad, item.get("id"), ticker)
            if dyn:
                if dyn.auc_agreement is not None and dyn.confidence_delta is not None:
                    if dyn.auc_agreement <= 0.20 and dyn.confidence_delta == 0:
                        debate_failed = True
                        metrics["debate_fails"] += 1

        if is_end_to_end_error:
            metrics["total_errors"] += 1
            # Attribution Causale (Cascade)
            if absa_failed:
                metrics["cause_absa"] += 1
            elif debate_failed:
                metrics["cause_debate"] += 1
            else:
                metrics["cause_consensus"] += 1

    # Rapport
    total_e = metrics["total_errors"]
    if total_e > 0:
        pc_filt = metrics["cause_filtrage"] / total_e
        pc_absa = metrics["cause_absa"] / total_e
        pc_deb = metrics["cause_debate"] / total_e
        pc_cons = metrics["cause_consensus"] / total_e
    else:
        pc_filt = pc_absa = pc_deb = pc_cons = 0.0

    print(f"  Erreurs bout-en-bout (Mismatches Finaux) : {total_e} sur {metrics['total']}")
    if total_e > 0:
        print("\n  Attribution Causale Cascade (Qui a causé l'erreur ?) :")
        print(f"    - Filtrage Routing Error : {metrics['cause_filtrage']:>3} ({pc_filt:>4.0%})")
        print(f"    - ABSA Recall Error      : {metrics['cause_absa']:>3} ({pc_absa:>4.0%})")
        print(f"    - Debate Echo Chamber    : {metrics['cause_debate']:>3} ({pc_deb:>4.0%})")
        print(f"    - Consensus Synthesis    : {metrics['cause_consensus']:>3} ({pc_cons:>4.0%})")

    print("\n  Probabilités Conditionnelles :")
    if metrics["absa_fails"] > 0:
        p_err_given_absa = metrics["cause_absa"] / metrics["absa_fails"]
        print(f"    P(Erreur | Fail_ABSA)  = {p_err_given_absa:.0%}  (Taux mortel de l'ABSA)")
    else:
        print("    P(Erreur | Fail_ABSA)  = N/A (Aucun fail ABSA)")

    if metrics["debate_fails"] > 0:
        p_err_given_deb = metrics["cause_debate"] / metrics["debate_fails"]
        print(f"    P(Erreur | Fail_Debat) = {p_err_given_deb:.0%}  (Taux mortel de l'Echo Chamber)")
    else:
        print("    P(Erreur | Fail_Debat) = N/A (Aucun echo chamber detecté)")

    print("=" * 70)

    return {
        "sub": "causality",
        "total_errors": metrics["total_errors"],
        "cause_filtrage": metrics["cause_filtrage"],
        "cause_absa": metrics["cause_absa"],
        "cause_debate": metrics["cause_debate"],
        "cause_consensus": metrics["cause_consensus"],
    }


if __name__ == "__main__":
    analyze_causality()
