"""
evaluate_pipeline.py — Couche 2 : Évaluation qualitative du pipeline complet
=============================================================================
Exécute le pipeline complet (filtrage → ABSA → débat → YOLO) sur le
benchmark_dataset.json annoté manuellement, et compare les sorties
avec les labels de référence.

Métriques calculées :
  - Accuracy filtrage (pertinent vs hors_scope)
  - F1-score par signal (Achat / Vente / Neutre)
  - Accuracy globale des signaux
  - Taux d'erreurs de parsing (JSON malformé)
  - YOLO : distribution des niveaux de risque

Lancé via : python eval/run_eval.py --layer 2 --limit N
"""

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)  # Silence AutoGen/LangGraph logs
logger = logging.getLogger("EvalPipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Calcul des métriques classification (F1, précision, rappel)
# ---------------------------------------------------------------------------


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def _compute_classification_metrics(predictions: list[dict]) -> dict:
    """
    Calcule les métriques multi-classe (Achat/Vente/Neutre).
    predictions : liste de {"ground_truth": str, "prediction": str}
    """
    classes = ["Achat", "Vente", "Neutre"]
    metrics = {}
    correct = 0

    for cls in classes:
        tp = sum(1 for p in predictions if p["ground_truth"] == cls and p["prediction"] == cls)
        fp = sum(1 for p in predictions if p["ground_truth"] != cls and p["prediction"] == cls)
        fn = sum(1 for p in predictions if p["ground_truth"] == cls and p["prediction"] != cls)
        prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
        support = sum(1 for p in predictions if p["ground_truth"] == cls)
        metrics[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        correct += tp

    total = len(predictions)
    metrics["accuracy"] = round(correct / total, 3) if total > 0 else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Exécution du pipeline sur un article du benchmark
# ---------------------------------------------------------------------------


def _run_one_article(article: dict) -> dict:
    """
    Exécute le pipeline complet sur un article et retourne le résultat brut.
    Gère les exceptions pour que l'évaluation continue en cas d'erreur.
    """
    from src.agents.agent_absa import run_absa
    from src.agents.agent_debat import workflow_debat_actualite
    from src.agents.agent_filtrage import workflow_filtrer_actualite
    from src.pipelines.agent_pipeline import _calculer_metrics_objectives
    from src.utils.yolo_classifier import classify_risk

    ticker = article["ticker"]
    content = f"{article['title']}\n\n{article['content']}"
    gt = article["ground_truth"]

    result = {
        "id": article["id"],
        "ticker": ticker,
        "title": article["title"][:60],
        "ground_truth": gt,
        "filtrage_pred": None,
        "signal_pred": None,
        "argument_dominant": None,
        "risk_level": None,
        "consensus_rate": None,
        "impact_strength": None,
        "parsing_error": False,
        "pipeline_error": None,
        "latency_s": None,
    }

    t0 = time.time()
    try:
        # Étape 1 — Filtrage
        filtrage = workflow_filtrer_actualite.invoke(
            {
                "texte_article": content,
                "ticker_symbol": ticker,
            }
        )
        pertinent = filtrage.get("pertinent", True)
        result["filtrage_pred"] = "pertinent" if pertinent else "hors_scope"

        if not pertinent:
            result["latency_s"] = round(time.time() - t0, 2)
            return result  # Arrêt si filtré

        # Étape 2 — ABSA
        absa_result = run_absa(content, ticker)

        # Étape 3 — Débat
        contexte_marche = {}  # Pas de données live en mode benchmark
        decision = workflow_debat_actualite.invoke(
            {
                "texte_article": content,
                "ticker_symbol": ticker,
                "contexte_marche": contexte_marche,
                "absa_result": absa_result,
            }
        )

        signal = decision.get("signal", "Neutre")
        argument = decision.get("argument_dominant", "")
        scratchpad = decision.get("scratchpad_xml", "")

        if argument == "Parsing impossible":
            result["parsing_error"] = True

        # Étape 4 — Métriques déterministes
        # FinBERT score simulé à 0.70 (neutre agréable) en mode benchmark —
        # l'important est la cohérence du pipeline, pas le score absolu.
        score_finbert_sim = 0.70
        consensus_rate, impact_strength = _calculer_metrics_objectives(signal, score_finbert_sim, absa_result)

        # Étape 5 — YOLO
        yolo = classify_risk(
            signal_final=signal,
            consensus_rate=consensus_rate,
            impact_strength=impact_strength,
            scratchpad_xml=scratchpad,
            absa_result=absa_result,
            score_finbert=score_finbert_sim,
            contexte_marche=contexte_marche,
        )

        result.update(
            {
                "signal_pred": signal,
                "argument_dominant": argument[:100],
                "risk_level": yolo.risk_level,
                "consensus_rate": round(consensus_rate, 3),
                "impact_strength": round(impact_strength, 3),
            }
        )

    except Exception as e:
        result["pipeline_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    result["latency_s"] = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Rapport des résultats
# ---------------------------------------------------------------------------


def _print_report(results: list[dict], metrics: dict, latency_total: float) -> None:
    """Affiche le rapport complet et synthétique."""
    total = len(results)
    errors = [r for r in results if r.get("pipeline_error")]
    parse_errors = [r for r in results if r.get("parsing_error")]

    print("\n" + "=" * 70)
    print("COUCHE 2 : Rapport d'évaluation pipeline complet")
    print("=" * 70)

    # Filtrage
    filtrage_correct = sum(1 for r in results if r["filtrage_pred"] == r["ground_truth"]["filtrage"])
    print("\n  Filtrage:")
    print(f"    Accuracy : {filtrage_correct}/{total} = {filtrage_correct / total:.0%}")

    # Signaux (articles pertinents uniquement)
    signal_results = [
        r
        for r in results
        if r["ground_truth"]["filtrage"] == "pertinent" and r["signal_pred"] is not None and not r.get("parsing_error")
    ]
    if signal_results:
        print(f"\n  Signaux ({len(signal_results)} articles pertinents) :")
        sig_metrics = _compute_classification_metrics(
            [{"ground_truth": r["ground_truth"]["signal"], "prediction": r["signal_pred"]} for r in signal_results]
        )
        print(f"    Accuracy globale : {sig_metrics['accuracy']:.0%}")
        print(f"    {'Signal':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"    {'-' * 46}")
        for cls in ["Achat", "Vente", "Neutre"]:
            m = sig_metrics.get(cls, {})
            print(
                f"    {cls:<10} {m.get('precision', 0):>10.0%} "
                f"{m.get('recall', 0):>10.0%} {m.get('f1', 0):>10.0%} "
                f"{m.get('support', 0):>10}"
            )
    else:
        print("\n  Signaux : Pas assez de résultats valides.")
        sig_metrics = {}

    # YOLO Distribution
    risk_dist = defaultdict(int)
    for r in results:
        if r.get("risk_level"):
            risk_dist[r["risk_level"]] += 1
    print("\n  YOLO Risk Distribution :")
    for level in ["FAIBLE", "MOYEN", "ELEVE"]:
        count = risk_dist.get(level, 0)
        bar = "#" * count
        print(f"    {level:<8} : {bar} ({count})")

    # Erreurs
    print(f"\n  Erreurs de parsing    : {len(parse_errors)}/{total}")
    print(f"  Erreurs de pipeline   : {len(errors)}/{total}")
    print(f"  Latence totale        : {latency_total:.1f}s")
    print(f"  Latence moyenne/art.  : {latency_total / total:.1f}s")

    # Détail par article
    print("\n  Détail par article :")
    print(f"  {'ID':<12} {'Label':>8} {'Prédit':>8} {'Risque':>8} {'Correct':>8} {'Lat(s)':>8}")
    print(f"  {'-' * 56}")
    for r in results:
        label = r["ground_truth"]["signal"] or "filtré"
        pred = r["signal_pred"] or ("filtré" if r["filtrage_pred"] == "hors_scope" else "—")
        risk = r.get("risk_level") or "—"
        ok = "OK" if label == pred else "FAIL"
        lat = r.get("latency_s", "—")
        print(f"  {r['id']:<12} {label:>8} {pred:>8} {risk:>8} {ok:>8} {str(lat):>8}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def run_pipeline_benchmark(limit: int = 0, save_results: bool = True) -> dict:
    """
    Lance la Couche 2 sur le dataset de référence.
    limit : nb d'articles à évaluer (0 = tous)
    """
    dataset_path = Path(__file__).parent / "benchmark_dataset.json"
    if not dataset_path.exists():
        print("[ERREUR] benchmark_dataset.json introuvable. Lancez d'abord la Couche 1.")
        return {}

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data["articles"]
    if limit > 0:
        articles = articles[:limit]

    print(f"\n{'=' * 70}")
    print(f"COUCHE 2 : Évaluation pipeline complet ({len(articles)} articles)")
    print(f"{'=' * 70}")

    results = []
    latency_total = 0.0

    for i, article in enumerate(articles, 1):
        print(f"  [{i:02d}/{len(articles):02d}] {article['id']} — {article['title'][:50]}...", end=" ", flush=True)
        r = _run_one_article(article)
        results.append(r)
        latency_total += r.get("latency_s", 0.0)

        label = r["ground_truth"]["signal"] or "filtré"
        pred = r["signal_pred"] or ("filtré" if r["filtrage_pred"] == "hors_scope" else "ERR")
        ok = "OK" if label == pred else "FAIL"
        print(f"[{ok}] ({r.get('latency_s', '?')}s)")

    signal_results = [
        r
        for r in results
        if r["ground_truth"]["filtrage"] == "pertinent" and r["signal_pred"] is not None and not r.get("parsing_error")
    ]
    metrics = {}
    if signal_results:
        metrics = _compute_classification_metrics(
            [{"ground_truth": r["ground_truth"]["signal"], "prediction": r["signal_pred"]} for r in signal_results]
        )

    _print_report(results, metrics, latency_total)

    # Sauvegarde dans eval_results/
    if save_results:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(__file__).parent / "eval_results" / f"{timestamp}_pipeline_layer2"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "details.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        summary = {
            "timestamp": timestamp,
            "n_articles": len(articles),
            "filtrage_accuracy": sum(1 for r in results if r["filtrage_pred"] == r["ground_truth"]["filtrage"])
            / len(results),
            "signal_accuracy": metrics.get("accuracy", None),
            "signal_f1": {cls: metrics.get(cls, {}).get("f1") for cls in ["Achat", "Vente", "Neutre"]},
            "parsing_errors": sum(1 for r in results if r.get("parsing_error")),
            "pipeline_errors": sum(1 for r in results if r.get("pipeline_error")),
            "latency_total_s": round(latency_total, 2),
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  Résultats sauvegardés dans : {run_dir}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run_pipeline_benchmark(limit=args.limit)
