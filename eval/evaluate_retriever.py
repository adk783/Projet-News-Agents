"""
evaluate_retriever.py — Couche 7b : Évaluation du Retriever (module ABSA)
=========================================================================
Vérifie que le module ABSA "récupère" les bons aspects financiers au bon moment :
- ni trop peu  → oubli d'un signal critique (ex: "guidance" manquée)
- ni trop       → bruit inutile polluant le raisonnement des agents

Le "Retriever" de notre architecture est le module ABSA (Aspect-Based Sentiment Analysis).
Son rôle est identique à un Retriever RAG : extraire les tokens sémantiques pertinents
(les "aspects") d'un document avant de les passer aux agents de décision.

Les "ground truth" (les aspects attendus) viennent de benchmark_dataset.json → key_aspects.

Métriques calculées :
  - Context Precision  : Parmi les aspects retournés, combien sont pertinents ?
  - Context Recall     : Parmi les aspects attendus, combien ont été retrouvés ?
  - F1 Retriever       : Harmonie Precision/Recall (=NDCG simplifié sans ranking)
  - NDCG               : Normalized Discounted Cumulative Gain — pénalise les aspects
                         pertinents trouvés tardivement (ordre de retour)

Lancé via : python eval/run_eval.py --layer 7 --sub7 retriever
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("EvalRetriever")
logger.setLevel(logging.WARNING)

BENCHMARK_PATH = Path(__file__).parent / "benchmark_dataset.json"


# ---------------------------------------------------------------------------
# Métriques Retriever
# ---------------------------------------------------------------------------


def _context_precision(retrieved: list[str], relevant: list[str]) -> float:
    """
    Context Precision = |Retrouvés ∩ Pertinents| / |Retrouvés|
    Mesure : l'agent ne ramène-t-il QUE des choses utiles ?
    """
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved if r in relevant_set)
    return round(hits / len(retrieved), 3)


def _context_recall(retrieved: list[str], relevant: list[str]) -> float:
    """
    Context Recall = |Retrouvés ∩ Pertinents| / |Pertinents|
    Mesure : l'agent n'oublie-t-il PAS d'informations critiques ?
    """
    if not relevant:
        return 1.0  # Rien à trouver = succès
    relevant_set = set(relevant)
    hits = sum(1 for r in relevant_set if r in set(retrieved))
    return round(hits / len(relevant), 3)


def _ndcg(retrieved: list[str], relevant: list[str], k: int = None) -> float:
    """
    NDCG@k (Normalized Discounted Cumulative Gain).
    Récompense les aspects pertinents trouvés en premier dans la liste de retour.
    Un aspect pertinent trouvé en position 1 vaut plus qu'en position 5.

    ndcg = DCG / IDCG
    DCG  = Σ rel_i / log2(i+2)  (i=0-based)
    IDCG = DCG si tous les pertinents étaient en tête de liste
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0

    k = k or len(retrieved)
    retrieved_k = retrieved[:k]

    # DCG : gain réel
    dcg = sum((1.0 / math.log2(i + 2)) for i, r in enumerate(retrieved_k) if r in relevant_set)

    # IDCG : gain idéal (tous les pertinents en tête)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return round(dcg / idcg, 3) if idcg > 0 else 0.0


def _f1_retriever(precision: float, recall: float) -> float:
    """F1 harmonique entre precision et recall du retriever."""
    if (precision + recall) == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


# ---------------------------------------------------------------------------
# Normalisation des aspects ABSA
# ---------------------------------------------------------------------------

# Mapping utilitaire pour normaliser les aspects ABSA vers le vocabulaire du benchmark
# (le module ABSA peut retourner des variantes légèrement différentes)
ASPECT_NORMALIZE = {
    "earnings": "earnings",
    "revenu": "revenue",
    "revenue": "revenue",
    "revenues": "revenue",
    "guidance": "guidance",
    "product launch": "product_launch",
    "product_launch": "product_launch",
    "macro": "macro_exposure",
    "macro_exposure": "macro_exposure",
    "leadership": "leadership",
    "management": "leadership",
}


def _normalize_aspects(aspects: list[str]) -> list[str]:
    normalized = []
    for a in aspects:
        key = a.lower().strip()
        normalized.append(ASPECT_NORMALIZE.get(key, key))
    return normalized


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


def run_retriever_evaluation(limit: int = 0) -> dict:
    """
    Évalue le module ABSA comme Retriever financier sur le benchmark annoté.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 7b : Evaluation du Retriever (module ABSA)")
    print(f"{'=' * 70}")

    # Chargement du benchmark
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    articles = [
        a
        for a in dataset["articles"]
        if a["ground_truth"]["filtrage"] == "pertinent" and len(a["ground_truth"]["key_aspects"]) > 0
    ]
    if limit > 0:
        articles = articles[:limit]

    print(f"\n  Articles a evaluer   : {len(articles)}")

    from src.agents.agent_absa import run_absa

    precisions = []
    recalls = []
    ndcgs = []
    f1s = []
    details = []

    for art in tqdm(articles, desc="  Evaluation ABSA", ncols=80):
        content = art["content"]
        ticker = art["ticker"]
        expected = art["ground_truth"]["key_aspects"]  # GT

        try:
            absa_result = run_absa(content)
            # L'ABSA retourne une liste d'aspects avec sentiments
            retrieved_raw = [item["aspect"] for item in absa_result.get("aspects", [])]
            retrieved = _normalize_aspects(retrieved_raw)
        except Exception as e:
            logger.warning("ABSA error on %s: %s", art["id"], e)
            retrieved = []

        prec = _context_precision(retrieved, expected)
        rec = _context_recall(retrieved, expected)
        ndcg = _ndcg(retrieved, expected)
        f1 = _f1_retriever(prec, rec)

        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)
        f1s.append(f1)

        details.append(
            {
                "id": art["id"],
                "ticker": ticker,
                "expected": expected,
                "retrieved": retrieved,
                "precision": prec,
                "recall": rec,
                "ndcg": ndcg,
                "f1": f1,
            }
        )

    # Agrégation
    n = len(details) or 1
    avg_prec = round(sum(precisions) / n, 3)
    avg_rec = round(sum(recalls) / n, 3)
    avg_ndcg = round(sum(ndcgs) / n, 3)
    avg_f1 = round(sum(f1s) / n, 3)

    # Affichage
    print(f"\n{'=' * 70}")
    print(f"RESULTATS — {len(articles)} articles evalues")
    print(f"{'=' * 70}")
    print(f"\n  Context Precision  : {avg_prec:.3f}  ({'bon' if avg_prec >= 0.7 else 'a ameliorer'})")
    print(f"  Context Recall     : {avg_rec:.3f}  ({'bon' if avg_rec >= 0.7 else 'a ameliorer'})")
    print(f"  NDCG               : {avg_ndcg:.3f}  ({'bon' if avg_ndcg >= 0.7 else 'a ameliorer'})")
    print(f"  F1 Retriever       : {avg_f1:.3f}")

    # Diagnostic automatique
    print("\n  Diagnostic :")
    if avg_rec < 0.5:
        print("  [!] RECALL FAIBLE : l'ABSA manque des aspects critiques (bruit = agents sous-informés).")
    if avg_prec < 0.5:
        print("  [!] PRECISION FAIBLE : l'ABSA injecte du bruit (aspects hors-sujet polluent le débat).")
    if avg_ndcg < 0.5:
        print(
            "  [!] NDCG FAIBLE : les aspects pertinents apparaissent en fin de liste "
            "(le premier argument reçu par les agents est moins pertinent)."
        )
    if avg_rec >= 0.7 and avg_prec >= 0.7:
        print("  [OK] Le module ABSA récupère correctement les aspects financiers clés.")

    # Pires cas
    worst = sorted(details, key=lambda x: x["f1"])[:3]
    if worst and worst[0]["f1"] < avg_f1:
        print("\n  Pires cas (F1 Retriever) :")
        for w in worst:
            print(f"    [{w['ticker']}] {w['id']:20s} P={w['precision']:.2f} R={w['recall']:.2f} F1={w['f1']:.2f}")
            print(f"      Attendu   : {w['expected']}")
            print(f"      Retrouve  : {w['retrieved']}")

    print("=" * 70)

    # Sauvegarde des résultats
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(__file__).parent / "eval_results" / f"{timestamp}_retriever_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "metrics": {
            "context_precision": avg_prec,
            "context_recall": avg_rec,
            "ndcg": avg_ndcg,
            "f1_retriever": avg_f1,
            "n_evaluated": len(articles),
        },
        "details": details,
    }

    with open(out_dir / "retriever_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats complets sauvegardés dans : {out_dir}/retriever_metrics.json")

    return {
        "sub": "retriever",
        "n_evaluated": len(articles),
        "context_precision": avg_prec,
        "context_recall": avg_rec,
        "ndcg": avg_ndcg,
        "f1_retriever": avg_f1,
    }


if __name__ == "__main__":
    run_retriever_evaluation()
