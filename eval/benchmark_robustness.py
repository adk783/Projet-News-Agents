"""
benchmark_robustness.py -- Couche 4 : Tests de robustesse de l'architecture
=============================================================================
Évalue les 4 propriétés de robustesse avancée :

  TEST 1 — ANONYMISATION (Biais de connaissance implicite)
    Objectif : Mesurer si le signal change quand les noms de l'entreprise
               sont masqués. Un signal identique = pas de biais implicite.
    Sortie   : Taux de biais détecté (idéal = 0%)

  TEST 2 — BARRIÈRE TEMPORELLE (Intégrité du backtesting)
    Objectif : Vérifier que le contexte marché transmis en évaluation
               correspond bien à la date de l'article (pas d'aujourd'hui).
    Sortie   : Nb d'articles avec contamination temporelle détectée.

  TEST 3 — STRESS (Robustesse aux inputs dégradés)
    Objectif : Tester la stabilité des signaux face à des articles
               tronqués, bruités, ou sémantiquement inversés.
    Sortie   : Score de stabilité (0-1) et taux d'inversions correctes.

  TEST 4 — REPRODUCTIBILITÉ (Variance stochastique)
    Objectif : Mesurer la variance des décisions sur N runs identiques.
    Sortie   : Variance moyenne et proportion de signaux stables.

Lancé via : python eval/run_eval.py --layer 4 [--n-articles 3] [--n-runs 3]
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Robustness")
logger.setLevel(logging.INFO)

DATASET_PATH = Path(__file__).parent / "benchmark_dataset.json"


def _load_dataset(n_articles: int = 3) -> list[dict]:
    """Charge N articles du benchmark (pertinents uniquement, avec signal connu)."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    articles = [
        a
        for a in data["articles"]
        if a["ground_truth"]["filtrage"] == "pertinent" and a["ground_truth"]["signal"] is not None
    ]
    return articles[:n_articles]


# ---------------------------------------------------------------------------
# TEST 1 : Anonymisation — Détection du biais de connaissance implicite
# ---------------------------------------------------------------------------


def test_anonymization(articles: list[dict]) -> dict:
    """
    Pour chaque article, exécute le débat avec et sans anonymisation.
    Détecte si les signaux divergent (= le modèle utilisait son prior implicite).
    """
    from src.agents.agent_debat import workflow_debat_actualite
    from src.utils.anonymizer import anonymize_article, compute_bias_score

    print("\n  [TEST 1] Anonymisation")
    results = []

    for article in articles:
        ticker = article["ticker"]
        content = f"{article['title']}\n\n{article['content']}"
        gt = article["ground_truth"]["signal"]
        print(f"    {article['id']} ({ticker})...", end=" ", flush=True)

        t0 = time.time()
        try:
            # Run sans anonymisation
            decision_orig = workflow_debat_actualite.invoke(
                {
                    "texte_article": content,
                    "ticker_symbol": ticker,
                    "contexte_marche": {},
                    "absa_result": {"aspects": []},
                }
            )
            signal_orig = decision_orig.get("signal", "?")

            # Run avec anonymisation
            anon = anonymize_article(content, ticker)
            decision_anon = workflow_debat_actualite.invoke(
                {
                    "texte_article": anon.text,
                    "ticker_symbol": anon.ticker_alias,
                    "contexte_marche": {},
                    "absa_result": {"aspects": []},
                }
            )
            signal_anon = decision_anon.get("signal", "?")

            bias = compute_bias_score(signal_orig, signal_anon)
            lat = round(time.time() - t0, 1)

            status = "BIAIS" if bias["biased"] else "OK"
            print(f"[{status}] orig={signal_orig} anon={signal_anon} ({lat}s)")

            results.append(
                {
                    "id": article["id"],
                    "ticker": ticker,
                    "gt_signal": gt,
                    "signal_original": signal_orig,
                    "signal_anonymized": signal_anon,
                    "biased": bias["biased"],
                    "entities_masked": anon.entities_replaced,
                    "latency_s": lat,
                }
            )

        except Exception as e:
            print(f"[ERR] {e}")
            results.append({"id": article["id"], "error": str(e), "biased": False})

    n_biased = sum(1 for r in results if r.get("biased"))
    bias_rate = n_biased / len(results) if results else 0.0

    print(f"\n    Taux de biais detecte : {bias_rate:.0%} ({n_biased}/{len(results)})")
    if bias_rate > 0.25:
        print("    ATTENTION : Plus de 25% des articles montrent un biais implicite.")
        print("    Les agents s'appuient partiellement sur leur prior (connaissance Apple, Tesla...)")
    else:
        print("    Les agents raisonnent principalement depuis l'article. Biais faible.")

    return {
        "test": "anonymization",
        "bias_rate": round(bias_rate, 3),
        "n_biased": n_biased,
        "n_total": len(results),
        "details": results,
    }


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TEST 2 : Intégrité temporelle — Backtesting propre
# ---------------------------------------------------------------------------


def test_temporal_integrity(articles: list[dict]) -> dict:
    """
    Vérifie que le contexte marché utilisé during eval est cohérent
    avec la date de l'article et non la date d'aujourd'hui.
    """
    from src.utils.temporal_fence import check_survivor_bias, get_clean_eval_context, validate_context_temporality

    print("\n  [TEST 2] Integrite temporelle")
    results = []
    contaminated = 0

    for article in articles:
        ticker = article["ticker"]
        date_str = article.get("date", "")
        if not date_str:
            print(f"    {article['id']} : PAS DE DATE - ignoré")
            continue

        # Contexte PROPRE (via temporal fence - données historiques réelles)
        clean_ctx, integrity_clean = get_clean_eval_context(ticker, date_str, check_bias=False)

        # Contexte LIVE (comme si on appelait yfinance aujourd'hui - possiblement futur)
        integrity_live = validate_context_temporality(
            context={"current_price": 999.0},  # Faux prix actuel simulé
            article_date_str=date_str,
            memory_injected=False,
        )

        status_clean = "OK" if integrity_clean.is_temporally_valid else "CONTAMINE"

        # Vérification du biais de survie
        bias = check_survivor_bias(ticker, date_str)

        bias_str = f"Bias={bias.bias_risk}"
        delisted = " [DELISTE?]" if bias.is_potentially_delisted else ""

        print(
            f"    {article['id']} {date_str} | Fence: {status_clean} | "
            f"Live: {integrity_live.days_contaminated}j | {bias_str}{delisted}"
        )

        if not integrity_clean.is_temporally_valid:
            contaminated += 1

        results.append(
            {
                "id": article["id"],
                "ticker": ticker,
                "date": date_str,
                "has_hist_context": bool(clean_ctx),
                "days_contaminated_if_live": integrity_live.days_contaminated,
                "fence_valid": integrity_clean.is_temporally_valid,
                "warnings": integrity_clean.warnings,
                # Biais de survie
                "survivor_bias_risk": bias.bias_risk if bias else "INCONNU",
                "is_potentially_delisted": bias.is_potentially_delisted if bias else False,
                "price_at_article": bias.price_at_article if bias else None,
                "price_today": bias.price_today if bias else None,
                "bias_notes": bias.notes if bias else [],
            }
        )

    live_contamination_days = [r["days_contaminated_if_live"] for r in results]
    avg_contamination = sum(live_contamination_days) / len(live_contamination_days) if live_contamination_days else 0

    # Résumé biais de survie
    n_delisted = sum(1 for r in results if r["is_potentially_delisted"])
    n_bias_eleve = sum(1 for r in results if r["survivor_bias_risk"] == "ELEVE")
    n_bias_modere = sum(1 for r in results if r["survivor_bias_risk"] == "MODERE")

    print(f"\n    Articles avec contexte propre : {len(results) - contaminated}/{len(results)}")
    print(f"    Contamination moyenne si live : {avg_contamination:.0f} jours dans le futur")

    print("\n    [Biais de Survie]")
    print(f"    Tickers potentiellement délités : {n_delisted}/{len(results)}")
    print(f"    Risque ELEVE  : {n_bias_eleve} ticker(s)")
    print(f"    Risque MODERE : {n_bias_modere} ticker(s)")

    if n_delisted > 0:
        delisted_list = [r["ticker"] for r in results if r["is_potentially_delisted"]]
        print(f"    !! Tickers suspects : {', '.join(set(delisted_list))}")
        print("    !! Ces tickers DOIVENT être exclus de l'évaluation des performances.")
    elif n_bias_eleve == 0 and n_bias_modere == 0:
        print("    OK Aucun biais de survie détecté sur ce sous-ensemble.")

    print("\n    Conclusion : Utiliser temporal_fence.get_clean_eval_context() en eval.")

    return {
        "test": "temporal_integrity",
        "n_valid_with_fence": len(results) - contaminated,
        "n_total": len(results),
        "avg_contamination_days": round(avg_contamination, 0),
        "n_potentially_delisted": n_delisted,
        "n_bias_eleve": n_bias_eleve,
        "n_bias_modere": n_bias_modere,
        "details": results,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_stress_robustness(articles: list[dict], n_articles: int = 2) -> dict:
    """
    Génère les variants de stress pour N articles et lance le pipeline sur chacun.
    Mesure la stabilité des signaux sous dégradation.
    """
    from src.agents.agent_debat import workflow_debat_actualite
    from src.utils.stress_generator import compute_stability_score, generate_stress_variants

    print("\n  [TEST 3] Robustesse au stress")
    all_results = []

    stress_articles = articles[:n_articles]
    for article in stress_articles:
        ticker = article["ticker"]
        content = f"{article['title']}\n\n{article['content']}"
        gt = article["ground_truth"]["signal"]
        print(f"\n    Article {article['id']} (signal attendu: {gt})")

        variants = generate_stress_variants(content, article["id"], gt)
        variant_signals = []

        for v in variants:
            print(f"      [{v.stress_type}/{v.severity}] {v.description[:50]}...", end=" ", flush=True)
            t0 = time.time()
            try:
                decision = workflow_debat_actualite.invoke(
                    {
                        "texte_article": v.text,
                        "ticker_symbol": ticker,
                        "contexte_marche": {},
                        "absa_result": {"aspects": []},
                    }
                )
                signal_stressed = decision.get("signal", "?")
                lat = round(time.time() - t0, 1)
                print(f"-> {signal_stressed} (expected: {v.expected_shift}) ({lat}s)")
                variant_signals.append((v.stress_type, v.severity, signal_stressed))
            except Exception as e:
                print(f"ERR: {e}")
                variant_signals.append((v.stress_type, v.severity, None))

        stability = compute_stability_score(gt, variant_signals)
        print(f"    Score de stabilite : {stability['stability_score']:.0%} | {stability['verdict']}")
        all_results.append(
            {
                "id": article["id"],
                "original_signal": gt,
                "stability_score": stability["stability_score"],
                "verdict": stability["verdict"],
                "tests": stability["tests"],
            }
        )

    avg_stability = sum(r["stability_score"] for r in all_results) / len(all_results) if all_results else 0
    print(f"\n    Stabilite moyenne : {avg_stability:.0%}")

    return {
        "test": "stress_robustness",
        "avg_stability": round(avg_stability, 3),
        "n_articles": len(all_results),
        "details": all_results,
    }


# ---------------------------------------------------------------------------
# TEST 4 : Reproductibilité stochastique
# ---------------------------------------------------------------------------


def test_reproducibility(articles: list[dict], n_runs: int = 3, n_articles: int = 2) -> dict:
    """
    Lance le pipeline N fois sur le même article et mesure la variance inter-run.
    """
    from src.utils.consensus_voter import format_vote_report, run_with_majority_vote

    print(f"\n  [TEST 4] Reproductibilite ({n_runs} runs par article)")
    all_results = []

    repro_articles = articles[:n_articles]
    for article in repro_articles:
        ticker = article["ticker"]
        content = f"{article['title']}\n\n{article['content']}"
        gt = article["ground_truth"]["signal"]
        print(f"\n    Article {article['id']} (signal attendu: {gt})")

        t0 = time.time()
        try:
            vote = run_with_majority_vote(
                article_text=content,
                ticker=ticker,
                contexte_marche={},
                absa_result={"aspects": []},
                n_runs=n_runs,
            )
            lat = round(time.time() - t0, 1)
            print(format_vote_report(vote))
            print(f"    Temps total : {lat}s")

            all_results.append(
                {
                    "id": article["id"],
                    "gt_signal": gt,
                    "winner": vote.signal,
                    "confidence": vote.confidence,
                    "variance": vote.variance,
                    "stable": vote.stable,
                    "vote_breakdown": vote.vote_breakdown,
                    "correct": vote.signal == gt,
                }
            )
        except Exception as e:
            print(f"    ERR: {e}")
            all_results.append({"id": article["id"], "error": str(e), "stable": False})

    n_stable = sum(1 for r in all_results if r.get("stable", False))
    avg_variance = sum(r.get("variance", 1.0) for r in all_results) / len(all_results) if all_results else 1.0

    print(f"\n    Signaux stables     : {n_stable}/{len(all_results)}")
    print(f"    Variance moyenne    : {avg_variance:.0%}")
    if avg_variance > 0.33:
        print("    ATTENTION : Haute variance — les decisions sont peu reproductibles.")
        print("    Envisager : temperature=0 sur le modele consensus, ou augmenter n_runs.")
    else:
        print("    Bonne reproductibilite — les signaux sont stables sur plusieurs runs.")

    return {
        "test": "reproducibility",
        "n_stable": n_stable,
        "n_total": len(all_results),
        "avg_variance": round(avg_variance, 3),
        "details": all_results,
    }


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_robustness_benchmark(
    n_articles: int = 3,
    n_runs: int = 3,
    tests: list[str] | None = None,
    save_results: bool = True,
) -> dict:
    """
    Lance le benchmark de robustesse complet (Couche 4).

    Args:
        n_articles : Nb d'articles à tester (entre 2 et 5 recommandé)
        n_runs     : Nb de runs pour le test de reproductibilité
        tests      : Liste des tests à exécuter (None = tous)
                     Options: ["anonymization", "temporal", "stress", "reproducibility"]
    """
    if not DATASET_PATH.exists():
        print("[ERREUR] benchmark_dataset.json introuvable.")
        return {}

    articles = _load_dataset(n_articles)
    tests = tests or ["anonymization", "temporal", "stress", "reproducibility"]

    print(f"\n{'=' * 70}")
    print(f"COUCHE 4 : Benchmark Robustesse ({len(articles)} articles, {n_runs} runs/article)")
    print(f"Tests : {', '.join(tests)}")
    print(f"{'=' * 70}")

    all_results = {"layer": 4}

    if "anonymization" in tests:
        all_results["anonymization"] = test_anonymization(articles)

    if "temporal" in tests:
        all_results["temporal"] = test_temporal_integrity(articles)

    if "stress" in tests:
        all_results["stress"] = test_stress_robustness(articles, n_articles=min(2, n_articles))

    if "reproducibility" in tests:
        all_results["reproducibility"] = test_reproducibility(articles, n_runs=n_runs, n_articles=min(2, n_articles))

    # Résumé
    print(f"\n{'=' * 70}")
    print("RESUME COUCHE 4")
    print(f"{'=' * 70}")

    bias = all_results.get("anonymization", {})
    temp = all_results.get("temporal", {})
    stress = all_results.get("stress", {})
    repro = all_results.get("reproducibility", {})

    if bias:
        print(f"  Anonymisation  : Biais detete sur {bias.get('bias_rate', 0):.0%} des articles")
    if temp:
        print(f"  Temporalite    : {temp.get('avg_contamination_days', 0):.0f}j de contamination moyenne si live")
    if stress:
        print(f"  Stress tests   : Stabilite moyenne {stress.get('avg_stability', 0):.0%}")
    if repro:
        print(
            f"  Reproductibilite : Variance moyenne {repro.get('avg_variance', 1.0):.0%}, "
            f"{repro.get('n_stable', 0)}/{repro.get('n_total', 0)} signaux stables"
        )

    print("=" * 70)

    # Sauvegarde
    if save_results:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = Path(__file__).parent / "eval_results" / f"{ts}_robustness_layer4"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Resultats sauvegardes dans : {out}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-articles", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--tests", nargs="+", choices=["anonymization", "temporal", "stress", "reproducibility"], default=None
    )
    args = parser.parse_args()
    run_robustness_benchmark(n_articles=args.n_articles, n_runs=args.n_runs, tests=args.tests)
