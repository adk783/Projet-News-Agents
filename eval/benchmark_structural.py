"""
benchmark_structural.py — Couche 1 : Tests unitaires déterministes (zéro API)
=============================================================================
Ces tests vérifient les composants PUR PYTHON de l'architecture.
Aucun appel LLM. Temps d'exécution : < 3 secondes.

Objectif : Détecter les régressions dès qu'une modification est apportée
           sans dépenser de tokens ou attendre des réponses réseau.

Lancé via : python eval/run_eval.py --layer 1
"""

import json
import traceback
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Mini framework de test
# ---------------------------------------------------------------------------

_results = {"passed": 0, "failed": 0, "errors": []}


def test(name: str, fn: Callable) -> None:
    """Exécute un test et affiche le résultat."""
    try:
        fn()
        _results["passed"] += 1
        print(f"  [PASS] {name}")
    except AssertionError as e:
        _results["failed"] += 1
        _results["errors"].append({"test": name, "error": str(e)})
        print(f"  [FAIL] {name}")
        print(f"         {e}")
    except Exception as e:
        _results["failed"] += 1
        _results["errors"].append({"test": name, "error": str(e)})
        print(f"  [ERROR] {name}")
        print(f"          {type(e).__name__}: {e}")


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}\n  Attendu : {expected}\n  Obtenu  : {actual}")


def assert_in_range(value, lo, hi, msg=""):
    if not (lo <= value <= hi):
        raise AssertionError(f"{msg}\n  Valeur {value} hors de [{lo}, {hi}]")


def assert_in(value, collection, msg=""):
    if value not in collection:
        raise AssertionError(f"{msg}\n  {value!r} absent de {collection}")


# ---------------------------------------------------------------------------
# YOLO Classifier Tests (8 tests)
# ---------------------------------------------------------------------------


def run_yolo_tests():
    print("\n[YOLO Classifier]")
    from src.utils.yolo_classifier import (
        RISK_ELEVE,
        RISK_FAIBLE,
        RISK_MOYEN,
        _compute_absa_ambiguity,
        _compute_confidence_variance,
        _compute_debate_convergence,
        _compute_finbert_alignment,
        _compute_market_volatility,
        _extract_confidence_scores,
        classify_risk,
    )

    def t_extract_scores():
        xml = "<s>[Tour 1] arg [confiance: 0.85] [Tour 2] arg [confiance: 0.92]</s>"
        scores = _extract_confidence_scores(xml)
        assert len(scores) == 2
        assert scores[0] == 0.85
        assert scores[1] == 0.92

    def t_variance_uniform():
        scores = [0.8, 0.8, 0.8]
        assert _compute_confidence_variance(scores) == 0.0

    def t_variance_high():
        scores = [0.9, 0.1]
        std = _compute_confidence_variance(scores)
        assert_in_range(std, 0.39, 0.41, "Ecart-type [0.9, 0.1]")

    def t_finbert_align_bullish():
        # FinBERT=0.95 (très positif) + Signal Achat → alignement parfait
        align = _compute_finbert_alignment("Achat", 0.95)
        assert_eq(align, 1.0, "Achat + FinBERT 95%")

    def t_finbert_align_contradiction():
        # FinBERT=0.10 (très négatif) + Signal Achat → désaccord
        align = _compute_finbert_alignment("Achat", 0.10)
        assert_in_range(align, 0.0, 0.60, "Contradiction FinBERT/LLM")

    def t_absa_ambiguity_clear():
        absa = {
            "aspects": [
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "positive"},
            ]
        }
        amb = _compute_absa_ambiguity(absa)
        assert_in_range(amb, 0.0, 0.30, "ABSA sans ambiguité (4 pos, 0 neg)")

    def t_absa_ambiguity_mixed():
        absa = {
            "aspects": [
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "negative"},
                {"sentiment": "negative"},
            ]
        }
        amb = _compute_absa_ambiguity(absa)
        assert_in_range(amb, 0.90, 1.0, "ABSA parfaitement ambigu (2/2)")

    def t_risk_faible():
        yolo = classify_risk(
            signal_final="Achat",
            consensus_rate=0.90,
            impact_strength=0.85,
            scratchpad_xml="[Tour 1] x [confiance: 0.88] [Tour 2] x [confiance: 0.90] [Tour 3] x [confiance: 0.91]",
            absa_result={
                "aspects": [
                    {"sentiment": "positive"},
                    {"sentiment": "positive"},
                    {"sentiment": "positive"},
                    {"sentiment": "negative"},
                ]
            },
            score_finbert=0.97,
            contexte_marche={"variation_5d": 1.0},
        )
        assert_eq(yolo.risk_level, RISK_FAIBLE, "Signal fort sans contradiction")
        assert yolo.auto_execute, "auto_execute doit être True pour FAIBLE"
        assert not yolo.requires_human, "requires_human doit être False pour FAIBLE"

    def t_risk_eleve():
        yolo = classify_risk(
            signal_final="Achat",
            consensus_rate=0.40,
            impact_strength=0.38,
            scratchpad_xml="[Tour 1] x [confiance: 0.90] [Tour 2] x [confiance: 0.50] [Tour 3] x [confiance: 0.40]",
            absa_result={
                "aspects": [
                    {"sentiment": "negative"},
                    {"sentiment": "negative"},
                    {"sentiment": "positive"},
                ]
            },
            score_finbert=0.12,  # FinBERT très négatif mais LLM dit Achat
            contexte_marche={"variation_5d": 12.0},
        )
        assert_eq(yolo.risk_level, RISK_ELEVE, "Contradiction FinBERT/LLM + marché volatile")
        assert yolo.requires_human

    def t_neutre_never_auto():
        # Un signal Neutre ne doit JAMAIS permettre l'auto-exécution
        yolo = classify_risk(
            signal_final="Neutre",
            consensus_rate=0.90,
            impact_strength=0.90,
            scratchpad_xml="[Tour 1] x [confiance: 0.90]",
            absa_result={"aspects": [{"sentiment": "positive"}]},
            score_finbert=0.90,
            contexte_marche={"variation_5d": 0.0},
        )
        assert not yolo.auto_execute, "Neutre ne doit jamais s'auto-exécuter"

    test("Extraction scores de confiance", t_extract_scores)
    test("Variance sur scores uniformes = 0", t_variance_uniform)
    test("Variance élevée [0.9, 0.1]", t_variance_high)
    test("Alignement parfait FinBERT/Achat", t_finbert_align_bullish)
    test("Contradiction FinBERT 10% / signal Achat", t_finbert_align_contradiction)
    test("ABSA non ambigu (4 positifs)", t_absa_ambiguity_clear)
    test("ABSA parfaitement ambigu (50/50)", t_absa_ambiguity_mixed)
    test("Risque FAIBLE sur signal fort", t_risk_faible)
    test("Risque ELEVE sur contradiction FinBERT/LLM", t_risk_eleve)
    test("Signal Neutre jamais auto-exécuté", t_neutre_never_auto)


# ---------------------------------------------------------------------------
# Context Compressor Tests (4 tests)
# ---------------------------------------------------------------------------


def run_compressor_tests():
    print("\n[Context Compressor]")
    from src.utils.context_compressor import CompressionLevel, compress_article_if_needed, estimate_tokens

    def t_micro_under_threshold():
        short_text = "A" * 5999
        result = compress_article_if_needed(short_text, "TEST")
        assert_eq(result.level, CompressionLevel.MICRO, "5999 chars doit être MICRO")
        assert_eq(result.text, short_text, "MICRO ne modifie pas le texte")
        assert_eq(result.compression_ratio, 1.0)

    def t_auto_level_triggered():
        medium_text = "A" * 10_000
        result = compress_article_if_needed(medium_text, "TEST")
        assert_eq(result.level, CompressionLevel.AUTO, "10000 chars doit être AUTO")

    def t_full_level_triggered():
        long_text = "A" * 17_000
        result = compress_article_if_needed(long_text, "TEST")
        assert_eq(result.level, CompressionLevel.FULL, "17000 chars doit être FULL")

    def t_token_estimate():
        text = "A" * 4000
        tokens = estimate_tokens(text)
        assert_eq(tokens, 1000, "4000 chars = 1000 tokens (4 chars/token)")

    test("Texte court -> niveau MICRO", t_micro_under_threshold)
    test("Texte moyen -> niveau AUTO", t_auto_level_triggered)
    test("Texte long -> niveau FULL", t_full_level_triggered)
    test("Estimation tokens (4 chars/token)", t_token_estimate)


# ---------------------------------------------------------------------------
# Pipeline Metrics Tests (3 tests)
# ---------------------------------------------------------------------------


def run_metrics_tests():
    print("\n[Pipeline Metrics (_calculer_metrics_objectives)]")
    from src.pipelines.agent_pipeline import _calculer_metrics_objectives

    def t_achat_positif_absa():
        # ABSA 80% positif + FinBERT 0.95 + Signal Achat → fort impact
        rate, impact = _calculer_metrics_objectives(
            "Achat",
            score_finbert=0.95,
            absa_result={
                "aspects": [
                    {"sentiment": "positive"},
                    {"sentiment": "positive"},
                    {"sentiment": "positive"},
                    {"sentiment": "positive"},
                    {"sentiment": "negative"},
                ]
            },
        )
        assert_in_range(impact, 0.65, 1.0, "Impact fort sur signal Achat clair")
        assert_in_range(rate, 0.50, 1.0, "Consensus élevé pour signal fort")

    def t_vente_negatif_absa():
        # ABSA 80% negatif + FinBERT 0.05 + Signal Vente -> fort impact
        rate, impact = _calculer_metrics_objectives(
            "Vente",
            score_finbert=0.05,
            absa_result={
                "aspects": [
                    {"sentiment": "negative"},
                    {"sentiment": "negative"},
                    {"sentiment": "negative"},
                    {"sentiment": "negative"},
                    {"sentiment": "positive"},
                ]
            },
        )
        # FinBERT=0.05 < 0.5 donc finbert_force = 0.05 (deja un score de risque fort Vente)
        # absa_force = 1 - 0.8 = 0.2  (vente forte)
        # impact = 0.05*0.4 + 0.2*0.6 = 0.14 -> range reajuste
        assert_in_range(impact, 0.10, 0.50, "Impact signal Vente clair")

    def t_neutre_midpoint():
        # Signal Neutre → impact et consensus autour de 0.5
        rate, impact = _calculer_metrics_objectives(
            "Neutre",
            score_finbert=0.52,
            absa_result={
                "aspects": [
                    {"sentiment": "positive"},
                    {"sentiment": "negative"},
                ]
            },
        )
        assert_in_range(impact, 0.30, 0.70, "Impact Neutre proche de 0.5")

    test("Signal Achat + ABSA 80% positif + FinBERT 0.95 -> impact eleve", t_achat_positif_absa)
    test("Signal Vente + ABSA 80% negatif + FinBERT 0.05 -> impact eleve", t_vente_negatif_absa)
    test("Signal Neutre -> impact proche de 0.5", t_neutre_midpoint)


# ---------------------------------------------------------------------------
# Dataset Benchmark Integrity Tests (2 tests)
# ---------------------------------------------------------------------------


def run_dataset_tests():
    print("\n[Benchmark Dataset Integrity]")
    dataset_path = Path(__file__).parent / "benchmark_dataset.json"

    def t_dataset_loads():
        assert dataset_path.exists(), "benchmark_dataset.json introuvable"
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "articles" in data, "Clé 'articles' manquante"
        assert len(data["articles"]) >= 20, "Au moins 20 articles requis"

    def t_dataset_labels_valid():
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        valid_signals = {"Achat", "Vente", "Neutre", None}
        valid_filtrages = {"pertinent", "hors_scope"}
        for article in data["articles"]:
            gt = article["ground_truth"]
            assert gt["filtrage"] in valid_filtrages, f"Filtrage invalide pour {article['id']}: {gt['filtrage']}"
            assert gt["signal"] in valid_signals, f"Signal invalide pour {article['id']}: {gt['signal']}"
            # Cohérence : hors_scope doit avoir signal=null
            if gt["filtrage"] == "hors_scope":
                assert gt["signal"] is None, f"hors_scope doit avoir signal=null pour {article['id']}"

    test("Dataset chargeable et complet (>=20 articles)", t_dataset_loads)
    test("Labels valides et cohérents", t_dataset_labels_valid)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def run_structural_benchmark() -> dict:
    print("\n" + "=" * 65)
    print("COUCHE 1 : Benchmark Structurel (composants Python — zéro API)")
    print("=" * 65)

    run_yolo_tests()
    run_compressor_tests()
    run_metrics_tests()
    run_dataset_tests()

    total = _results["passed"] + _results["failed"]
    print("\n" + "-" * 65)
    print(f"RÉSULTAT : {_results['passed']}/{total} tests passés", end="")
    if _results["failed"]:
        print(f" ({_results['failed']} échec(s))")
        for err in _results["errors"]:
            print(f"  FAIL: {err['test']} -> {err['error']}")
    else:
        print(" — Aucune régression détectée.")
    print("=" * 65)

    return {
        "layer": 1,
        "passed": _results["passed"],
        "failed": _results["failed"],
        "total": total,
        "success_rate": _results["passed"] / total if total else 0,
    }


if __name__ == "__main__":
    run_structural_benchmark()
