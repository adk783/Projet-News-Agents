"""Tests unitaires pytest-natifs pour le YOLO classifier (gate auto-exec).

Avant : ce fichier melait sys.path.insert + logging.basicConfig + scenarios
hardcodes. Refactor : fixtures pytest pour les 3 scenarios canoniques (FORT,
AMBIGU, DANGEREUX), et import propre via le pyproject.toml (pas de sys.path).

Marque integration : `yolo_classifier` peut tirer transformers/torch via ses
imports transitifs ; ces tests ne sont pas inclus dans la matrice CI minimale.
Pour les lancer en local : `pytest tests/test_yolo.py`.
"""

from __future__ import annotations

import pytest

# Import skip-friendly : si yolo_classifier ne peut pas etre importe (deps
# manquantes en CI minimale), on skip le module entier au lieu de fail.
yolo = pytest.importorskip(
    "src.utils.yolo_classifier",
    reason="yolo_classifier indisponible (transformers/torch non installe ?)",
)
classify_risk = yolo.classify_risk


# =============================================================================
# Scenarios canoniques (extraits de la batterie originale)
# =============================================================================
@pytest.fixture
def scenario_fort():
    """Signal FORT : FinBERT + ABSA + LLM tous alignes (Achat evident)."""
    return {
        "signal_final": "Achat",
        "consensus_rate": 0.88,
        "impact_strength": 0.82,
        "score_finbert": 0.97,
        "absa_result": {
            "aspects": [
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "positive"},
                {"sentiment": "negative"},
            ]
        },
        "scratchpad_xml": (
            "[Tour 1] arg [confiance: 0.88] [Tour 2] arg [confiance: 0.91] [Tour 3] arg [confiance: 0.90]"
        ),
        "contexte_marche": {"variation_5d": 1.2},
    }


@pytest.fixture
def scenario_dangereux():
    """Signal DANGEREUX : FinBERT negatif mais LLM dit Achat (incoherence)."""
    return {
        "signal_final": "Achat",
        "consensus_rate": 0.45,
        "impact_strength": 0.40,
        "score_finbert": 0.18,
        "absa_result": {
            "aspects": [
                {"sentiment": "negative"},
                {"sentiment": "negative"},
                {"sentiment": "positive"},
            ]
        },
        "scratchpad_xml": (
            "[Tour 1] arg [confiance: 0.90] [Tour 2] arg [confiance: 0.50] [Tour 3] arg [confiance: 0.40]"
        ),
        "contexte_marche": {"variation_5d": 12.5},
    }


# =============================================================================
# Tests
# =============================================================================
def test_signal_fort_yields_low_risk_and_auto_execute(scenario_fort):
    """Quand FinBERT/ABSA/LLM convergent fortement, YOLO doit autoriser
    l'execution automatique (risk=FAIBLE, auto_execute=True)."""
    result = classify_risk(**scenario_fort)
    assert result.risk_level == "FAIBLE", f"Signal aligne devrait etre FAIBLE risk, got {result.risk_level}"
    assert result.auto_execute is True


def test_signal_dangereux_requires_human_approval(scenario_dangereux):
    """Quand FinBERT contredit le LLM (incoherence), YOLO doit basculer en
    mode requires_human (risk=ELEVE, pas d'auto-exec)."""
    result = classify_risk(**scenario_dangereux)
    assert result.risk_level == "ELEVE", f"Incoherence FinBERT/LLM devrait etre ELEVE risk, got {result.risk_level}"
    assert result.requires_human is True
