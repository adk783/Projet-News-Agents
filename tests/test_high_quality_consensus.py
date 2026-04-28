"""Tests pour HighQualityConsensus (second avis sur decisions HIGH_RISK).

Couvre :
- is_enabled() reflete ENABLE_HIGH_QUALITY_CONSENSUS env var.
- should_invoke() heuristique : YOLO ELEVE ou audit MODERATE/SEVERE -> True.
- evaluate() parse correctement un JSON valide.
- Tolerance : NIM down / JSON invalide / signal invalide -> fallback to original.
- agrees_with_original calcule correctement.
- Confidence clamp dans [0.0, 1.0].
- Provider absent -> fallback sans crash.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.agents.agent_high_quality_consensus import (
    HighQualityConsensus,
    HighQualityVerdict,
)
from src.utils.llm_client import LLMClient


# =============================================================================
# Helper : payload JSON simulant la reponse Llama 405B
# =============================================================================
def _make_response(signal="Achat", conf=0.75, reason="Strong fundamentals."):
    return json.dumps(
        {
            "signal_final": signal,
            "confiance": conf,
            "raisonnement": reason,
        }
    )


# =============================================================================
# is_enabled
# =============================================================================
class TestIsEnabled:
    def test_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert HighQualityConsensus.is_enabled() is False

    def test_enabled_when_one(self):
        with patch.dict("os.environ", {"ENABLE_HIGH_QUALITY_CONSENSUS": "1"}):
            assert HighQualityConsensus.is_enabled() is True

    def test_disabled_when_other_value(self):
        with patch.dict("os.environ", {"ENABLE_HIGH_QUALITY_CONSENSUS": "true"}):
            assert HighQualityConsensus.is_enabled() is False


# =============================================================================
# should_invoke (heuristique de declenchement)
# =============================================================================
class TestShouldInvoke:
    """Le HQC ne doit s'invoquer que sur signaux HIGH_RISK."""

    def test_no_args_returns_false(self):
        assert HighQualityConsensus.should_invoke() is False

    def test_yolo_eleve_triggers(self):
        assert HighQualityConsensus.should_invoke(yolo_risk_level="ELEVE") is True

    def test_yolo_eleve_lowercase_triggers(self):
        """Robustesse : 'eleve' et 'ELEVE' equivalents."""
        assert HighQualityConsensus.should_invoke(yolo_risk_level="eleve") is True

    def test_yolo_faible_does_not_trigger(self):
        assert HighQualityConsensus.should_invoke(yolo_risk_level="FAIBLE") is False

    def test_yolo_moyen_does_not_trigger(self):
        """MOYEN n'est pas critique au point de justifier un second avis cher."""
        assert HighQualityConsensus.should_invoke(yolo_risk_level="MOYEN") is False

    def test_audit_severe_triggers(self):
        assert HighQualityConsensus.should_invoke(audit_severity="severe") is True

    def test_audit_moderate_triggers(self):
        assert HighQualityConsensus.should_invoke(audit_severity="moderate") is True

    def test_audit_minor_does_not_trigger(self):
        assert HighQualityConsensus.should_invoke(audit_severity="minor") is False

    def test_audit_clean_does_not_trigger(self):
        assert HighQualityConsensus.should_invoke(audit_severity="clean") is False

    def test_either_criterion_triggers(self):
        """OR logique : un seul critere suffit."""
        assert HighQualityConsensus.should_invoke(yolo_risk_level="FAIBLE", audit_severity="severe") is True


# =============================================================================
# evaluate() : parsing JSON et fallbacks
# =============================================================================
class TestEvaluate:
    """L'evaluation doit etre robuste : tout chemin d'erreur -> fallback."""

    def _make_hqc_with_stub(self, raw: str) -> HighQualityConsensus:
        """HQC avec client stub deterministe."""
        client = LLMClient(stub_response=raw)
        client._api_keys = {"nvidia_nim": "nvapi-test"}
        return HighQualityConsensus(client=client)

    def test_evaluate_agree_returns_same_signal(self):
        """Si HQC valide le signal original, agrees_with_original=True."""
        hqc = self._make_hqc_with_stub(_make_response("Achat", 0.78))
        verdict = hqc.evaluate(
            article_summary="AAPL beat estimates",
            scratchpad="...debate...",
            original_signal="Achat",
            original_confidence=0.6,
        )
        assert verdict.signal_final == "Achat"
        assert verdict.agrees_with_original is True
        assert verdict.confiance == 0.78  # HQC peut renforcer la confidence

    def test_evaluate_disagree_flags_disagreement(self):
        """Si HQC change le signal, agrees_with_original=False."""
        hqc = self._make_hqc_with_stub(_make_response("Vente", 0.7))
        verdict = hqc.evaluate(
            article_summary="x",
            scratchpad="x",
            original_signal="Achat",
            original_confidence=0.55,
        )
        assert verdict.signal_final == "Vente"
        assert verdict.agrees_with_original is False

    def test_no_nim_provider_falls_back_to_original(self):
        """Si pas de cle NIM, retourne directement l'original."""
        client = LLMClient(api_keys={})  # aucun provider
        hqc = HighQualityConsensus(client=client)
        verdict = hqc.evaluate(
            article_summary="x",
            scratchpad="x",
            original_signal="Achat",
            original_confidence=0.7,
            original_reasoning="initial reasoning",
        )
        assert verdict.signal_final == "Achat"
        assert verdict.confiance == 0.7
        assert verdict.agrees_with_original is True
        assert verdict.provider_used == "fallback"

    def test_invalid_json_falls_back_to_original(self):
        hqc = self._make_hqc_with_stub("This is not JSON")
        verdict = hqc.evaluate(
            article_summary="x",
            scratchpad="x",
            original_signal="Vente",
            original_confidence=0.65,
        )
        assert verdict.signal_final == "Vente"
        assert verdict.confiance == 0.65
        assert verdict.provider_used == "fallback"

    def test_invalid_signal_falls_back(self):
        """Signal hors {Achat, Vente, Neutre} -> fallback."""
        hqc = self._make_hqc_with_stub(_make_response(signal="STRONG_BUY", conf=0.8))
        verdict = hqc.evaluate(
            article_summary="x",
            scratchpad="x",
            original_signal="Achat",
            original_confidence=0.6,
        )
        # Signal invalide -> retourne l'original
        assert verdict.signal_final == "Achat"
        assert verdict.provider_used == "fallback"

    def test_confidence_clamped_to_unit_interval(self):
        """Si LLM rend conf=2.5, clamp a 1.0."""
        hqc = self._make_hqc_with_stub(_make_response(conf=2.5))
        verdict = hqc.evaluate(
            article_summary="x",
            scratchpad="x",
            original_signal="Achat",
            original_confidence=0.6,
        )
        assert verdict.confiance == 1.0

    def test_negative_confidence_clamped_to_zero(self):
        hqc = self._make_hqc_with_stub(_make_response(conf=-0.3))
        verdict = hqc.evaluate(
            article_summary="x",
            scratchpad="x",
            original_signal="Achat",
            original_confidence=0.6,
        )
        assert verdict.confiance == 0.0

    def test_llm_failure_falls_back(self):
        """Si AllProvidersFailedError, retourne l'original sans crash."""
        client = LLMClient(api_keys={"nvidia_nim": "nvapi-x"}, max_retries=1, backoff_base_sec=0.0)
        hqc = HighQualityConsensus(client=client)
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("network down")
            verdict = hqc.evaluate(
                article_summary="x",
                scratchpad="x",
                original_signal="Achat",
                original_confidence=0.7,
            )
            assert verdict.signal_final == "Achat"
            assert verdict.provider_used == "fallback"
