"""Tests pour le ReasoningAuditor (audit post-debat via modele 'thinking').

Couvre :
- is_enabled() reflete ENABLE_REASONING_AUDITOR env var.
- audit() parse correctement un JSON valide retourne par le LLM.
- Tolerance : JSON invalide / absent -> CLEAN par defaut (degrade gracieux).
- Severity invalide -> CLEAN.
- Fallacies inconnues -> filtrees silencieusement.
- confidence_adjustment clamp dans [-1.0, 0.0].
- AllProvidersFailedError -> retourne CLEAN, pas de crash.
- to_dict() : enums serialises en str.
- Scratchpad vide / trop court -> skip avec CLEAN.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent_reasoning_auditor import (
    AuditSeverity,
    FallacyType,
    ReasoningAudit,
    ReasoningAuditor,
)
from src.utils.llm_client import AllProvidersFailedError, LLMClient


# =============================================================================
# Helpers
# =============================================================================
def _make_audit_response(payload: dict) -> str:
    """Simule la reponse JSON du modele auditeur."""
    import json

    return json.dumps(payload)


def _make_thinking_response(payload: dict) -> str:
    """Simule la reponse d'un modele 'thinking' qui prefixe avec du CoT."""
    import json

    return f"Okay, let me analyze the debate step by step...\n\nFinal answer:\n{json.dumps(payload)}"


# =============================================================================
# Tests is_enabled
# =============================================================================
class TestIsEnabled:
    def test_disabled_by_default(self):
        """Sans ENABLE_REASONING_AUDITOR, doit retourner False."""
        with patch.dict("os.environ", {}, clear=True):
            assert ReasoningAuditor.is_enabled() is False

    def test_enabled_when_env_set_to_one(self):
        with patch.dict("os.environ", {"ENABLE_REASONING_AUDITOR": "1"}):
            assert ReasoningAuditor.is_enabled() is True

    def test_disabled_when_env_set_to_zero(self):
        with patch.dict("os.environ", {"ENABLE_REASONING_AUDITOR": "0"}):
            assert ReasoningAuditor.is_enabled() is False

    def test_disabled_when_env_set_to_other_string(self):
        """Toute valeur != '1' est consideree disabled (strict)."""
        with patch.dict("os.environ", {"ENABLE_REASONING_AUDITOR": "true"}):
            assert ReasoningAuditor.is_enabled() is False


# =============================================================================
# Tests audit() : parsing du JSON
# =============================================================================
class TestAuditParsing:
    """Le ReasoningAuditor doit parser robustement la sortie LLM."""

    def _make_auditor_with_stub(self, raw_response: str) -> ReasoningAuditor:
        """Cree un auditor dont le client LLM retourne une reponse fixe."""
        client = LLMClient(stub_response=raw_response)
        return ReasoningAuditor(client=client)

    def test_clean_audit_no_fallacies(self):
        """Reponse CLEAN avec 0 fallacy : severity=CLEAN, adjustment=0."""
        payload = {
            "severity": "clean",
            "fallacies": [],
            "summary": "Solid reasoning, no fallacies detected.",
            "confidence_adjustment": 0.0,
        }
        auditor = self._make_auditor_with_stub(_make_audit_response(payload))
        audit = auditor.audit("[Tour 1] argument bull [Tour 2] counter-arg [Tour 3] consensus")

        assert audit.severity == AuditSeverity.CLEAN
        assert audit.fallacies == []
        assert audit.confidence_adjustment == 0.0

    def test_severe_audit_with_multiple_fallacies(self):
        """Reponse SEVERE avec 2 fallacies : parsing complet + adjustment <= -0.7."""
        payload = {
            "severity": "severe",
            "fallacies": [
                {
                    "fallacy_type": "confirmation_bias",
                    "quoted_excerpt": "ignores the bear's strong counter-evidence",
                    "explanation": "Bull agent dismissed all bearish data without analysis.",
                    "severity_local": 0.8,
                },
                {
                    "fallacy_type": "ad_hoc_rescue",
                    "quoted_excerpt": "well, in this exceptional case...",
                    "explanation": "Invented a special case to dodge the counter-argument.",
                    "severity_local": 0.7,
                },
            ],
            "summary": "Reasoning structurally flawed by ad hoc rescues.",
            "confidence_adjustment": -0.85,
        }
        auditor = self._make_auditor_with_stub(_make_audit_response(payload))
        audit = auditor.audit("[Tour 1] biased argument [Tour 2] dismissal [Tour 3] ad hoc")

        assert audit.severity == AuditSeverity.SEVERE
        assert len(audit.fallacies) == 2
        assert audit.fallacies[0].fallacy_type == FallacyType.CONFIRMATION_BIAS
        assert audit.fallacies[1].fallacy_type == FallacyType.AD_HOC_RESCUE
        assert audit.confidence_adjustment == -0.85

    def test_thinking_model_prefix_handled(self):
        """Le modele thinking peut prefixer avec du CoT verbeux : on l'ignore."""
        payload = {
            "severity": "minor",
            "fallacies": [],
            "summary": "Mostly clean.",
            "confidence_adjustment": -0.1,
        }
        auditor = self._make_auditor_with_stub(_make_thinking_response(payload))
        scratchpad = (
            "[Tour 1] " + "argument bullish " * 5 + "[Tour 2] " + "counter " * 5 + "[Tour 3] " + "consensus " * 5
        )
        audit = auditor.audit(scratchpad)
        assert audit.severity == AuditSeverity.MINOR
        assert audit.confidence_adjustment == -0.1

    def test_invalid_json_returns_clean(self):
        """Si la reponse n'est pas du JSON valide, retourne CLEAN, pas crash."""
        auditor = self._make_auditor_with_stub("This is not JSON at all, just text.")
        audit = auditor.audit("[Tour 1] something [Tour 2] something [Tour 3] something")
        assert audit.severity == AuditSeverity.CLEAN
        assert audit.fallacies == []

    def test_severity_invalid_falls_back_to_clean(self):
        """Severity en dehors de l'enum -> CLEAN."""
        payload = {"severity": "catastrophic", "fallacies": [], "summary": "", "confidence_adjustment": 0.0}
        auditor = self._make_auditor_with_stub(_make_audit_response(payload))
        audit = auditor.audit(
            "[Tour 1] "
            + "argument bullish " * 5
            + "[Tour 2] "
            + "counter-argument bearish " * 5
            + "[Tour 3] "
            + "consensus reached " * 5
        )
        assert audit.severity == AuditSeverity.CLEAN

    def test_unknown_fallacy_type_filtered(self):
        """Une fallacy hors taxonomie est ignoree silencieusement."""
        payload = {
            "severity": "minor",
            "fallacies": [
                {
                    "fallacy_type": "unknown_fallacy_xyz",
                    "quoted_excerpt": "x",
                    "explanation": "y",
                    "severity_local": 0.5,
                },
                {
                    "fallacy_type": "anchoring",
                    "quoted_excerpt": "fixated on EPS beat",
                    "explanation": "Ignored guidance.",
                    "severity_local": 0.4,
                },
            ],
            "summary": "Minor anchoring.",
            "confidence_adjustment": -0.1,
        }
        auditor = self._make_auditor_with_stub(_make_audit_response(payload))
        audit = auditor.audit(
            "[Tour 1] "
            + "argument bullish " * 5
            + "[Tour 2] "
            + "counter-argument bearish " * 5
            + "[Tour 3] "
            + "consensus reached " * 5
        )
        # Seule la fallacy valide reste
        assert len(audit.fallacies) == 1
        assert audit.fallacies[0].fallacy_type == FallacyType.ANCHORING

    def test_confidence_adjustment_clamped_to_unit_interval(self):
        """Si le LLM rend -2.5 ou +0.5, on clamp dans [-1.0, 0.0]."""
        payload_low = {"severity": "severe", "fallacies": [], "summary": "x", "confidence_adjustment": -2.5}
        auditor = self._make_auditor_with_stub(_make_audit_response(payload_low))
        audit = auditor.audit(
            "[Tour 1] "
            + "argument bullish " * 5
            + "[Tour 2] "
            + "counter-argument bearish " * 5
            + "[Tour 3] "
            + "consensus reached " * 5
        )
        assert audit.confidence_adjustment == -1.0

        payload_high = {"severity": "clean", "fallacies": [], "summary": "x", "confidence_adjustment": 0.7}
        auditor = self._make_auditor_with_stub(_make_audit_response(payload_high))
        audit = auditor.audit(
            "[Tour 1] "
            + "argument bullish " * 5
            + "[Tour 2] "
            + "counter-argument bearish " * 5
            + "[Tour 3] "
            + "consensus reached " * 5
        )
        assert audit.confidence_adjustment == 0.0


# =============================================================================
# Tests robustesse : LLM down, scratchpad invalide
# =============================================================================
class TestRobustness:
    """Le ReasoningAuditor ne doit JAMAIS crasher la pipeline."""

    def test_empty_scratchpad_returns_clean(self):
        client = LLMClient(stub_response="")
        auditor = ReasoningAuditor(client=client)
        audit = auditor.audit("")
        assert audit.severity == AuditSeverity.CLEAN

    def test_too_short_scratchpad_returns_clean(self):
        client = LLMClient(stub_response="")
        auditor = ReasoningAuditor(client=client)
        audit = auditor.audit("hi")  # < 50 chars
        assert audit.severity == AuditSeverity.CLEAN

    def test_llm_failure_returns_clean(self):
        """Si TOUS les providers tombent, retourne CLEAN sans lever."""
        client = LLMClient(api_keys={"groq": "gsk_x"}, max_retries=1, backoff_base_sec=0.0)
        auditor = ReasoningAuditor(client=client)

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("network down")

            audit = auditor.audit("[Tour 1] valid scratchpad [Tour 2] x [Tour 3] consensus reached")
            assert audit.severity == AuditSeverity.CLEAN
            assert audit.fallacies == []


# =============================================================================
# Tests serialization
# =============================================================================
class TestSerialization:
    """to_dict() doit produire un dict JSON-serializable."""

    def test_to_dict_serializes_enums_as_strings(self):
        import json

        from src.agents.agent_reasoning_auditor import FallacyDetection

        client = LLMClient(stub_response="")
        auditor = ReasoningAuditor(client=client)
        audit = ReasoningAudit(
            severity=AuditSeverity.MODERATE,
            fallacies=[
                FallacyDetection(
                    fallacy_type=FallacyType.ANCHORING,
                    quoted_excerpt="EPS beat",
                    explanation="ignored guidance",
                    severity_local=0.6,
                )
            ],
            summary="moderate anchoring",
            confidence_adjustment=-0.4,
        )
        d = auditor.to_dict(audit)
        # Serialisable en JSON sans erreur
        json_str = json.dumps(d)
        assert "moderate" in json_str
        assert "anchoring" in json_str

    def test_to_dict_round_trip_via_json(self):
        """Dump JSON puis load doit redonner les memes champs cles."""
        import json

        from src.agents.agent_reasoning_auditor import FallacyDetection

        client = LLMClient(stub_response="")
        auditor = ReasoningAuditor(client=client)
        audit = ReasoningAudit(
            severity=AuditSeverity.MINOR,
            fallacies=[
                FallacyDetection(
                    fallacy_type=FallacyType.CHERRY_PICKING,
                    quoted_excerpt="picks only positive evidence",
                    explanation="biased selection",
                    severity_local=0.3,
                )
            ],
            summary="minor cherry-picking",
            confidence_adjustment=-0.15,
        )
        d = auditor.to_dict(audit)
        round_trip = json.loads(json.dumps(d))
        assert round_trip["severity"] == "minor"
        assert round_trip["fallacies"][0]["fallacy_type"] == "cherry_picking"


# =============================================================================
# Tests d'integration : ReasoningAuditor + best_model_for_task
# =============================================================================
class TestModelRouting:
    """L'auditor doit utiliser par defaut le modele du registre BEST_MODELS_BY_TASK."""

    def test_default_model_from_registry(self):
        """Sans override, l'auditor utilise BEST_MODELS_BY_TASK['reasoning_audit']."""
        from src.utils.llm_client import best_model_for_task

        client = LLMClient(stub_response="")
        auditor = ReasoningAuditor(client=client)
        expected_provider, expected_model = best_model_for_task("reasoning_audit")
        assert auditor._provider == expected_provider
        assert auditor._model == expected_model

    def test_explicit_override_takes_precedence(self):
        """provider/model explicites dans le constructeur prennent le pas."""
        client = LLMClient(stub_response="")
        auditor = ReasoningAuditor(
            client=client,
            provider="nvidia_nim",
            model="meta/llama-3.1-405b-instruct",
        )
        assert auditor._provider == "nvidia_nim"
        assert auditor._model == "meta/llama-3.1-405b-instruct"
