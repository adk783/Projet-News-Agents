"""Reasoning Auditor : audit post-debat de la chaine logique multi-agent.

CONTEXTE
--------
Le debat 3-agents (Bull/Bear/Neutre) produit un scratchpad XML avec les
arguments de chaque tour. Aujourd'hui, l'audit existant (`agent_critic`,
`agent_verifier`) regarde **arguments individuels** : claim X est-il
supporte par evidence Y ?

Ce module ajoute une couche distincte : **audit de la STRUCTURE LOGIQUE
GLOBALE** du debat. Il detecte les fallacies argumentatives (non sequitur,
ad hoc, ancrage, confirmation bias) que l'analyse argument-par-argument
ne capture pas.

POURQUOI UN MODELE "THINKING" SPECIFIQUE
----------------------------------------
La chain-of-thought explicite (Wei et al. 2022) est un avantage net pour
l'audit logique : on **veut voir** le raisonnement, pas seulement la
conclusion. Les modeles "thinking" (Qwen3-thinking, DeepSeek-R1, magistral)
sont entraines en RLHF specifique pour produire un CoT structure.

C'est different de l'agent_critic actuel (qui peut tourner sur n'importe
quel LLM Groq/Mistral/Cerebras) — ici on cherche **explicitement**
l'analyse step-by-step.

REFERENCE
---------
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in
  Large Language Models." NeurIPS.
- Du, Y. et al. (2023). "Improving Factuality and Reasoning in Language
  Models through Multiagent Debate." NeurIPS.
- DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in
  LLMs via Reinforcement Learning."

POLITIQUE D'ADOPTION
--------------------
- **Opt-in** via env var `ENABLE_REASONING_AUDITOR=1`. Pas active par defaut
  (1 appel LLM supplementaire = +0.5-3s de latence + cout).
- Recommande pour le mode **audit hebdomadaire** (`scripts/audit_hebdomadaire.py`)
  ou pour les decisions HIGH_RISK (YOLO classifier `requires_human=True`).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from src.utils.llm_client import (
    AllProvidersFailedError,
    LLMClient,
    best_model_for_task,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================
class FallacyType(str, Enum):
    """Taxonomie des fallacies argumentatives detectables."""

    NON_SEQUITUR = "non_sequitur"  # conclusion ne suit pas des premisses
    AD_HOC_RESCUE = "ad_hoc_rescue"  # ajustement opportuniste pour sauver une these
    ANCHORING = "anchoring"  # focalisation sur 1 chiffre/quote initial
    CONFIRMATION_BIAS = "confirmation_bias"  # ignore les contre-evidences
    BASE_RATE_NEGLECT = "base_rate_neglect"  # ignore les frequences statistiques
    CHERRY_PICKING = "cherry_picking"  # selection biaisee d'evidences
    HASTY_GENERALIZATION = "hasty_generalization"  # echantillon trop petit
    FALSE_DICHOTOMY = "false_dichotomy"  # presente faux 2 options exclusives
    APPEAL_TO_AUTHORITY = "appeal_to_authority"  # "expert dit X donc X" sans evidence


class AuditSeverity(str, Enum):
    """Severite globale de l'audit."""

    CLEAN = "clean"  # 0 fallacy detectee, raisonnement solide
    MINOR = "minor"  # 1-2 fallacies de faible impact
    MODERATE = "moderate"  # 3+ fallacies ou 1 majeure
    SEVERE = "severe"  # raisonnement structurellement defaillant


@dataclass
class FallacyDetection:
    """Detection d'une fallacy specifique."""

    fallacy_type: FallacyType
    quoted_excerpt: str  # citation textuelle du scratchpad
    explanation: str  # pourquoi c'est une fallacy (1-2 phrases)
    severity_local: float  # 0.0-1.0 : impact de cette fallacy isolement


@dataclass
class ReasoningAudit:
    """Resultat complet d'un audit."""

    severity: AuditSeverity
    fallacies: list[FallacyDetection] = field(default_factory=list)
    summary: str = ""  # 1-3 phrases de synthese
    confidence_adjustment: float = 0.0  # multiplicateur a appliquer a la confiance debat (-1.0 a 0.0)
    raw_audit_text: str = ""  # CoT complet du modele auditeur (pour debug)
    provider_used: str = ""
    model_used: str = ""


# =============================================================================
# Prompt
# =============================================================================
_AUDIT_PROMPT = """You are a Reasoning Auditor for a financial multi-agent debate.

The agents (Bullish, Bearish, Neutral) just debated about a stock. Below is the
COMPLETE debate scratchpad (3 rounds). Your job is to detect ARGUMENTATIVE
FALLACIES in their reasoning structure, NOT to take a position on the stock.

TAXONOMY OF FALLACIES TO DETECT:
- non_sequitur          : conclusion does not follow from premises
- ad_hoc_rescue         : agent invents an explanation to save its thesis after counter-evidence
- anchoring             : argument fixates on one initial number/quote and ignores adjustments
- confirmation_bias     : agent ignores or dismisses contradicting evidence
- base_rate_neglect     : ignores statistical baselines (e.g., "78% of M&A fail" ignored)
- cherry_picking        : selectively quotes only supportive evidence
- hasty_generalization  : conclusion drawn from too small a sample
- false_dichotomy       : presents only 2 options when more exist
- appeal_to_authority   : "expert X says Y, therefore Y" without showing evidence

DEBATE SCRATCHPAD:
\"\"\"
{scratchpad}
\"\"\"

OUTPUT FORMAT (strict JSON, no preamble, no markdown):
{{
  "severity": "clean" | "minor" | "moderate" | "severe",
  "fallacies": [
    {{
      "fallacy_type": "<one of the taxonomy>",
      "quoted_excerpt": "<verbatim quote from scratchpad, max 200 chars>",
      "explanation": "<1-2 sentences why this is a fallacy>",
      "severity_local": 0.0 to 1.0
    }}
  ],
  "summary": "<1-3 sentences synthesis of the audit>",
  "confidence_adjustment": <float in [-1.0, 0.0]; 0.0 = no adjustment, -0.5 = halve final confidence, -1.0 = signal unusable>
}}

RULES:
1. Be RIGOROUS but FAIR : a strong argument with one weak link is "minor", not "severe".
2. Quote VERBATIM from the scratchpad (no paraphrase).
3. If the debate is CLEAN, return severity="clean" and fallacies=[].
4. confidence_adjustment must be in [-1.0, 0.0]. severe -> -0.7 to -1.0, moderate -> -0.3 to -0.6, minor -> -0.1 to -0.2, clean -> 0.0.
5. CRITICAL: You MAY think out loud first, but your FINAL output MUST contain a complete JSON object matching the format above. Conclude your response with the JSON, even if your reasoning is long. Do not omit the JSON.
"""


# =============================================================================
# API publique
# =============================================================================
class ReasoningAuditor:
    """Audite un scratchpad de debat multi-agent via un modele 'thinking'.

    Usage canonique :
        auditor = ReasoningAuditor()  # client + modele auto-selectionnes
        if auditor.is_enabled():
            audit = auditor.audit(scratchpad_xml)
            if audit.severity in (AuditSeverity.MODERATE, AuditSeverity.SEVERE):
                # Ajuster la confiance, requerir une revue humaine, etc.
                final_confidence *= (1.0 + audit.confidence_adjustment)
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ):
        """Init du Reasoning Auditor.

        Parameters
        ----------
        client
            LLMClient injectable. Defaut : LLMClient.from_env().
        provider, model
            Override pour utiliser un modele specifique. Defaut : selection
            via `best_model_for_task("reasoning_audit")` (Qwen3-thinking).
        max_tokens
            Max output tokens. Les modeles thinking (Qwen3-thinking,
            magistral, nemotron-ultra) consomment beaucoup de tokens en
            chain-of-thought AVANT d'ecrire le JSON final. 4096 garantit
            ~3000 tokens de CoT + ~1000 tokens pour le JSON. Tester live
            avec moins risque de tronquer le JSON et faire echouer le parsing.
        """
        self._client = client or LLMClient.from_env()
        if provider is None or model is None:
            default_provider, default_model = best_model_for_task("reasoning_audit")
            provider = provider or default_provider
            model = model or default_model
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def is_enabled() -> bool:
        """Retourne True si ENABLE_REASONING_AUDITOR=1 dans l'env."""
        return os.getenv("ENABLE_REASONING_AUDITOR", "0") == "1"

    def audit(self, scratchpad_xml: str) -> ReasoningAudit:
        """Audite le scratchpad et retourne un ReasoningAudit structure.

        En cas d'erreur (LLM down, parsing JSON failed), retourne un audit
        vide avec severity=CLEAN et adjustment=0.0 (degrade gracieusement,
        n'impacte pas la decision de trading).
        """
        if not scratchpad_xml or len(scratchpad_xml.strip()) < 50:
            logger.warning("[ReasoningAuditor] scratchpad vide ou trop court, skip")
            return ReasoningAudit(severity=AuditSeverity.CLEAN)

        prompt = _AUDIT_PROMPT.format(scratchpad=scratchpad_xml[:8000])

        try:
            resp = self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                model_preference=[self._provider],
                model_override=self._model,
                max_tokens=self._max_tokens,
                temperature=0.0,  # deterministique pour audit reproductible
            )
        except AllProvidersFailedError as e:
            logger.error("[ReasoningAuditor] LLM call failed : %s", e)
            return ReasoningAudit(severity=AuditSeverity.CLEAN)

        return self._parse_response(resp.content, resp.provider_used, resp.model_used)

    def _parse_response(self, raw: str, provider: str, model: str) -> ReasoningAudit:
        """Extrait le JSON de la reponse (tolerant : extrait le 1er bloc JSON)."""
        # Modeles thinking peuvent prefixer la sortie de CoT verbeux.
        # On extrait le 1er bloc JSON au format {"severity": ...}.
        json_match = re.search(r'\{.*"severity".*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning(
                "[ReasoningAuditor] aucun JSON dans la reponse (head=%s)",
                raw[:200].replace("\n", " "),
            )
            return ReasoningAudit(
                severity=AuditSeverity.CLEAN,
                raw_audit_text=raw,
                provider_used=provider,
                model_used=model,
            )

        try:
            payload = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning("[ReasoningAuditor] JSON invalide : %s", e)
            return ReasoningAudit(
                severity=AuditSeverity.CLEAN,
                raw_audit_text=raw,
                provider_used=provider,
                model_used=model,
            )

        # Severity : valider l'enum
        sev_str = (payload.get("severity") or "clean").lower()
        try:
            severity = AuditSeverity(sev_str)
        except ValueError:
            logger.warning("[ReasoningAuditor] severity invalide '%s', fallback CLEAN", sev_str)
            severity = AuditSeverity.CLEAN

        # Fallacies : valider chaque entree
        fallacies = []
        for item in payload.get("fallacies", []):
            try:
                ftype = FallacyType(item.get("fallacy_type", "").lower())
            except ValueError:
                logger.debug("Fallacy type inconnu : %s", item.get("fallacy_type"))
                continue
            fallacies.append(
                FallacyDetection(
                    fallacy_type=ftype,
                    quoted_excerpt=str(item.get("quoted_excerpt", ""))[:300],
                    explanation=str(item.get("explanation", "")),
                    severity_local=float(item.get("severity_local", 0.5)),
                )
            )

        # Confidence adjustment : clamp dans [-1.0, 0.0]
        try:
            adj = float(payload.get("confidence_adjustment", 0.0))
        except (TypeError, ValueError):
            adj = 0.0
        adj = max(-1.0, min(0.0, adj))

        return ReasoningAudit(
            severity=severity,
            fallacies=fallacies,
            summary=str(payload.get("summary", "")),
            confidence_adjustment=adj,
            raw_audit_text=raw,
            provider_used=provider,
            model_used=model,
        )

    def to_dict(self, audit: ReasoningAudit) -> dict:
        """Serialise un audit en dict (pour persistence JSONL/SQLite)."""
        d = asdict(audit)
        # Enums -> str pour JSON
        d["severity"] = audit.severity.value
        d["fallacies"] = [{**asdict(f), "fallacy_type": f.fallacy_type.value} for f in audit.fallacies]
        return d


__all__ = [
    "ReasoningAuditor",
    "ReasoningAudit",
    "FallacyDetection",
    "FallacyType",
    "AuditSeverity",
]
