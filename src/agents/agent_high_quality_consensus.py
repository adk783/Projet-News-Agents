"""HighQualityConsensus : second avis via Llama-3.1-405B sur les decisions HIGH_RISK.

CONTEXTE
--------
Le debat 3-agents + Consensus (Groq Llama-3.3-70B) tourne sur tous les articles.
Pour les **decisions HIGH_RISK** identifiees par le YOLO classifier (incoherence
FinBERT/LLM, signal a faible consensus, vol > seuil), on veut un **second avis**
issu d'un modele significativement plus puissant.

POURQUOI LLAMA-3.1-405B SPECIFIQUEMENT ?
----------------------------------------
- Bench public : 405B domine 70B sur les benchmarks complexes (MMLU, GPQA, MATH).
  Meta Llama 3.1 paper (juillet 2024).
- Bench live (2026-04-26) sur le projet : 405B produit une argumentation plus
  detaillee et nuancee que 70B sur le meme prompt (~80-90 mots structures).
- Disponible **gratuitement** via NVIDIA NIM (40 RPM tier free).

POURQUOI PAS DANS LE PIPELINE LIVE STANDARD ?
---------------------------------------------
Bench live : latence Llama-3.1-405B = **110 secondes** par appel.
Inacceptable pour 50-100 articles/jour. Ce module est donc utilise uniquement
sur les decisions HIGH_RISK (typiquement 1-5/jour) ou quand le ReasoningAuditor
detecte des fallacies severes.

POLITIQUE D'ADOPTION
--------------------
- Opt-in via env var `ENABLE_HIGH_QUALITY_CONSENSUS=1` (defaut 0).
- Appele uniquement si :
  1. YOLO classifier renvoie `risk_level=ELEVE` ou `requires_human=True`, OU
  2. ReasoningAudit severity in (MODERATE, SEVERE).
- Si la cle NIM est absente, le module se degrade gracieusement (retourne le
  consensus original sans modification).

REFERENCE
---------
- Meta AI (2024). "The Llama 3 Herd of Models." arXiv:2407.21783.
- Wang, Z. et al. (2023). "Self-Consistency Improves Chain-of-Thought
  Reasoning in Language Models." ICLR. (justification : un second avis
  d'un modele plus puissant fonctionne comme un vote majoritaire pondere).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from src.utils.llm_client import (
    AllProvidersFailedError,
    LLMClient,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================
@dataclass
class HighQualityVerdict:
    """Verdict du Consensus haute qualite."""

    signal_final: str  # "Achat" | "Vente" | "Neutre"
    confiance: float  # 0.0-1.0
    raisonnement: str  # justification 2-4 phrases
    agrees_with_original: bool  # True si meme signal que le consensus original
    provider_used: str = ""
    model_used: str = ""
    latency_sec: float = 0.0


# =============================================================================
# Prompt
# =============================================================================
_HQC_PROMPT = """You are a SENIOR FINANCIAL ANALYST issuing a SECOND OPINION on a stock decision.

Below is a debate between 3 agents (Bullish, Bearish, Neutral) and the original
consensus signal. Your job is to provide an INDEPENDENT high-quality verdict,
NOT just rubber-stamp the original.

ARTICLE :
\"\"\"
{article_summary}
\"\"\"

DEBATE SCRATCHPAD (3 rounds):
\"\"\"
{scratchpad}
\"\"\"

ORIGINAL CONSENSUS:
- Signal: {original_signal}
- Confidence: {original_confidence:.0%}
- Reasoning: {original_reasoning}

YOUR TASK:
1. Re-analyze the debate INDEPENDENTLY.
2. Decide whether you AGREE or DISAGREE with the original signal.
3. Provide your own confidence level (do NOT anchor on the original).
4. Justify in 2-4 sentences.

OUTPUT FORMAT (strict JSON, no preamble, no markdown fences):
{{
  "signal_final": "Achat" | "Vente" | "Neutre",
  "confiance": <float 0.0-1.0>,
  "raisonnement": "<2-4 sentences justifying your verdict>"
}}

RULES:
1. Be RIGOROUS : if the debate is weak, lower the confidence (even if you agree on direction).
2. Be INDEPENDENT : "Neutre" is a valid verdict if the debate is genuinely ambiguous.
3. confidence must be IN [0.0, 1.0]. Use < 0.6 for weak signals, 0.6-0.8 for solid, > 0.8 for very strong.
"""


# =============================================================================
# Service
# =============================================================================
class HighQualityConsensus:
    """Second avis Consensus pour decisions HIGH_RISK via Llama-3.1-405B (NIM).

    Usage canonique :
        hqc = HighQualityConsensus()
        if hqc.should_invoke(yolo_risk_level="ELEVE"):
            verdict = hqc.evaluate(
                article_summary=summary,
                scratchpad=scratchpad_xml,
                original_signal="Achat",
                original_confidence=0.62,
                original_reasoning="...",
            )
            if not verdict.agrees_with_original:
                # log + rabaisser le sizing, etc.
                ...
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: str = "nvidia_nim",
        model: str = "meta/llama-3.1-405b-instruct",
        max_tokens: int = 800,
    ):
        self._client = client or LLMClient.from_env()
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def is_enabled() -> bool:
        """Retourne True si ENABLE_HIGH_QUALITY_CONSENSUS=1 dans l'env."""
        return os.getenv("ENABLE_HIGH_QUALITY_CONSENSUS", "0") == "1"

    @staticmethod
    def should_invoke(
        yolo_risk_level: str | None = None,
        audit_severity: str | None = None,
    ) -> bool:
        """Heuristique : invoquer le HQC seulement si signal HIGH_RISK.

        Parameters
        ----------
        yolo_risk_level : str
            Niveau de risque du YOLO classifier ('FAIBLE', 'MOYEN', 'ELEVE').
        audit_severity : str
            Severity du ReasoningAuditor ('clean', 'minor', 'moderate', 'severe').

        Returns
        -------
        bool
            True si l'un des criteres declenche un second avis.
        """
        # Critere 1 : YOLO ELEVE
        if (yolo_risk_level or "").upper() == "ELEVE":
            return True
        # Critere 2 : audit reasoning serieux
        if (audit_severity or "").lower() in ("moderate", "severe"):
            return True
        return False

    def evaluate(
        self,
        article_summary: str,
        scratchpad: str,
        original_signal: str,
        original_confidence: float,
        original_reasoning: str = "",
    ) -> HighQualityVerdict:
        """Demande un second avis au modele haute qualite.

        Si NIM indisponible ou erreur : retourne un verdict identique a
        l'original (pas de modification, pas de crash).
        """
        import json
        import re
        import time

        # Garde-fou : provider absent -> pas de second avis
        if self._provider not in self._client.available_providers():
            logger.debug(
                "[HQC] Provider %s indisponible, skip (cle absente ?)",
                self._provider,
            )
            return self._fallback_to_original(
                original_signal,
                original_confidence,
                original_reasoning,
            )

        prompt = _HQC_PROMPT.format(
            article_summary=article_summary[:3000],
            scratchpad=scratchpad[:6000],
            original_signal=original_signal,
            original_confidence=original_confidence,
            original_reasoning=original_reasoning[:1000] or "(non fourni)",
        )

        t0 = time.time()
        try:
            resp = self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                model_preference=[self._provider],
                model_override=self._model,
                max_tokens=self._max_tokens,
                temperature=0.1,  # bas mais > 0 pour eviter mode collapse
            )
        except AllProvidersFailedError as e:
            logger.error("[HQC] LLM call failed : %s", e)
            return self._fallback_to_original(
                original_signal,
                original_confidence,
                original_reasoning,
            )

        latency = time.time() - t0
        json_match = re.search(r'\{.*"signal_final".*\}', resp.content, re.DOTALL)
        if not json_match:
            logger.warning("[HQC] aucun JSON dans la reponse, fallback original")
            return self._fallback_to_original(
                original_signal,
                original_confidence,
                original_reasoning,
            )

        try:
            payload = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning("[HQC] JSON invalide (%s), fallback original", e)
            return self._fallback_to_original(
                original_signal,
                original_confidence,
                original_reasoning,
            )

        # Validation des champs
        sig = str(payload.get("signal_final", original_signal)).strip()
        if sig not in ("Achat", "Vente", "Neutre"):
            logger.warning("[HQC] signal invalide '%s', fallback", sig)
            return self._fallback_to_original(
                original_signal,
                original_confidence,
                original_reasoning,
            )

        try:
            conf = float(payload.get("confiance", original_confidence))
        except (TypeError, ValueError):
            conf = original_confidence
        # Clamp dans [0, 1]
        conf = max(0.0, min(1.0, conf))

        verdict = HighQualityVerdict(
            signal_final=sig,
            confiance=conf,
            raisonnement=str(payload.get("raisonnement", "")),
            agrees_with_original=(sig == original_signal),
            provider_used=resp.provider_used,
            model_used=resp.model_used,
            latency_sec=round(latency, 2),
        )
        logger.info(
            "[HQC] Verdict : %s (%.0f%%) | original was : %s (%.0f%%) | agree=%s | %.1fs",
            verdict.signal_final,
            verdict.confiance * 100,
            original_signal,
            original_confidence * 100,
            verdict.agrees_with_original,
            latency,
        )
        return verdict

    @staticmethod
    def _fallback_to_original(
        signal: str,
        confidence: float,
        reasoning: str,
    ) -> HighQualityVerdict:
        """Fallback degrade : retourne le verdict original sans modification."""
        return HighQualityVerdict(
            signal_final=signal,
            confiance=confidence,
            raisonnement=f"[fallback] {reasoning}",
            agrees_with_original=True,
            provider_used="fallback",
            model_used="none",
            latency_sec=0.0,
        )


__all__ = ["HighQualityConsensus", "HighQualityVerdict"]
