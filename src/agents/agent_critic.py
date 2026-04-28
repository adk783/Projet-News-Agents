"""
agent_critic.py — Actor-Critic pattern pour audit des arguments de debat.

IDEE
----
Dans une architecture multi-agent classique (Haussier vs Baissier), chaque
agent produit des arguments qui sont agreges par un Consensus. Rien ne
penalise les arguments biaises, non supportes, ou logiquement bancals.

Le pattern Actor-Critic (Zhao et al. 2024, "Critic-RLR") introduit un agent
dedie a l'audit :
  - Il ne produit pas d'arguments directionnels
  - Il lit la sortie des debatteurs et identifie :
      * Biais de raisonnement (ancrage, confirmation, selection)
      * Trous logiques (premisses manquantes, non-sequitur)
      * Claims non-supportes par l'evidence fournie
  - Il emet un feedback structure (CriticFeedback Pydantic)
  - Le Consensus peut ponderer les arguments par la severite du feedback

UTILISATION
-----------
    critic = CriticAgent(call_llm=my_llm_callable)
    fb = critic.review(argument=debate_arg, context={"absa": ..., "article": ...})
    if fb.severity == Severity.HIGH:
        # desavouer l'argument ou demander une revision
        ...

DEPENDANCES
-----------
- structured_output.CriticFeedback (schema)
- structured_output.structured_call (parsing + retry)

REFERENCES
----------
- Zhao, W. X. et al. (2024). "Critic-RLR: Enhancing LLM Reasoning with
  Critic-in-the-Loop Refinement."
- Madaan, A. et al. (2023). "Self-Refine: Iterative Refinement with
  Self-Feedback." NeurIPS.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Optional

# Imports absolus (cf. ADR-008, mode editable `pip install -e .`).
from src.utils.structured_output import (
    CriticFeedback,
    DebateArgument,
    Severity,
    structured_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """Tu es un Critique expert en analyse financiere.
Ton role est UNIQUEMENT d'auditer un argument produit par un debatteur.
Tu ne prends PAS de position directionnelle (ni Achat ni Vente).

Tu dois identifier, de maniere rigoureuse et sans complaisance :
  1. BIAIS de raisonnement : ancrage, confirmation, selection,
     disponibilite, survival bias, hindsight bias, cherry-picking.
  2. TROUS LOGIQUES : premisse implicite non etablie, non-sequitur,
     post hoc ergo propter hoc, faux dilemme, generalisation hative.
  3. CLAIMS NON-SUPPORTES : affirmations factuelles sans evidence
     (chiffre non source, causalite affirmee sans donnee, analogie faible).

Tu dois aussi proposer des revisions concretes pour corriger l'argument.

FORMAT DE SORTIE — EXCLUSIVEMENT JSON, pas de markdown :
{{
  "target_debater_id": "<id du debatteur analyse>",
  "target_round": <int>,
  "biases_detected": ["biais1", "biais2"],
  "logical_gaps": ["gap1"],
  "unsupported_claims": ["claim1", "claim2"],
  "severity": "low" | "medium" | "high",
  "suggested_revisions": ["revision1", "revision2"]
}}

Regles :
- Tes listes sont vides si tu ne trouves rien (c'est l'honnetete qui compte).
- severity = "high" uniquement si l'argument s'effondre sans les claims non-supportes.
- severity = "medium" si l'argument tient mais est fragile sur 1-2 points.
- severity = "low" si l'argument est propre.
- Chaque item doit etre concret et citable (eviter "l'argument est faible").
"""


CRITIC_USER_TEMPLATE = """## Argument a auditer

Debatteur : {debater_id}
Round : {round}
Position : {position}
Confiance : {confidence}

Thesis :
{thesis}

Evidence citee :
{evidence}

Contre-argument de l'opposant (si fourni) :
{counters_opponent}

## Contexte factuel disponible

{context_blob}

## Ta tache

Audite cet argument selon tes criteres. Renvoie UN SEUL objet JSON conforme au schema."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CriticAgent:
    """Actor-Critic : audite un DebateArgument, retourne un CriticFeedback."""

    def __init__(
        self,
        call_llm: Callable[[str], str],
        system_prompt: str = CRITIC_SYSTEM_PROMPT,
        user_template: str = CRITIC_USER_TEMPLATE,
        max_retries: int = 2,
    ):
        self.call_llm = call_llm
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_retries = max_retries

    def _format_context(self, context: Optional[dict]) -> str:
        if not context:
            return "(aucun contexte supplementaire fourni)"
        try:
            return json.dumps(context, ensure_ascii=False, indent=2)[:2000]
        except Exception:
            return str(context)[:2000]

    def _build_prompt(self, argument: DebateArgument, context: Optional[dict]) -> str:
        evidence_txt = "\n".join(f"  - {e}" for e in (argument.evidence or [])) or "(aucune)"
        user = self.user_template.format(
            debater_id=argument.debater_id,
            round=argument.round,
            position=getattr(argument.position, "value", argument.position),
            confidence=argument.confidence,
            thesis=argument.thesis,
            evidence=evidence_txt,
            counters_opponent=argument.counters_opponent or "(aucun)",
            context_blob=self._format_context(context),
        )
        return f"{self.system_prompt}\n\n{user}"

    def review(
        self,
        argument: DebateArgument,
        context: Optional[dict] = None,
    ) -> CriticFeedback:
        """Audite un argument et renvoie un feedback valide."""
        prompt = self._build_prompt(argument, context)
        try:
            return structured_call(
                self.call_llm,
                prompt,
                CriticFeedback,
                max_retries=self.max_retries,
            )
        except ValueError as e:
            logger.warning("CriticAgent validation echouee (%s) — fallback vide", e)
            return CriticFeedback(
                target_debater_id=argument.debater_id,
                target_round=argument.round,
                biases_detected=[],
                logical_gaps=[],
                unsupported_claims=[],
                severity=Severity.LOW,
                suggested_revisions=[],
            )


# ---------------------------------------------------------------------------
# Helper : agregation du feedback d'une ronde pour Consensus
# ---------------------------------------------------------------------------


def aggregate_critic_feedback(feedbacks: list[CriticFeedback]) -> dict:
    """
    Consolide une liste de feedbacks pour transmettre au Consensus.
    Renvoie un dict { debater_id : { severity_avg, issues_count, revisions } }.
    """
    sev_map = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
    by_deb: dict[str, list[CriticFeedback]] = {}
    for f in feedbacks:
        by_deb.setdefault(f.target_debater_id, []).append(f)

    out: dict[str, dict] = {}
    for deb, fs in by_deb.items():
        sev_vals = [sev_map[f.severity] for f in fs]
        issues = sum(len(f.biases_detected) + len(f.logical_gaps) + len(f.unsupported_claims) for f in fs)
        revisions: list[str] = []
        for f in fs:
            revisions.extend(f.suggested_revisions)
        out[deb] = {
            "avg_severity": sum(sev_vals) / max(1, len(sev_vals)),
            "max_severity": max(sev_vals) if sev_vals else 0,
            "issues_count": issues,
            "revisions": revisions[:10],
        }
    return out


def severity_to_weight_penalty(avg_severity: float) -> float:
    """
    Convertit severity moyen [0,2] en coefficient multiplicatif [0.4, 1.0].
    Un debatteur avec severity=2 (toujours HIGH) voit son poids tomber a 0.4.
    """
    sev = max(0.0, min(2.0, avg_severity))
    return 1.0 - 0.3 * sev  # 0->1.0, 1->0.7, 2->0.4


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mock LLM qui renvoie un feedback valide
    def fake_llm(prompt: str) -> str:
        return json.dumps(
            {
                "target_debater_id": "haussier",
                "target_round": 1,
                "biases_detected": ["biais de confirmation : focus sur metriques positives uniquement"],
                "logical_gaps": ["premise implicite : hausse revenue => hausse marge"],
                "unsupported_claims": ["affirmation de PE < 15 sans source"],
                "severity": "medium",
                "suggested_revisions": [
                    "Citer la source du PE",
                    "Evaluer la marge brute, pas seulement le chiffre d'affaires",
                ],
            }
        )

    arg = DebateArgument(
        debater_id="haussier",
        round=1,
        position="bull",
        thesis="Apple doit etre achete : revenue +8%, guidance raised, PE < 15.",
        evidence=["Q3 revenue 89.5B (+8%)", "Guidance raised 2%"],
        confidence=0.8,
    )

    critic = CriticAgent(call_llm=fake_llm)
    fb = critic.review(argument=arg, context={"absa_summary": "Sentiment global: bullish"})
    logger.info("Feedback :")
    logger.info(f"  severity : {fb.severity}")
    logger.info(f"  biases   : {fb.biases_detected}")
    logger.info(f"  gaps     : {fb.logical_gaps}")
    logger.info(f"  unsupp.  : {fb.unsupported_claims}")
    logger.info(f"  revisions: {fb.suggested_revisions}")

    agg = aggregate_critic_feedback([fb])
    logger.info(f"\nAgregation : {agg}")
    w = severity_to_weight_penalty(agg["haussier"]["avg_severity"])
    logger.info(f"Poids resultant : {w:.2f}")

    # Cas : LLM produit du JSON invalide mais validable au 2eme essai
    class Counter:
        def __init__(self):
            self.n = 0

    c = Counter()

    def bad_then_good(p: str) -> str:
        c.n += 1
        if c.n == 1:
            return "pas du tout du json"
        return json.dumps(
            {
                "target_debater_id": "baissier",
                "target_round": 2,
                "biases_detected": [],
                "logical_gaps": [],
                "unsupported_claims": [],
                "severity": "low",
                "suggested_revisions": [],
            }
        )

    critic2 = CriticAgent(call_llm=bad_then_good)
    arg2 = DebateArgument(debater_id="baissier", round=2, position="bear", thesis="vente", evidence=[], confidence=0.5)
    fb2 = critic2.review(argument=arg2)
    logger.info(f"\nRetry OK apres {c.n} tentatives : severity={fb2.severity}")
