"""
agent_verifier.py — Chain-of-Verification (CoVe, Dhuliawala et al. 2023).

PROBLEME
--------
Les LLMs hallucinent des faits : chiffres inventes, citations fabriquees,
causalites implicites qui ne tiennent pas. Dans un pipeline trading, meme
une hallucination rare mais confiante peut declencher un trade dommageable.

IDEE (CoVe)
-----------
Pattern en 4 etapes :
  1. Generate : produire une reponse initiale (e.g. la thesis d'un debatteur).
  2. Plan verification : extraire des claims factuels atomiques.
  3. Verify independently : poser une question par claim, SANS contexte,
     pour eviter la reproduction des hallucinations.
  4. Final : reconstruire une reponse en ne gardant que les claims verifies.

L'etape 3 est cruciale : les questions sont posees a un second LLM
SANS lui donner la thesis d'origine, pour casser le biais de reproduction.

IMPLEMENTATION
--------------
VerifierAgent.verify(text, context_evidence) -> list[VerificationResult]

- `text` : la reponse a verifier (ou l'ensemble des claims)
- `context_evidence` : texte des articles sources disponibles
- Retour : liste de VerificationResult (verdict in {supported, contradicted, unverifiable})

Un score agrege est produit : verification_ratio = supported / total.

REFERENCES
----------
- Dhuliawala, S. et al. (2023). "Chain-of-Verification Reduces Hallucination
  in Large Language Models." arXiv:2309.11495.
- Gao, L. et al. (2023). "RARR: Researching and Revising What Language
  Models Say, Using Language Models."
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# Imports absolus (cf. ADR-008, mode editable `pip install -e .`).
from src.utils.structured_output import (
    Claim,
    VerificationResult,
    parse_llm_json,
    structured_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_CLAIMS_PROMPT = """Tu es un auditeur factuel. Tu recois un texte
d'analyse financiere et tu dois extraire les AFFIRMATIONS FACTUELLES
verifiables, une par ligne.

Une affirmation factuelle est une assertion dont la verite peut etre
tranchee par consultation d'un document source :
  - Chiffres precis (revenu, PE, croissance)
  - Entites nommees (CEO, produit, date d'evenement)
  - Causalites explicites (X a cause Y)
  - Relations temporelles (avant/apres/durant)

EXCLUE :
  - Opinions subjectives ("c'est une bonne affaire")
  - Predictions futures ("va monter")
  - Modalites ("probablement", "semble")
  - Exhortations ("il faut acheter")

FORMAT — JSON UNIQUEMENT :
{{
  "claims": [
    {{"text": "Apple Q3 revenue = $89.5B", "category": "numeric", "verifiable": true}},
    {{"text": "Tim Cook is CEO of Apple", "category": "entity", "verifiable": true}},
    {{"text": "The earnings caused the stock to rise 2%", "category": "causal", "verifiable": true}}
  ]
}}

Categories valides : numeric | entity | causal | temporal | other

Texte a analyser :
---
{text}
---
"""


VERIFY_CLAIM_PROMPT = """Tu es un verificateur factuel INDEPENDANT. Tu recois
UNE affirmation et un corpus de sources. Tu dois rendre un verdict :
  - "supported"     : l'affirmation est clairement soutenue par les sources
  - "contradicted"  : les sources contredisent explicitement l'affirmation
  - "unverifiable"  : les sources ne permettent ni de confirmer ni d'infirmer

IMPORTANT :
  - Si le chiffre exact est dans les sources : supported. Si approche (±5%) : supported.
  - Si un chiffre DIFFERENT figure : contradicted.
  - Si non mentionne : unverifiable.
  - Ne pas halluciner : si incertain, dire unverifiable.

FORMAT — JSON UNIQUEMENT :
{{
  "claim": {{"text": "...", "category": "...", "verifiable": true}},
  "verdict": "supported|contradicted|unverifiable",
  "evidence": "citation courte des sources qui appuie le verdict (max 300 chars)",
  "confidence": 0.0..1.0
}}

Affirmation a verifier :
"{claim_text}"

Category : {claim_category}

Sources disponibles :
---
{sources}
---
"""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class VerificationReport:
    results: List[VerificationResult] = field(default_factory=list)
    claims_extracted: int = 0

    @property
    def supported(self) -> int:
        return sum(1 for r in self.results if r.verdict == "supported")

    @property
    def contradicted(self) -> int:
        return sum(1 for r in self.results if r.verdict == "contradicted")

    @property
    def unverifiable(self) -> int:
        return sum(1 for r in self.results if r.verdict == "unverifiable")

    @property
    def verification_ratio(self) -> float:
        """supported / total. 1.0 = tout verifie, 0.0 = rien verifie."""
        if not self.results:
            return 0.0
        return self.supported / len(self.results)

    def summary(self) -> str:
        return (
            f"CoVe: {self.supported} supported, {self.contradicted} contradicted, "
            f"{self.unverifiable} unverifiable (ratio={self.verification_ratio:.2%})"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class VerifierAgent:
    """Execute le pattern Chain-of-Verification."""

    def __init__(
        self,
        claim_extractor_llm: Callable[[str], str],
        verifier_llm: Callable[[str], str],
        max_claims: int = 8,
    ):
        self.claim_extractor_llm = claim_extractor_llm
        self.verifier_llm = verifier_llm
        self.max_claims = max_claims

    def _extract_claims(self, text: str) -> List[Claim]:
        prompt = EXTRACT_CLAIMS_PROMPT.format(text=text[:3000])
        try:
            raw = self.claim_extractor_llm(prompt)
            data = parse_llm_json(raw)
            claims_raw = data.get("claims", []) if isinstance(data, dict) else []
            claims: List[Claim] = []
            for cr in claims_raw[: self.max_claims]:
                if not isinstance(cr, dict):
                    continue
                try:
                    claims.append(Claim(**cr))
                except Exception:
                    # On tolere les champs manquants
                    try:
                        claims.append(
                            Claim(
                                text=str(cr.get("text", ""))[:300],
                                category=cr.get("category", "other"),
                                verifiable=bool(cr.get("verifiable", True)),
                            )
                        )
                    except Exception:
                        continue
            return claims
        except ValueError as e:
            logger.warning("Extract claims echec: %s", e)
            return []

    def _verify_one(self, claim: Claim, sources: str) -> Optional[VerificationResult]:
        prompt = VERIFY_CLAIM_PROMPT.format(
            claim_text=claim.text,
            claim_category=claim.category,
            sources=sources[:6000],
        )
        try:
            return structured_call(
                self.verifier_llm,
                prompt,
                VerificationResult,
                max_retries=1,
            )
        except ValueError as e:
            logger.warning("Verify echec pour '%s': %s", claim.text[:50], e)
            return VerificationResult(
                claim=claim,
                verdict="unverifiable",
                evidence="",
                confidence=0.0,
            )

    def verify(self, text: str, sources: str) -> VerificationReport:
        """Execute le pipeline CoVe complet sur un texte."""
        claims = self._extract_claims(text)
        report = VerificationReport(results=[], claims_extracted=len(claims))
        for claim in claims:
            if not claim.verifiable:
                continue
            vr = self._verify_one(claim, sources)
            if vr is not None:
                report.results.append(vr)
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def verification_ratio_to_confidence_multiplier(ratio: float) -> float:
    """
    Ratio [0,1] -> multiplicateur de confiance [0.5, 1.0].
    0.0 -> 0.50 (penalty max)
    0.5 -> 0.75
    1.0 -> 1.00
    """
    r = max(0.0, min(1.0, ratio))
    return 0.5 + 0.5 * r


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mock LLM pour extraction de claims
    def extractor(prompt: str) -> str:
        return json.dumps(
            {
                "claims": [
                    {"text": "Apple Q3 revenue = 89.5B USD", "category": "numeric", "verifiable": True},
                    {"text": "Tim Cook est CEO d'Apple", "category": "entity", "verifiable": True},
                    {"text": "Revenue guidance a ete relevee de 2%", "category": "numeric", "verifiable": True},
                    {"text": "Le marche va monter", "category": "other", "verifiable": False},  # doit etre skip
                ]
            }
        )

    # Mock verifier : supporte les 2 premiers, contredit le 3eme, unverifiable sur autres
    call_count = {"n": 0}

    def verifier(prompt: str) -> str:
        call_count["n"] += 1
        # On extrait la ligne "Affirmation a verifier :" pour matcher sur le claim seul
        import re as _re

        m = _re.search(r'Affirmation a verifier :\s*"([^"]+)"', prompt)
        claim_text = m.group(1) if m else ""
        if "89.5" in claim_text:
            return json.dumps(
                {
                    "claim": {"text": "Apple Q3 revenue = 89.5B USD", "category": "numeric", "verifiable": True},
                    "verdict": "supported",
                    "evidence": "Article : 'revenue of $89.5 billion'",
                    "confidence": 0.95,
                }
            )
        if "Tim Cook" in claim_text:
            return json.dumps(
                {
                    "claim": {"text": "Tim Cook est CEO d'Apple", "category": "entity", "verifiable": True},
                    "verdict": "supported",
                    "evidence": "Article : 'CEO Tim Cook'",
                    "confidence": 0.99,
                }
            )
        if "2%" in claim_text:
            return json.dumps(
                {
                    "claim": {
                        "text": "Revenue guidance a ete relevee de 2%",
                        "category": "numeric",
                        "verifiable": True,
                    },
                    "verdict": "contradicted",
                    "evidence": "Article : 'guidance raised 3%' - le chiffre est 3%, pas 2%",
                    "confidence": 0.90,
                }
            )
        return json.dumps(
            {
                "claim": {"text": "X", "category": "other", "verifiable": True},
                "verdict": "unverifiable",
                "evidence": "",
                "confidence": 0.0,
            }
        )

    verifier_agent = VerifierAgent(
        claim_extractor_llm=extractor,
        verifier_llm=verifier,
    )

    thesis = (
        "Apple doit etre achete : Q3 revenue 89.5B USD, guidance relevee de 2%, "
        "et Tim Cook en CEO rassure. Le marche va monter."
    )
    sources = (
        "Apple Inc. reported quarterly revenue of $89.5 billion, above consensus. "
        "CEO Tim Cook said the services segment grew strongly. Guidance was "
        "raised 3% for the next quarter."
    )

    report = verifier_agent.verify(text=thesis, sources=sources)
    print(report.summary())
    logger.info(f"Extracted {report.claims_extracted} claims, verified {len(report.results)}")
    for r in report.results:
        logger.info(f"  [{r.verdict:14}] conf={r.confidence:.2f}  {r.claim.text[:60]}")
        if r.evidence:
            logger.info(f"      evidence: {r.evidence[:100]}")

    mult = verification_ratio_to_confidence_multiplier(report.verification_ratio)
    logger.info(f"\nConfidence multiplier applicable : {mult:.2f}")
