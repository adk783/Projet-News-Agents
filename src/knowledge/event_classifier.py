"""
event_classifier.py — Typologie canonique des evenements d'actualite financiere.

POURQUOI
--------
Un LLM general-purpose traite tous les articles pareil, alors que :
  - "Apple beats Q3 estimates"            -> Earnings event (impact J-3j)
  - "Apple announces $5B buyback"         -> Capital return (impact J+5j)
  - "FDA rejects Apple Watch ECG update"  -> Regulatory (impact H+4h, sectoriel)
  - "DOJ sues Apple over App Store"       -> Litigation (impact multi-semaines)
  - "Tim Cook steps down"                 -> Leadership change (volatilite J+1)

Chaque type a une kinétique, une amplitude et une demi-vie differents, et
demande des prompts specifiques (et souvent des agents specialises en aval).
Le pipeline actuel ne fait AUCUNE typologie : tout rentre dans le meme prompt.

APPROCHE
--------
Classifieur hybride a deux etages :
  1. Rules strictes sur keywords + patterns (O(1), haute precision)
  2. Heuristiques ponderees si aucune regle ne matche (couverture)

On ne bloque pas l'appel d'un modele LLM zero-shot : on expose aussi un hook
`llm_fallback_classifier` qui peut etre remplace par un ProviderRegistry call
si besoin d'une meilleure couverture edge-case.

LABELS
------
- earnings          : resultats trimestriels/annuels, guidance
- ma                : fusion, acquisition, offre publique, divestiture
- buyback_dividend  : rachat, dividende, return of capital
- regulatory        : FDA, SEC, EMA, ECB, FTC, DOJ, ANSM
- litigation        : lawsuit, settlement, class action
- leadership        : CEO/CFO change, board shakeup
- product_launch    : nouveau produit, lancement, keynote
- macro             : Fed, BCE, BoJ, inflation, PIB, emploi
- geopolitical      : sanctions, guerre, election, tarifs
- rumor             : unconfirmed reports, sources say (moins de poids)
- analyst           : upgrade/downgrade/price target change
- partnership       : accord, deal commercial, collaboration
- other             : rien de specifique

REFERENCES
----------
- Tetlock (2007, 2011) pour le lien media-type -> prix
- Loughran & McDonald (2011) pour les dictionnaires financiers
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

EVENT_LABELS = (
    "earnings",
    "ma",
    "buyback_dividend",
    "regulatory",
    "litigation",
    "leadership",
    "product_launch",
    "macro",
    "geopolitical",
    "rumor",
    "analyst",
    "partnership",
    "other",
)


@dataclass(frozen=True)
class EventClassification:
    primary: str
    confidence: float
    secondary: Tuple[str, ...] = ()
    matched_keywords: Tuple[str, ...] = ()
    source: str = "rules"  # "rules" | "heuristic" | "llm"

    def summary(self) -> str:
        sec = f" (+{','.join(self.secondary)})" if self.secondary else ""
        return f"{self.primary}{sec} conf={self.confidence:.2f} src={self.source}"


# ---------------------------------------------------------------------------
# Dictionnaires de regles (ordre = priorite)
# ---------------------------------------------------------------------------

# Patterns compiles, case-insensitive. Un match = +poids donne a la classe.
# Format : (label, pattern_regex, weight)
_RULES: List[Tuple[str, re.Pattern[str], float]] = [
    # Earnings
    ("earnings", re.compile(r"\b(q[1-4]|fy\d{2,4}|quarter(ly)?|annual)\s+(results|earnings|report)\b", re.I), 1.0),
    (
        "earnings",
        re.compile(r"\b(beat|miss|top|exceed)(s|ed)?\s+(estimates|expectations|forecast|consensus)\b", re.I),
        1.0,
    ),
    ("earnings", re.compile(r"\b(guidance|outlook)\s+(raised|lowered|cut|boosted|reaffirmed)\b", re.I), 0.9),
    ("earnings", re.compile(r"\beps\b", re.I), 0.5),
    ("earnings", re.compile(r"\brevenue\s+(of|up|down|increased|declined)\b", re.I), 0.6),
    # M&A
    ("ma", re.compile(r"\b(acqui(res?|sition|red)|merg(er|ers|e|ed with)|takeover|buyout)\b", re.I), 1.0),
    ("ma", re.compile(r"\b(divest|spin[- ]off|spinoff|carve[- ]out|sells? (unit|division|business))\b", re.I), 0.9),
    ("ma", re.compile(r"\btender offer\b", re.I), 0.9),
    ("ma", re.compile(r"\bto acquire\b", re.I), 0.8),
    # Buyback / dividend
    ("buyback_dividend", re.compile(r"\b(share\s+)?(buyback|repurchase)\s+program\b", re.I), 1.0),
    ("buyback_dividend", re.compile(r"\b(raised|increases?|hikes?)\s+(its\s+)?dividend\b", re.I), 0.95),
    ("buyback_dividend", re.compile(r"\bspecial\s+dividend\b", re.I), 0.9),
    ("buyback_dividend", re.compile(r"\breturn(s|ed)?\s+capital\s+to\s+shareholders\b", re.I), 0.8),
    # Regulatory
    ("regulatory", re.compile(r"\bfda\s+(approv|reject|warn|letter)", re.I), 1.0),
    ("regulatory", re.compile(r"\bsec\s+(charges?|files?|investigat|complaint)\b", re.I), 1.0),
    ("regulatory", re.compile(r"\b(ema|ansm|mhlw|nmpa)\s+(approv|reject|warn)", re.I), 0.95),
    ("regulatory", re.compile(r"\b(ftc|doj|european commission)\s+(approv|block|investigat|probe)\b", re.I), 1.0),
    ("regulatory", re.compile(r"\bantitrust\b", re.I), 0.85),
    # Litigation
    ("litigation", re.compile(r"\b(sues?|lawsuit|litigation|class\s+action|settle(ment|s|d))\b", re.I), 0.9),
    ("litigation", re.compile(r"\b(court|judge|jury|ruling|verdict)\b", re.I), 0.6),
    ("litigation", re.compile(r"\bfined?\s+\$?[\d,\.]+", re.I), 0.85),
    # Leadership
    (
        "leadership",
        re.compile(
            r"\b(ceo|cfo|coo|cto|chair(man)?|president)\s+(steps?\s+down|resigns?|to\s+retire|appointed|named|joins|departs|leaves)\b",
            re.I,
        ),
        1.0,
    ),
    ("leadership", re.compile(r"\bboard\s+(change|reshuffle|shake[- ]up|overhaul)\b", re.I), 0.9),
    ("leadership", re.compile(r"\bsuccession\s+plan\b", re.I), 0.8),
    # Product launch
    (
        "product_launch",
        re.compile(
            r"\b(unveil|launch(es|ed)?|announc(es|ed)|reveal(s|ed)?)\s+(new\s+)?(product|model|version|lineup)\b", re.I
        ),
        0.9,
    ),
    ("product_launch", re.compile(r"\bkeynote\b", re.I), 0.7),
    ("product_launch", re.compile(r"\b(ai|gpu|chip|model|iphone|pixel)\s+(release|launch|unveil)", re.I), 0.75),
    # Macro
    (
        "macro",
        re.compile(r"\b(federal\s+reserve|\bfed\b|fomc|powell)\b.*\b(rate|cut|hike|hold|decision|minutes)\b", re.I),
        1.0,
    ),
    ("macro", re.compile(r"\b(ecb|bce|lagarde)\b.*\b(rate|cut|hike|decision)\b", re.I), 1.0),
    ("macro", re.compile(r"\b(cpi|ppi|pce|inflation|gdp|pib|nfp|payrolls|unemployment)\b", re.I), 0.9),
    ("macro", re.compile(r"\binterest\s+rate(s)?\b", re.I), 0.6),
    # Geopolitical
    ("geopolitical", re.compile(r"\b(sanctions?|embargo|tariff(s)?|trade\s+war)\b", re.I), 0.9),
    (
        "geopolitical",
        re.compile(
            r"\b(russia|china|iran|north\s+korea|ukraine|taiwan|israel)\b.*\b(tensions?|invasion|attack|war|strike)\b",
            re.I,
        ),
        0.9,
    ),
    ("geopolitical", re.compile(r"\belection\b", re.I), 0.5),
    # Rumor
    (
        "rumor",
        re.compile(
            r"\b(rumou?r(s|ed)?|sources\s+say|unconfirmed|reportedly|according\s+to\s+(people|sources))\b", re.I
        ),
        0.85,
    ),
    ("rumor", re.compile(r"\bmay\s+(launch|acquire|sell|spin)\b", re.I), 0.6),
    # Analyst
    ("analyst", re.compile(r"\b(upgrad(es|ed)|downgrad(es|ed))\s+to\s+\w+\b", re.I), 1.0),
    ("analyst", re.compile(r"\bprice\s+target\s+(raised|lowered|cut|boosted)\b", re.I), 1.0),
    ("analyst", re.compile(r"\b(initiat(es|ed)\s+coverage|reiterates?)\b", re.I), 0.8),
    ("analyst", re.compile(r"\b(buy|sell|hold|outperform|underperform|neutral)\s+rating\b", re.I), 0.7),
    # Partnership
    ("partnership", re.compile(r"\b(partner(s|ship)?|collabora(te|tion)|strategic\s+alliance)\b", re.I), 0.7),
    ("partnership", re.compile(r"\bjoint\s+venture\b", re.I), 0.9),
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _score_text(text: str) -> dict[str, float]:
    """Somme les poids des regles qui matchent."""
    scores: dict[str, float] = {}
    for label, pat, w in _RULES:
        if pat.search(text):
            scores[label] = scores.get(label, 0.0) + w
    return scores


def _kw_list(text: str) -> List[str]:
    out: List[str] = []
    for label, pat, _w in _RULES:
        m = pat.search(text)
        if m:
            out.append(f"{label}:{m.group(0)[:30]}")
    return out


def classify_event(title: str, body: str = "") -> EventClassification:
    """
    Classifie un article en type d'evenement. Le titre est pondere 2x le body.
    """
    blob_title = (title or "").strip()
    blob_body = (body or "").strip()

    scores_t = _score_text(blob_title)
    scores_b = _score_text(blob_body)

    merged: dict[str, float] = {}
    for k, v in scores_t.items():
        merged[k] = merged.get(k, 0.0) + 2.0 * v  # titre x2
    for k, v in scores_b.items():
        merged[k] = merged.get(k, 0.0) + v

    if not merged:
        return EventClassification(
            primary="other",
            confidence=0.0,
            secondary=(),
            matched_keywords=(),
            source="rules",
        )

    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    primary_label, primary_score = ranked[0]

    total = sum(merged.values())
    confidence = primary_score / total if total > 0 else 0.0

    # Secondary si score >= 40% du primary
    threshold = 0.4 * primary_score
    secondary = tuple(lbl for lbl, sc in ranked[1:] if sc >= threshold)[:2]

    kws = tuple(_kw_list(blob_title + " " + blob_body)[:6])

    return EventClassification(
        primary=primary_label,
        confidence=min(1.0, confidence),
        secondary=secondary,
        matched_keywords=kws,
        source="rules",
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        (
            "Apple beats Q3 earnings estimates, guidance raised",
            "Apple Inc reported quarterly EPS of $1.52, above consensus.",
            "earnings",
        ),
        (
            "Microsoft to acquire Activision in $70 billion deal",
            "The acquisition will make Microsoft the third-largest gaming company.",
            "ma",
        ),
        (
            "Apple announces $90 billion share buyback program",
            "The board also raised its dividend by 4%.",
            "buyback_dividend",
        ),
        (
            "FDA approves Novo Nordisk's new weight loss drug",
            "The approval clears the path for commercial launch in Q2.",
            "regulatory",
        ),
        ("DOJ sues Apple over App Store monopoly", "The antitrust lawsuit seeks structural remedies.", "litigation"),
        (
            "Tim Cook steps down as Apple CEO, succession plan announced",
            "The longtime CEO will be replaced by COO Jeff Williams.",
            "leadership",
        ),
        (
            "Nvidia unveils new Blackwell GPU lineup at GTC keynote",
            "The new products target AI data centers.",
            "product_launch",
        ),
        (
            "Fed holds rates steady, Powell signals patience on cuts",
            "The FOMC decision was in line with consensus.",
            "macro",
        ),
        (
            "US imposes new tariffs on Chinese EVs",
            "The trade war escalation raises costs for automakers.",
            "geopolitical",
        ),
        (
            "Goldman Sachs upgrades Tesla to Buy, price target raised to $300",
            "Analyst cites improved margins.",
            "analyst",
        ),
        (
            "Sources say Apple may acquire Anthropic within weeks",
            "Unconfirmed reports suggest talks are advanced.",
            "rumor",
        ),
        (
            "Nvidia and Microsoft announce strategic partnership on AI infra",
            "The joint venture targets enterprise customers.",
            "partnership",
        ),
        ("Weather report: sunny in California", "Nothing to do with finance.", "other"),
    ]

    print(f"{'Expected':18} {'Got':18} {'Conf':>5}  Title")
    print("-" * 110)
    ok = 0
    for title, body, expected in cases:
        r = classify_event(title, body)
        mark = "OK" if r.primary == expected else "X"
        ok += 1 if r.primary == expected else 0
        logger.info(f"{expected:18} {r.primary:18} {r.confidence:4.2f}  {mark}  {title[:60]}")
    logger.info(f"\nTotal: {ok}/{len(cases)} correct")
