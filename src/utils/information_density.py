"""
information_density.py — Information Density (ID) Score

Objectif :
  Pénaliser la verbosité des arguments LLM en calculant un score de densité :

      ID = (Entités_uniques + Faits_numériques) / Mots_totaux

  Un argument verbeux sans faits concrets reçoit un ID faible → décote.
  Un argument concis bourré de chiffres reçoit un ID élevé → plein poids.

Définitions :
  - Entités_uniques :
      • Tickers boursiers (2-5 lettres MAJ, ex: AAPL, MSFT)
      • Acronymes financiers connus (EPS, EBITDA, FCF, P/E, ROE, etc.)
      • Noms propres capitalisés (Apple, Microsoft, Fed, etc.)
  - Faits_numériques :
      • Tout nombre accompagné d'une unité financière :
        "$X", "X%", "XB€", "Xbn", "Xbn$", "X milliards", etc.
      • Dates et périodes explicites (Q3 2024, FY2025, 2024, etc.)
  - Mots_totaux : tokenisation naïve (split sur espaces)

Seuil de décote :
  ID < ID_THRESHOLD (0.04) → confiance réduite de PENALTY_FACTOR (25%)
  L'argument reste dans le scratchpad, mais avec moins d'influence.

Usage :
  from src.utils.information_density import compute_id_score, apply_id_penalty

  score = compute_id_score(argument_text)
  adjusted_confidence = apply_id_penalty(confidence=0.85, id_score=score)
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

ID_THRESHOLD = 0.04  # En dessous → décote de confiance appliquée
PENALTY_FACTOR = 0.25  # Réduction de confiance pour argument peu dense

# Acronymes financiers reconnus comme "entités" (liste extensible)
FINANCIAL_ACRONYMS = {
    "EPS",
    "EBITDA",
    "FCF",
    "PE",
    "P/E",
    "ROE",
    "ROA",
    "ROIC",
    "IPO",
    "M&A",
    "ETF",
    "VIX",
    "CPI",
    "PPI",
    "GDP",
    "FED",
    "FMI",
    "BCE",
    "SEC",
    "FDIC",
    "CAGR",
    "WACC",
    "DCF",
    "YoY",
    "QoQ",
    "BPA",
    "RNE",
    "CA",
    "BFCPE",
    "S&P",
    "NASDAQ",
    "NYSE",
    "OPEC",
    "FOMC",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "H1",
    "H2",
    "FY",
    "TTM",
    "LTM",
    "NTM",
    "SaaS",
    "ESG",
    "SPAC",
}


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------


@dataclass
class IDScoreResult:
    """Résultat du calcul de densité d'information d'un argument."""

    id_score: float  # Score ID brut
    entities_count: int  # Nombre d'entités uniques détectées
    facts_count: int  # Nombre de faits numériques détectés
    word_count: int  # Nombre de mots total
    is_penalized: bool  # True si ID < seuil → décote appliquée
    entities_found: list[str]  # Debug : entités détectées


# ---------------------------------------------------------------------------
# Détection des entités
# ---------------------------------------------------------------------------


def _extract_entities(text: str) -> set[str]:
    """
    Extrait les entités uniques financières d'un texte.

    Catégories :
    1. Tickers boursiers : 2-5 lettres MAJ en isolation (AAPL, MSFT, GOOGL)
    2. Acronymes financiers connus (liste FINANCIAL_ACRONYMS)
    3. Noms propres capitalisés de 1+ mots consécutifs
    """
    entities = set()

    # 1. Tickers : mot de 2-5 lettres majuscules isolé (pas un acronyme de phrase)
    ticker_pattern = re.compile(r"\b([A-Z]{2,5})\b")
    for m in ticker_pattern.finditer(text):
        entities.add(m.group(1))

    # 2. Acronymes financiers explicitement listés
    words_upper = {w.strip(".,;:()[]") for w in text.split()}
    entities.update(FINANCIAL_ACRONYMS & words_upper)

    # 3. Noms propres : première lettre majuscule, pas en début de phrase
    # On détecte les mots en capitalisation au milieu d'une phrase
    proper_pattern = re.compile(r"(?<!\.\s)(?<!\A)\b([A-Z][a-z]{2,})\b")
    for m in proper_pattern.finditer(text):
        entities.add(m.group(1))

    return entities


# ---------------------------------------------------------------------------
# Détection des faits numériques
# ---------------------------------------------------------------------------


def _extract_numerical_facts(text: str) -> list[str]:
    """
    Extrait les faits numériques financiers d'un texte.

    Exemples de patterns détectés :
      - "$2.18", "€1.5bn", "2.3%", "12 milliards", "Q3 2024", "FY2025"
      - "14%", "+3.5%", "-12%", "15 billion", "200M$"
    """
    facts = []

    patterns = [
        # Montants en devise : $X, €X, $Xbn, XM$
        re.compile(r"[\$€£]\s*[\d,.]+\s*(?:bn|B|M|K|billion|million|milliard)?", re.IGNORECASE),
        # Pourcentages : 12%, +3.5%, -20%
        re.compile(r"[+-]?\s*[\d,.]+\s*%"),
        # Chiffres avec unités volumétriques : 15 milliards, 200 millions
        re.compile(r"\b[\d,.]+\s*(?:milliard|million|billion|trillion|bn|Bn|M|Md)\b", re.IGNORECASE),
        # Périodes financières : Q3 2024, FY2025, H1 2025
        re.compile(r"\b(?:Q[1-4]|H[12]|FY)\s*\d{4}\b", re.IGNORECASE),
        # Années explicites : 2023, 2024, 2025
        re.compile(r"\b(20[12]\d)\b"),
        # EPS / BPA avec valeur : "EPS de 2.18$", "BPA à 1.5"
        re.compile(r"\b(?:EPS|BPA|EPS)\s+(?:de|à|of)?\s*[\d,.]+", re.IGNORECASE),
    ]

    for pattern in patterns:
        facts.extend(pattern.findall(text))

    return facts


# ---------------------------------------------------------------------------
# Calcul du score ID
# ---------------------------------------------------------------------------


def compute_id_score(text: str) -> IDScoreResult:
    """
    Calcule le score de densité d'information d'un texte argument.

    Formule : ID = (Entités_uniques + Faits_numériques) / Mots_totaux

    Args:
        text : texte brut de l'argument (extrait du scratchpad)

    Returns:
        IDScoreResult avec le score et les composantes pour audit
    """
    if not text or not text.strip():
        return IDScoreResult(
            id_score=0.0, entities_count=0, facts_count=0, word_count=0, is_penalized=True, entities_found=[]
        )

    words = text.strip().split()
    word_count = max(len(words), 1)  # Évite division par zéro

    entities = _extract_entities(text)
    facts = _extract_numerical_facts(text)

    entities_unique = len(entities)
    facts_count = len(facts)

    id_score = round((entities_unique + facts_count) / word_count, 4)
    is_penalized = id_score < ID_THRESHOLD

    logger.debug(
        "[ID Score] %.4f (entités=%d, faits=%d, mots=%d) | Pénalisé=%s",
        id_score,
        entities_unique,
        facts_count,
        word_count,
        is_penalized,
    )

    return IDScoreResult(
        id_score=id_score,
        entities_count=entities_unique,
        facts_count=facts_count,
        word_count=word_count,
        is_penalized=is_penalized,
        entities_found=sorted(entities),
    )


# ---------------------------------------------------------------------------
# Application de la décote de confiance
# ---------------------------------------------------------------------------


def apply_id_penalty(confidence: float, id_result: IDScoreResult) -> float:
    """
    Applique la décote de confiance si l'argument est trop peu dense.

    La décote est proportionnelle à l'écart entre le seuil et le score :
      - Si ID = 0.0 (aucun fait) : décote pleine (PENALTY_FACTOR = 25%)
      - Si ID = 0.039 (juste en dessous du seuil) : décote minimale (~2%)
      - Si ID >= seuil : aucune décote

    Args:
        confidence : score de confiance original [0.0, 1.0]
        id_result  : résultat du calcul ID

    Returns:
        Confiance ajustée [0.0, 1.0]
    """
    if not id_result.is_penalized:
        return confidence

    # Décote proportionnelle : plus l'ID est bas, plus la pénalité est forte
    shortfall_ratio = max(0.0, (ID_THRESHOLD - id_result.id_score) / ID_THRESHOLD)
    effective_penalty = PENALTY_FACTOR * shortfall_ratio
    adjusted = round(confidence * (1.0 - effective_penalty), 4)

    logger.debug(
        "[ID Penalty] Confiance %.2f → %.2f (ID=%.4f, pénalité=%.1f%%)",
        confidence,
        adjusted,
        id_result.id_score,
        effective_penalty * 100,
    )

    return max(0.0, adjusted)


def format_id_xml_attr(id_result: IDScoreResult) -> str:
    """Formate l'ID score en attribut XML pour injection dans le scratchpad."""
    penalized_str = "true" if id_result.is_penalized else "false"
    return f'id_score="{id_result.id_score}" id_penalized="{penalized_str}"'
