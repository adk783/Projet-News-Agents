"""
ese_entropy.py — Ensemble Semantic Entropy (ESE)

Objectif :
  Mesurer le DÉSACCORD ÉPISTÉMIQUE réel entre les agents du débat, au-delà
  des simples labels "Achat/Vente/Neutre".

  L'ESE calcule l'entropie de Shannon sur la distribution des n-grammes
  distincts extraits des arguments de chaque agent. Un ESE élevé signifie
  que les agents ne partagent pas le même vocabulaire conceptuel — ils
  raisonnent sur des réalités différentes → désaccord épistémique profond.

Différence avec la variance Inter-run (consensus_voter.py) :
  - Variance inter-run : mesure la divergence des LABELS entre runs
  - ESE intra-débat    : mesure la divergence des ARGUMENTS sur un seul run
                         (utilisable sans multi-run, donc applicable partout)
  - ESE inter-run      : agrège les distributions de modèles hétérogènes
                         (utilisé dans consensus_voter.py si n_runs >= 2)

Formule ESE :
  Pour chaque agent i, on extrait son ensemble de trigrammes de mots Gi.
  On construit la distribution globale D sur l'union de tous les Gi.
  H(ESE) = -Σ p(gram) * log2(p(gram)) pour chaque gram unique

  Normalisé en [0, 1] par H_max = log2(|total_unique_grams|).

Interprétation :
  ESE = 0.0 → Les agents utilisent exactement le même vocabulaire conceptuel
               → Consensus rhétorique (risque d'écho-chambre)
  ESE = 1.0 → Les agents raisonnent sur des bases complètement différentes
               → Désaccord épistémique profond → activer décote de confiance
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paramètre
# ---------------------------------------------------------------------------

ESE_HIGH_THRESHOLD = 0.70  # ESE > 0.70 → désaccord épistémique élevé


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------


@dataclass
class ESEResult:
    """Résultat du calcul d'entropie sémantique d'ensemble."""

    ese_score: float  # Entropie normalisée [0.0, 1.0]
    raw_entropy: float  # Entropie de Shannon brute (bits)
    total_unique_grams: int  # Nombre de trigrammes uniques
    agent_gram_counts: dict[str, int]  # Nb de trigrammes par agent
    is_high_divergence: bool  # ESE > ESE_HIGH_THRESHOLD
    mode: str  # "intra_debate" ou "inter_run"


# ---------------------------------------------------------------------------
# Extraction de trigrammes de mots
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """
    Tokenisation simple : minuscules, suppression ponctuation, split.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]  # Mots > 2 chars


def _extract_trigrams(text: str) -> list[tuple[str, str, str]]:
    """
    Extrait les trigrammes de mots d'un texte tokenisé.
    Exemple : "apple beats expectations" → [("apple", "beats", "expectations")]
    """
    tokens = _tokenize(text)
    if len(tokens) < 3:
        # Fallback : bigrammes si texte trop court
        return [(tokens[i], tokens[i + 1], "") for i in range(len(tokens) - 1)]
    return [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]


# ---------------------------------------------------------------------------
# Calcul ESE intra-débat (mode principal)
# ---------------------------------------------------------------------------


def compute_intra_debate_ese(agent_arguments: dict[str, str]) -> ESEResult:
    """
    Calcule l'ESE intra-débat : entropie sur les arguments des 3 agents
    d'un seul run.

    Args:
        agent_arguments : {"Haussier": "texte...", "Baissier": "texte...", "Neutre": "texte..."}

    Returns:
        ESEResult avec le score d'entropie normalisé
    """
    if not agent_arguments:
        return ESEResult(
            ese_score=0.0,
            raw_entropy=0.0,
            total_unique_grams=0,
            agent_gram_counts={},
            is_high_divergence=False,
            mode="intra_debate",
        )

    # Extraction des trigrammes par agent
    agent_grams: dict[str, Counter] = {}
    for agent_name, text in agent_arguments.items():
        if text and text.strip():
            grams = _extract_trigrams(text)
            agent_grams[agent_name] = Counter(grams)
        else:
            agent_grams[agent_name] = Counter()

    # Distribution globale : union de tous les trigrammes
    global_counter: Counter = Counter()
    for counter in agent_grams.values():
        global_counter.update(counter)

    total_grams = sum(global_counter.values())
    if total_grams == 0:
        return ESEResult(
            ese_score=0.0,
            raw_entropy=0.0,
            total_unique_grams=0,
            agent_gram_counts={n: 0 for n in agent_arguments},
            is_high_divergence=False,
            mode="intra_debate",
        )

    # Entropie de Shannon
    entropy = 0.0
    for count in global_counter.values():
        p = count / total_grams
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalisation par H_max = log2(nombre de trigrammes uniques)
    n_unique = len(global_counter)
    h_max = math.log2(n_unique) if n_unique > 1 else 1.0
    ese_score = round(entropy / h_max, 4) if h_max > 0 else 0.0

    agent_gram_counts = {n: sum(c.values()) for n, c in agent_grams.items()}

    logger.debug(
        "[ESE Intra-débat] Score=%.3f | H=%.3f bits | H_max=%.3f | Trigrammes uniques=%d",
        ese_score,
        entropy,
        h_max,
        n_unique,
    )

    return ESEResult(
        ese_score=ese_score,
        raw_entropy=round(entropy, 4),
        total_unique_grams=n_unique,
        agent_gram_counts=agent_gram_counts,
        is_high_divergence=ese_score > ESE_HIGH_THRESHOLD,
        mode="intra_debate",
    )


# ---------------------------------------------------------------------------
# Calcul ESE inter-runs (pour consensus_voter.py)
# ---------------------------------------------------------------------------


def compute_inter_run_ese(run_arguments: list[str]) -> ESEResult:
    """
    Calcule l'ESE inter-runs sur les arguments dominants de N runs.

    Chaque run a produit un argument dominant (string). On mesure si ces
    arguments raisonnent sur les mêmes concepts ou sur des réalités différentes.

    Args:
        run_arguments : liste d'arguments dominants (un par run)

    Returns:
        ESEResult avec le score d'entropie normalisé
    """
    if len(run_arguments) < 2:
        return ESEResult(
            ese_score=0.0,
            raw_entropy=0.0,
            total_unique_grams=0,
            agent_gram_counts={},
            is_high_divergence=False,
            mode="inter_run",
        )

    # Extraction des trigrammes par run
    run_grams: list[Counter] = [Counter(_extract_trigrams(arg)) for arg in run_arguments]

    # Distribution globale
    global_counter: Counter = Counter()
    for counter in run_grams:
        global_counter.update(counter)

    total_grams = sum(global_counter.values())
    if total_grams == 0:
        return ESEResult(
            ese_score=0.0,
            raw_entropy=0.0,
            total_unique_grams=0,
            agent_gram_counts={f"run_{i + 1}": 0 for i in range(len(run_arguments))},
            is_high_divergence=False,
            mode="inter_run",
        )

    # Entropie de Shannon
    entropy = 0.0
    for count in global_counter.values():
        p = count / total_grams
        if p > 0:
            entropy -= p * math.log2(p)

    n_unique = len(global_counter)
    h_max = math.log2(n_unique) if n_unique > 1 else 1.0
    ese_score = round(entropy / h_max, 4) if h_max > 0 else 0.0

    agent_gram_counts = {f"run_{i + 1}": sum(c.values()) for i, c in enumerate(run_grams)}

    logger.debug(
        "[ESE Inter-runs] Score=%.3f | H=%.3f bits | Runs=%d | Trigrammes uniques=%d",
        ese_score,
        entropy,
        len(run_arguments),
        n_unique,
    )

    return ESEResult(
        ese_score=ese_score,
        raw_entropy=round(entropy, 4),
        total_unique_grams=n_unique,
        agent_gram_counts=agent_gram_counts,
        is_high_divergence=ese_score > ESE_HIGH_THRESHOLD,
        mode="inter_run",
    )


# ---------------------------------------------------------------------------
# Extraction des arguments depuis le scratchpad XML
# ---------------------------------------------------------------------------


def extract_agent_arguments_from_scratchpad(scratchpad_xml: str) -> dict[str, str]:
    """
    Extrait les arguments concaténés de chaque agent depuis le scratchpad XML.

    Retourne : {"Haussier": "arg1 arg2 ...", "Baissier": "...", "Neutre": "..."}
    Utilisé pour le calcul ESE intra-débat depuis agent_debat.py.
    """
    import re as _re

    section_pattern = _re.compile(r'<section agent="([^"]+)">(.*?)</section>', _re.DOTALL | _re.IGNORECASE)
    arg_pattern = _re.compile(r"<argument>(.*?)</argument>", _re.DOTALL | _re.IGNORECASE)

    arguments: dict[str, str] = {}
    for section_match in section_pattern.finditer(scratchpad_xml):
        agent_name = section_match.group(1)
        section_content = section_match.group(2)
        # Concatène tous les <argument> de la section
        args = arg_pattern.findall(section_content)
        arguments[agent_name] = " ".join(args)

    return arguments


def ese_to_confidence_factor(ese_result: ESEResult) -> float:
    """
    Convertit l'ESE en facteur de confiance pour le consensus agent.

    ESE faible  (< 0.4)  → facteur = 1.0 (plein poids — peu de désaccord)
    ESE moyen   (0.4-0.7) → facteur linéaire décroissant
    ESE élevé   (> 0.7)  → facteur = 0.6 (décote significative)

    Returns:
        float [0.6, 1.0] à multiplier par la confiance finale
    """
    if ese_result.ese_score < 0.4:
        return 1.0
    elif ese_result.ese_score > ESE_HIGH_THRESHOLD:
        return 0.60
    else:
        # Interpolation linéaire entre 1.0 et 0.60 dans [0.4, 0.7]
        t = (ese_result.ese_score - 0.4) / (ESE_HIGH_THRESHOLD - 0.4)
        return round(1.0 - (t * 0.40), 4)
