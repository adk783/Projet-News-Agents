"""
htc_calibrator.py — Holistic Trajectory Calibration (HTC)

Objectif :
  Extraire un score de calibration unique depuis la TRAJECTOIRE COMPLÈTE
  des scores de confiance du débat multi-agent, sur 3 dimensions :

  1. Micro-stabilité  : volatilité tour-à-tour (les agents changent-ils
                        violemment d'un tour à l'autre ?)
  2. Macro-dynamique  : tendance globale (convergence ou divergence générale
                        sur l'ensemble du débat ?)
  3. Position finale  : niveau absolu de confiance en fin de débat
                        (les agents finissent-ils avec un score fort ?)

Score HTC final :
  HTC ∈ [0.0, 1.0]
  0.0 = trajectoire très instable / confiance finale faible / divergente
  1.0 = trajectoire stable, convergente, confiance finale élevée

Intégration :
  - Appelé par yolo_classifier.py après extraction des confiances
  - Le score est inséré dans le scratchpad XML sous <Calibrated_HTC_Score>
  - Remplace partiellement confidence_std dans le calcul du risque global
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------


@dataclass
class HTCResult:
    """Résultat de la calibration holiste de trajectoire."""

    htc_score: float  # Score global [0.0, 1.0]
    micro_stability: float  # Stabilité tour-à-tour [0.0, 1.0]
    macro_dynamic: float  # Score de convergence macro [0.0, 1.0]
    final_position: float  # Confiance finale normalisée [0.0, 1.0]
    n_samples: int  # Nombre de scores analysés
    xml_tag: str  # Balise XML prête à injecter dans le scratchpad


# ---------------------------------------------------------------------------
# Extraction des trajectoires par agent depuis le scratchpad XML
# ---------------------------------------------------------------------------


def extract_trajectories_by_agent(scratchpad_xml: str) -> dict[str, list[float]]:
    """
    Extrait les scores de confiance tour-par-tour, groupés par agent.

    Format attendu dans le scratchpad :
      [Tour N] ... [confiance: 0.85]

    Retourne :
      {
        "Haussier": [0.8, 0.85, 0.9],
        "Baissier":  [0.9, 0.75, 0.6],
        "Neutre":    [0.7, 0.72, 0.68],
      }
    """
    # Pattern pour extraire agent + tour + confiance depuis les sections XML
    section_pattern = re.compile(r'<section agent="([^"]+)">(.*?)</section>', re.DOTALL | re.IGNORECASE)
    conf_pattern = re.compile(r"\[Tour\s*(\d+)\].*?\[confiance:\s*([\d.]+)\]", re.DOTALL)

    trajectories: dict[str, list[float]] = {}

    for section_match in section_pattern.finditer(scratchpad_xml):
        agent_name = section_match.group(1)
        section_content = section_match.group(2)

        # Extrait les (tour, confiance) dans l'ordre
        tour_scores: list[tuple[int, float]] = []
        for m in conf_pattern.finditer(section_content):
            tour = int(m.group(1))
            score = float(m.group(2))
            tour_scores.append((tour, score))

        # Trie par numéro de tour et garde uniquement les scores
        tour_scores.sort(key=lambda x: x[0])
        trajectories[agent_name] = [s for _, s in tour_scores]

    return trajectories


# ---------------------------------------------------------------------------
# Dimension 1 — Micro-stabilité
# Mesure la volatilité tour-à-tour : delta entre tours consécutifs
# ---------------------------------------------------------------------------


def _compute_micro_stability(trajectories: dict[str, list[float]]) -> float:
    """
    Calcule la stabilité micro (anti-volatilité court terme).

    Pour chaque agent disposant d'au moins 2 tours, on calcule l'écart-type
    des deltas entre tours consécutifs.
    Un std élevé = l'agent change violemment de confiance = instable.

    Retourne [0.0, 1.0] : 1.0 = très stable
    """
    all_deltas: list[float] = []
    for scores in trajectories.values():
        if len(scores) < 2:
            continue
        deltas = [abs(scores[i + 1] - scores[i]) for i in range(len(scores) - 1)]
        all_deltas.extend(deltas)

    if not all_deltas:
        return 0.5  # Pas assez de données, valeur neutre

    mean_delta = sum(all_deltas) / len(all_deltas)
    # mean_delta = 0.0 → stability = 1.0 | mean_delta = 0.5 → stability ≈ 0.0
    stability = max(0.0, 1.0 - (mean_delta * 2))
    return round(stability, 4)


# ---------------------------------------------------------------------------
# Dimension 2 — Macro-dynamique
# Tendance globale : les agents convergent-ils vers un accord ?
# ---------------------------------------------------------------------------


def _compute_macro_dynamic(trajectories: dict[str, list[float]]) -> float:
    """
    Mesure la dynamique macro du débat.

    Approche :
      - Pour chaque agent, on calcule la pente (premier → dernier tour)
      - Convergence si les agents HAUSSIER et BAISSIER se rapprochent
        de la médiane (leurs extrêmes se réduisent)
      - Divergence si les positions s'éloignent

    Retourne [0.0, 1.0] : 1.0 = convergence saine, 0.0 = divergence totale
    """
    # Collecte les scores initiaux et finaux
    initial_scores: list[float] = []
    final_scores: list[float] = []

    for scores in trajectories.values():
        if len(scores) >= 2:
            initial_scores.append(scores[0])
            final_scores.append(scores[-1])

    if len(initial_scores) < 2:
        return 0.5

    # Range initial (écart max-min des premières positions)
    initial_range = max(initial_scores) - min(initial_scores)
    # Range final (écart max-min des positions finales)
    final_range = max(final_scores) - min(final_scores)

    if initial_range == 0.0:
        return 1.0  # Déjà unanimes dès le début

    # Ratio : à quel point le range a-t-il diminué ?
    convergence_ratio = 1.0 - (final_range / initial_range)
    # Clip entre 0 et 1 (peut être négatif si divergence)
    macro_dynamic = max(0.0, min(1.0, convergence_ratio))
    return round(macro_dynamic, 4)


# ---------------------------------------------------------------------------
# Dimension 3 — Position finale
# Niveau absolu de confiance en fin de débat
# ---------------------------------------------------------------------------


def _compute_final_position(trajectories: dict[str, list[float]]) -> float:
    """
    Calcule la confiance finale moyenne de tous les agents.
    Mesure si le débat se termine avec des agents "convaincus".

    Retourne [0.0, 1.0] : 1.0 = tous les agents finissent à confiance max
    """
    final_scores = [scores[-1] for scores in trajectories.values() if scores]
    if not final_scores:
        return 0.5
    return round(sum(final_scores) / len(final_scores), 4)


# ---------------------------------------------------------------------------
# Score HTC combiné
# ---------------------------------------------------------------------------


def compute_htc_score(
    scratchpad_xml: str,
    w_micro: float = 0.35,
    w_macro: float = 0.40,
    w_final: float = 0.25,
) -> HTCResult:
    """
    Calcule le score HTC complet depuis le scratchpad XML.

    Pondération par défaut :
      - Macro-dynamique  : 40% (la tendance globale est le signal le plus fort)
      - Micro-stabilité  : 35% (la cohérence interne est très importante)
      - Position finale  : 25% (utile mais peut être gonflée artificiellement)

    Args:
        scratchpad_xml : contenu XML du SharedScratchpad
        w_micro        : poids de la micro-stabilité
        w_macro        : poids de la macro-dynamique
        w_final        : poids de la position finale

    Returns:
        HTCResult avec le score HTC et ses composantes
    """
    trajectories = extract_trajectories_by_agent(scratchpad_xml)

    if not trajectories:
        logger.debug("[HTC] Aucune trajectoire extraite — scratchpad vide ou format incorrect.")
        result = HTCResult(
            htc_score=0.5,
            micro_stability=0.5,
            macro_dynamic=0.5,
            final_position=0.5,
            n_samples=0,
            xml_tag="<Calibrated_HTC_Score>0.5</Calibrated_HTC_Score>",
        )
        return result

    micro = _compute_micro_stability(trajectories)
    macro = _compute_macro_dynamic(trajectories)
    final = _compute_final_position(trajectories)

    # Score HTC pondéré
    htc = round((micro * w_micro) + (macro * w_macro) + (final * w_final), 4)

    total_samples = sum(len(v) for v in trajectories.values())

    xml_tag = f'<Calibrated_HTC_Score micro="{micro}" macro="{macro}" final_pos="{final}">{htc}</Calibrated_HTC_Score>'

    logger.debug(
        "[HTC] Score=%.3f | Micro=%.3f | Macro=%.3f | Final=%.3f | Agents=%s",
        htc,
        micro,
        macro,
        final,
        list(trajectories.keys()),
    )

    return HTCResult(
        htc_score=htc,
        micro_stability=micro,
        macro_dynamic=macro,
        final_position=final,
        n_samples=total_samples,
        xml_tag=xml_tag,
    )


def htc_to_risk_contribution(htc_result: HTCResult) -> float:
    """
    Convertit le score HTC en contribution au risque (sens inversé).

    HTC = 1.0 (excellent) → risk_contribution = 0.0
    HTC = 0.0 (catastrophique) → risk_contribution = 1.0

    Utilisé par yolo_classifier.py pour remplacer confidence_std.
    """
    return round(1.0 - htc_result.htc_score, 4)
