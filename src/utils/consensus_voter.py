"""
consensus_voter.py -- Vote majoritaire pour la reproductibilité stochastique
=============================================================================
OBJECTIF :
  Les LLMs sont non-déterministes (température > 0). Le même article
  peut produire "Achat", "Neutre", "Achat" sur 3 runs successifs.

  Ce module lance le pipeline N fois et calcule :
    1. Le signal majoritaire (vote pondéré par le risk_score YOLO)
    2. La variance inter-run (mesure de stabilité)
    3. Un flag "INSTABLE" si le consensus est trop faible

PRINCIPE DE PONDÉRATION :
  Un run avec risk_score = 0.15 (FAIBLE → signal fiable) pèse plus lourd
  dans le vote qu'un run avec risk_score = 0.72 (signal peu fiable).
  → On ne fait pas une moyenne bête : on fait un vote pondéré par la confiance.

USAGE :
  from src.utils.consensus_voter import run_with_majority_vote

  result = await run_with_majority_vote(
      article_text, ticker, contexte_marche, absa_result, n_runs=3
  )
  print(result["signal"])    # Signal majoritaire
  print(result["variance"])  # 0.0 = unanime, 1.0 = désaccord total
  print(result["stable"])    # False si trop de désaccord
"""

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.utils.ese_entropy import ESEResult, compute_inter_run_ese

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


@dataclass
class SingleRunResult:
    """Résultat d'un seul run du pipeline débat."""

    run_id: int
    signal: str
    argument: str
    risk_level: str
    risk_score: float
    impact_strength: float
    consensus_rate: float
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.signal in ("Achat", "Vente", "Neutre")

    @property
    def confidence_weight(self) -> float:
        """
        Poids du run dans le vote : inversement proportionnel au risk_score.
        Un run FAIBLE-risque (faible score) est plus fiable → pèse plus lourd.
        """
        return max(0.01, 1.0 - self.risk_score)


@dataclass
class VoteResult:
    """Résultat du vote majoritaire sur N runs."""

    signal: str  # Signal gagnant après vote pondéré
    confidence: float  # Fraction du poids qui a voté pour le gagnant (0-1)
    variance: float  # Variance inter-run (0=unanime, 1=désaccord total)
    stable: bool  # True si variance < SEUIL_INSTABILITE
    n_runs_total: int
    n_runs_valid: int
    vote_breakdown: dict  # {"Achat": 2, "Vente": 0, "Neutre": 1}
    best_argument: str  # Argument du run ayant le plus fort poids
    runs: list[SingleRunResult] = field(default_factory=list)
    instability_note: Optional[str] = None
    ese_result: Optional[ESEResult] = None  # Entropie sémantique inter-runs


SEUIL_INSTABILITE = 0.33  # Variance > 33% → signal marqué instable


# ---------------------------------------------------------------------------
# Exécution d'un run unique
# ---------------------------------------------------------------------------


async def _run_single(
    run_id: int,
    article_text: str,
    ticker: str,
    contexte_marche: dict,
    absa_result: dict,
) -> SingleRunResult:
    """Exécute un run unique du pipeline débat de façon asynchrone."""
    try:
        from src.agents.agent_debat import workflow_debat_actualite
        from src.pipelines.agent_pipeline import _calculer_metrics_objectives
        from src.utils.yolo_classifier import classify_risk

        decision = workflow_debat_actualite.invoke(
            {
                "texte_article": article_text,
                "ticker_symbol": ticker,
                "contexte_marche": contexte_marche,
                "absa_result": absa_result,
            }
        )

        signal = decision.get("signal", "Neutre")
        argument = decision.get("argument_dominant", "")
        scratchpad = decision.get("scratchpad_xml", "")

        # Score FinBERT simulé (en mode voter on n'a pas accès au vrai FinBERT score)
        # On utilise une valeur neutre — l'important est la cohérence inter-runs
        score_finbert_sim = 0.70
        consensus_rate, impact_strength = _calculer_metrics_objectives(signal, score_finbert_sim, absa_result)

        yolo = classify_risk(
            signal_final=signal,
            consensus_rate=consensus_rate,
            impact_strength=impact_strength,
            scratchpad_xml=scratchpad,
            absa_result=absa_result,
            score_finbert=score_finbert_sim,
            contexte_marche=contexte_marche,
        )

        return SingleRunResult(
            run_id=run_id,
            signal=signal,
            argument=argument,
            risk_level=yolo.risk_level,
            risk_score=yolo.risk_score,
            impact_strength=impact_strength,
            consensus_rate=consensus_rate,
        )

    except Exception as e:
        logger.error("[Voter] Run %d échoué: %s", run_id, e)
        return SingleRunResult(
            run_id=run_id,
            signal="Neutre",
            argument="",
            risk_level="ELEVE",
            risk_score=1.0,
            impact_strength=0.0,
            consensus_rate=0.0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Vote majoritaire pondéré
# ---------------------------------------------------------------------------


def _compute_weighted_vote(runs: list[SingleRunResult]) -> VoteResult:
    """
    Calcule le signal majoritaire par vote pondéré par confidence_weight.
    """
    valid_runs = [r for r in runs if r.is_valid]
    if not valid_runs:
        return VoteResult(
            signal="Neutre",
            confidence=0.0,
            variance=1.0,
            stable=False,
            n_runs_total=len(runs),
            n_runs_valid=0,
            vote_breakdown={},
            best_argument="Aucun run valide",
            runs=runs,
            instability_note="Tous les runs ont échoué — vérifier les clés API et le modèle.",
        )

    # Poids total et poids par signal
    total_weight = sum(r.confidence_weight for r in valid_runs)
    weights_by_signal: dict[str, float] = {}
    for r in valid_runs:
        weights_by_signal[r.signal] = weights_by_signal.get(r.signal, 0.0) + r.confidence_weight

    # Signal gagnant = signal avec le plus de poids
    winner = max(weights_by_signal, key=weights_by_signal.get)
    winner_share = weights_by_signal[winner] / total_weight

    # Breakdown en nombre de runs (lisible)
    vote_breakdown = Counter(r.signal for r in valid_runs)

    # Variance inter-run : 0 si unanimité, 1 si désaccord total
    n_signals = len(set(r.signal for r in valid_runs))
    n_total = len(valid_runs)
    variance = round(1.0 - (vote_breakdown.most_common(1)[0][1] / n_total), 3)

    # Meilleur argument = run avec le plus fort poids parmi les gagnants
    winner_runs = [r for r in valid_runs if r.signal == winner]
    best_run = max(winner_runs, key=lambda r: r.confidence_weight)

    # ---- ESE Inter-runs — Entropie Sémantique d'Ensemble (#3) ----
    # Mesure si les N runs raisonnent sur des réalités sémantiques différentes
    # Un ESE élevé = les modèles ont des raisonnements fondamentalement divergents
    all_arguments = [r.argument for r in valid_runs if r.argument]
    ese_result = compute_inter_run_ese(all_arguments) if len(all_arguments) >= 2 else None
    if ese_result and ese_result.is_high_divergence:
        logger.warning(
            "[Voter ESE] Désaccord épistémique inter-runs élevé (ESE=%.3f) — "
            "les modèles raisonnent sur des bases sémantiques différentes",
            ese_result.ese_score,
        )

    # Note d'instabilité
    note = None
    if variance > SEUIL_INSTABILITE:
        minority_signals = [s for s in vote_breakdown if s != winner]
        note = (
            f"Signal INSTABLE (variance={variance:.0%}). "
            f"{vote_breakdown[winner]}/{n_total} runs disent '{winner}', "
            f"mais {[(s, vote_breakdown[s]) for s in minority_signals]} en désaccord. "
            f"Interpréter avec prudence ou relancer avec plus de runs."
        )

    return VoteResult(
        signal=winner,
        confidence=round(winner_share, 3),
        variance=variance,
        stable=variance <= SEUIL_INSTABILITE,
        n_runs_total=len(runs),
        n_runs_valid=len(valid_runs),
        vote_breakdown=dict(vote_breakdown),
        best_argument=best_run.argument,
        runs=runs,
        instability_note=note,
        ese_result=ese_result,
    )


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


async def run_with_majority_vote_async(
    article_text: str,
    ticker: str,
    contexte_marche: dict,
    absa_result: dict,
    n_runs: int = 3,
) -> VoteResult:
    """
    Lance N runs du pipeline en parallèle et retourne le vote majoritaire.
    Version asynchrone (recommandée pour n_runs >= 3 — réduit la latence).

    Args:
        n_runs : 3 = rapide/économique, 5 = plus fiable mais 5x le coût API
    """
    logger.info("[Voter] Lancement de %d runs parallèles pour %s...", n_runs, ticker)

    tasks = [_run_single(i + 1, article_text, ticker, contexte_marche, absa_result) for i in range(n_runs)]
    runs = await asyncio.gather(*tasks)

    result = _compute_weighted_vote(list(runs))

    # Log du résultat
    logger.info(
        "[Voter] %s | Gagnant: %s (%.0f%% du poids) | Variance: %.0f%% | Stable: %s",
        ticker,
        result.signal,
        result.confidence * 100,
        result.variance * 100,
        result.stable,
    )
    if result.instability_note:
        logger.warning("[Voter] %s", result.instability_note)

    return result


def run_with_majority_vote(
    article_text: str,
    ticker: str,
    contexte_marche: dict,
    absa_result: dict,
    n_runs: int = 3,
) -> VoteResult:
    """
    Version synchrone (compatible avec le pipeline existant non-async).
    Lance les runs séquentiellement — plus lent mais plus simple à intégrer.
    """
    logger.info("[Voter] Lancement de %d runs séquentiels pour %s...", n_runs, ticker)
    runs = []
    for i in range(n_runs):
        run = asyncio.run(_run_single(i + 1, article_text, ticker, contexte_marche, absa_result))
        runs.append(run)
        logger.info("[Voter] Run %d/%d: signal=%s, risk=%s", i + 1, n_runs, run.signal, run.risk_level)

    return _compute_weighted_vote(runs)


def format_vote_report(result: VoteResult) -> str:
    """Formate un rapport lisible du vote."""
    lines = [
        f"  Signal majoritaire : {result.signal} (confiance: {result.confidence:.0%})",
        f"  Variance inter-run : {result.variance:.0%} {'[STABLE]' if result.stable else '[INSTABLE]'}",
        f"  Runs valides       : {result.n_runs_valid}/{result.n_runs_total}",
        f"  Répartition votes  : {result.vote_breakdown}",
    ]
    if result.ese_result:
        ese_tag = "[⚠ DIVERGENCE EPISTEMIQUE]" if result.ese_result.is_high_divergence else "[OK]"
        lines.append(
            f"  ESE inter-runs     : {result.ese_result.ese_score:.3f} {ese_tag} "
            f"(trigrammes uniques: {result.ese_result.total_unique_grams})"
        )
    if result.instability_note:
        lines.append(f"  Note : {result.instability_note}")
    lines.append(f"  Argument retenu    : {result.best_argument[:100]}...")
    return "\n".join(lines)
