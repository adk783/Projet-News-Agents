"""
bayesian_aggregator.py — Hierarchical Bayesian aggregation des agents de débat

OBJECTIF
--------
Remplacer l'extraction heuristique `consensus_rate / impact_strength` par une
agrégation bayésienne hiérarchique qui produit :
  1. Un posterior Beta(α, β) sur p(Achat)
  2. Une variance épistémique (désaccord inter-agents) séparée de la variance
     aléatoire (confiance intra-agent)
  3. Un facteur de scaling Kelly dérivé de cette variance
     (Medo-Pignatti-Pignatti 2013, Browne & Whitt 1996)

MODÈLE
------
Chaque agent k ∈ {Haussier, Baissier, Neutre} émet, au dernier tour :
    thesis_k ∈ {H, B, N}   (vote)
    conf_k   ∈ [0, 1]      (confidence auto-rapportée)

On convertit en pseudocounts Beta :
    p_k = P(Achat | agent k) = conf_k       si thesis_k = H
                              = 1 - conf_k   si thesis_k = B
                              = 0.5          si thesis_k = N
    α_k = w_k · conf_k         (pseudo-succès)
    β_k = w_k · (1 - conf_k)   (pseudo-échecs)

où w_k est la *capacité effective* de l'agent k (précision historique, peut
être dérivée d'une table de calibration). Par défaut w_k = 4 (~1 "trial" de
poids par tour sur 3 tours + 1 base).

POSTERIOR AGRÉGÉ :
    α_agg = α_prior + Σ α_k         (prior Beta(1, 1) = uniforme)
    β_agg = β_prior + Σ β_k

    E[p]   = α_agg / (α_agg + β_agg)
    Var[p] = α_agg · β_agg / ((α_agg + β_agg)² · (α_agg + β_agg + 1))

VARIANCE ÉPISTÉMIQUE :
    epi = Var[ p_k ]_{k=1..K}   (variance entre agents)
  vs ALÉATOIRE (moyenne des Var intra-agent sous Beta(conf·ν, (1-conf)·ν)).

KELLY SCALING :
    f_effective = f_Kelly · E[p] / (1 + κ · Var_total)

Ref : Lakshminarayanan et al. (2017) Deep Ensembles ; Raftery et al. (2005)
BMA ; Bates & Granger (1969) combining forecasts.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversion thesis/confidence → (α, β)
# ---------------------------------------------------------------------------


@dataclass
class AgentPosterior:
    name: str
    thesis: str  # HAUSSIER | BAISSIER | NEUTRE
    confidence: float  # [0, 1]
    weight: float  # pseudocount (capacité effective)
    alpha: float  # pseudo-succès
    beta: float  # pseudo-échecs

    @property
    def p_achat(self) -> float:
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5


def _thesis_to_p(thesis: str, confidence: float) -> float:
    """Normalise la confiance d'un agent en probabilité P(Achat)."""
    thesis_u = (thesis or "").upper()
    c = max(0.0, min(1.0, confidence))
    if "HAUS" in thesis_u:
        return c
    if "BAIS" in thesis_u:
        return 1.0 - c
    return 0.5


def agent_to_posterior(
    name: str,
    thesis: str,
    confidence: float,
    weight: float = 4.0,
) -> AgentPosterior:
    p = _thesis_to_p(thesis, confidence)
    return AgentPosterior(
        name=name,
        thesis=thesis,
        confidence=confidence,
        weight=weight,
        alpha=weight * p,
        beta=weight * (1 - p),
    )


# ---------------------------------------------------------------------------
# Extraction depuis scratchpad XML (compat agent_debat.py)
# ---------------------------------------------------------------------------

_STATUS_PAT = re.compile(
    r'<section\s+agent="([^"]+)">(.*?)</section>',
    re.DOTALL | re.IGNORECASE,
)
_THESIS_PAT = re.compile(r"<thesis>\s*([^<]+?)\s*</thesis>", re.IGNORECASE)
_CONF_PAT = re.compile(r"\[confiance:\s*([\d.]+)\]")
_TOUR_PAT = re.compile(r"\[Tour\s+(\d+)\]")


def extract_final_round_posteriors(
    scratchpad_xml: str,
    agent_weights: Optional[dict[str, float]] = None,
) -> list[AgentPosterior]:
    """
    Extrait, pour chaque agent, sa thèse + confiance du DERNIER tour.
    Retombe sur le dernier `[confiance: x]` trouvé dans sa section.
    """
    posteriors: list[AgentPosterior] = []
    for m in _STATUS_PAT.finditer(scratchpad_xml):
        agent = m.group(1)
        section = m.group(2)

        # Dernier tour mentionné
        tours = [(int(t.group(1)), t.end()) for t in _TOUR_PAT.finditer(section)]
        if tours:
            last_tour_end_pos = max(tours, key=lambda x: x[0])[1]
            trailing = section[last_tour_end_pos:]
        else:
            trailing = section

        # Confiance (du dernier tour si on a pu isoler, sinon dernière du string)
        conf_matches = _CONF_PAT.findall(trailing) or _CONF_PAT.findall(section)
        if not conf_matches:
            continue
        try:
            conf = float(conf_matches[-1])
        except ValueError:
            conf = 0.5

        # Thèse : inférée du nom d'agent à défaut d'un tag <thesis>
        thesis_match = _THESIS_PAT.search(section)
        if thesis_match:
            thesis = thesis_match.group(1).strip()
        else:
            thesis = agent  # "Haussier" / "Baissier" / "Neutre"

        weight = (agent_weights or {}).get(agent, 4.0)
        posteriors.append(agent_to_posterior(agent, thesis, conf, weight=weight))

    return posteriors


# ---------------------------------------------------------------------------
# Agrégation Bayésienne
# ---------------------------------------------------------------------------


@dataclass
class BayesianConsensus:
    p_mean: float  # E[p(Achat)]
    p_var_total: float  # variance totale du posterior
    p_var_epistemic: float  # variance entre agents (désaccord)
    p_var_aleatoric: float  # variance moyenne intra-agent
    ci95_lower: float
    ci95_upper: float
    signal: str  # Achat | Vente | Neutre
    signal_confidence: float  # |E[p] - 0.5| * 2
    kelly_scale: float  # [0, 1] — à multiplier au f_Kelly
    n_agents: int
    posteriors: list[AgentPosterior] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def beta_quantile(alpha: float, beta: float, q: float) -> float:
    """
    Approximation du quantile de Beta(α, β) via Cornish-Fisher + normal.
    Suffisant pour alpha,beta > 2. Alternative : scipy.stats.beta.ppf.
    """
    if alpha <= 0 or beta <= 0:
        return 0.5
    mu = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    sd = math.sqrt(var)
    # Inverse normal approximé (Beasley-Springer-Moro simplifié)
    if q <= 0 or q >= 1:
        return mu
    from math import erf
    from math import sqrt as msqrt

    # Bisection rapide de l'inverse normale standard
    z = _inv_norm(q)
    return max(0.0, min(1.0, mu + z * sd))


def _inv_norm(p: float) -> float:
    """Beasley-Springer-Moro approximation de Φ⁻¹(p)."""
    if p <= 0 or p >= 1:
        return 0.0
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def aggregate_posteriors(
    posteriors: list[AgentPosterior],
    kappa: float = 4.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    neutre_threshold: float = 0.10,
) -> BayesianConsensus:
    """
    Agrégation Beta-Binomial + calcul de variance épistémique + Kelly scale.

    Args:
        posteriors        : liste d'AgentPosterior
        kappa             : facteur de pénalité Kelly par unité de variance (sensibilité)
        prior_alpha/beta  : prior Beta (par défaut Beta(1,1) = uniforme)
        neutre_threshold  : |E[p] - 0.5| en dessous duquel on signale Neutre

    Retourne un BayesianConsensus prêt à l'emploi par le position sizer.
    """
    reasons = []

    if not posteriors:
        return BayesianConsensus(
            p_mean=0.5,
            p_var_total=0.25,
            p_var_epistemic=0.0,
            p_var_aleatoric=0.25,
            ci95_lower=0.0,
            ci95_upper=1.0,
            signal="Neutre",
            signal_confidence=0.0,
            kelly_scale=0.0,
            n_agents=0,
            reasons=["Aucun agent disponible"],
        )

    alpha_agg = prior_alpha + sum(p.alpha for p in posteriors)
    beta_agg = prior_beta + sum(p.beta for p in posteriors)

    p_mean = alpha_agg / (alpha_agg + beta_agg)
    p_var = alpha_agg * beta_agg / ((alpha_agg + beta_agg) ** 2 * (alpha_agg + beta_agg + 1))

    # Variance épistémique : variance des p_k individuels autour de leur moyenne pondérée
    weights = [p.weight for p in posteriors]
    W = sum(weights)
    p_means = [p.p_achat for p in posteriors]
    p_weighted_mean = sum(w * pm for w, pm in zip(weights, p_means)) / W if W > 0 else 0.5
    p_var_epi = sum(w * (pm - p_weighted_mean) ** 2 for w, pm in zip(weights, p_means)) / W if W > 0 else 0.0

    # Variance aléatoire : Var moyenne intra-agent sous Beta(conf·ν, (1-conf)·ν)
    # avec ν = weight. Var_k = conf·(1-conf)/(ν+1)
    p_var_ale = (
        sum(
            w * (p.alpha * p.beta) / ((p.alpha + p.beta) ** 2 * (p.alpha + p.beta + 1))
            for w, p in zip(weights, posteriors)
        )
        / W
        if W > 0
        else 0.0
    )

    p_var_total = p_var_epi + p_var_ale

    # CI 95%
    ci_lo = beta_quantile(alpha_agg, beta_agg, 0.025)
    ci_hi = beta_quantile(alpha_agg, beta_agg, 0.975)

    # Décision directionnelle
    delta = p_mean - 0.5
    if abs(delta) < neutre_threshold:
        signal = "Neutre"
        reasons.append(f"E[p]={p_mean:.3f} proche de 0.5 (|Δ|<{neutre_threshold})")
    elif delta > 0:
        signal = "Achat"
    else:
        signal = "Vente"
    signal_conf = round(abs(delta) * 2, 4)

    # Kelly scaling — plus la variance est grande, plus on sous-bette
    # kelly_scale ∈ [0, 1], décroissant avec p_var_total
    # Au limit : variance = 0 → scale = 2·|Δ| (i.e. pleine conviction)
    #            variance = 0.25 (uniforme) → scale ≈ 0
    kelly_scale = max(0.0, 2 * abs(delta)) / (1.0 + kappa * p_var_total)
    kelly_scale = round(min(1.0, kelly_scale), 4)

    if p_var_epi > 0.08:
        reasons.append(
            f"Désaccord épistémique fort (Var_epi={p_var_epi:.4f}) — "
            f"Kelly réduit par facteur {1 / (1 + kappa * p_var_total):.2f}x"
        )
    if p_var_ale > 0.08:
        reasons.append(
            f"Incertitude aléatoire forte (Var_ale={p_var_ale:.4f}) — les agents sont peu confiants individuellement"
        )

    return BayesianConsensus(
        p_mean=round(p_mean, 4),
        p_var_total=round(p_var_total, 6),
        p_var_epistemic=round(p_var_epi, 6),
        p_var_aleatoric=round(p_var_ale, 6),
        ci95_lower=round(ci_lo, 4),
        ci95_upper=round(ci_hi, 4),
        signal=signal,
        signal_confidence=signal_conf,
        kelly_scale=kelly_scale,
        n_agents=len(posteriors),
        posteriors=posteriors,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Point d'entrée haut niveau
# ---------------------------------------------------------------------------


def consensus_from_scratchpad(
    scratchpad_xml: str,
    agent_weights: Optional[dict[str, float]] = None,
    kappa: float = 4.0,
) -> BayesianConsensus:
    """Extrait + agrège en une étape pour usage direct depuis le pipeline."""
    posteriors = extract_final_round_posteriors(scratchpad_xml, agent_weights)
    return aggregate_posteriors(posteriors, kappa=kappa)


if __name__ == "__main__":
    # Petit test rapide
    scratchpad = """
    <scratchpad ticker="AAPL">
      <section agent="Haussier">
        <argument>[Tour 1] EPS beat [confiance: 0.7]</argument>
        <argument>[Tour 2] guidance raised [confiance: 0.85]</argument>
        <argument>[Tour 3] strong buy [confiance: 0.9]</argument>
      </section>
      <section agent="Baissier">
        <argument>[Tour 1] debt rising [confiance: 0.6]</argument>
        <argument>[Tour 2] weaker still [confiance: 0.5]</argument>
        <argument>[Tour 3] downgrade [confiance: 0.4]</argument>
      </section>
      <section agent="Neutre">
        <argument>[Tour 1] mixed [confiance: 0.5]</argument>
        <argument>[Tour 2] mixed [confiance: 0.5]</argument>
        <argument>[Tour 3] mixed [confiance: 0.55]</argument>
      </section>
    </scratchpad>
    """
    c = consensus_from_scratchpad(scratchpad)
    logger.info(f"Signal: {c.signal} | E[p]={c.p_mean} | Var={c.p_var_total:.4f}")
    logger.info(f"CI95: [{c.ci95_lower}, {c.ci95_upper}] | Kelly scale: {c.kelly_scale}")
    for r in c.reasons:
        logger.info(f"  - {r}")
