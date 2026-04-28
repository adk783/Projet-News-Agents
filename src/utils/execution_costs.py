"""
execution_costs.py — Modèle de friction d'exécution (mode simple + mode avancé)

DEUX MODES
----------
1) CHEAP MODE (défaut) — adapté petits volumes / retail
     C(Q) = commission_IB + half_spread · notional
   Pas de market impact (vous ne bougez pas le marché à 10k$ d'ordre).
   Pas de timing cost (latence négligeable vs. vol intra-day).
   Coût typique : 1-3 bps sur du large-cap liquide.

2) ADVANCED MODE (opt-in via use_almgren_chriss=True)
   Ajoute :
     + market_impact(Q, ADV, σ)  — Almgren-Chriss 2001
     + timing_cost(delay_ms)     — Kissell 2006
   Utile si on trade des millions de $ par ordre (>1% de l'ADV).

OBJECTIF
--------
Convertir un return "brut" en return "net of costs" dans le sizer Kelly,
et estimer le seuil break-even d'accuracy en dessous duquel la stratégie
est non rentable.

QUAND ACTIVER ADVANCED MODE ?
-----------------------------
Règle simple : si (quantity / ADV) > 0.5%, activez Almgren-Chriss.
Sinon, le modèle cheap est largement suffisant.

Références
----------
  Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio
      transactions." Journal of Risk, 3, 5-40.
  Kissell, R. (2014). "The Science of Algorithmic Trading and Portfolio
      Management." Academic Press.
  Bouchaud, J.-P. (2010). "Price impact." Encyclopedia of Quantitative Finance.
  Interactive Brokers Fixed Pricing : $0.005/share, min $1, max 1% notional.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import math
from dataclasses import dataclass

# Paramètres par défaut (marché US large-cap)
MIN_COMMISSION_USD = 1.0
COMMISSION_PER_SHARE = 0.005
COMMISSION_PCT_FALLBACK = 0.0005  # 5 bps si pas de commission par share
IMPACT_LAMBDA = 0.10  # Bouchaud meta-analysis, ~10bp pour Q/ADV = 1%
TIMING_ALPHA = 0.20  # empirical
SPREAD_FLOOR_BPS = 1.0  # 1 bp minimum
SPREAD_CEILING_BPS = 30.0  # 30 bps max pour small caps
ANNUALIZED_DAILY_VOL_DEFAULT = 0.018  # 1.8% daily vol (~28% annualized)
TRADING_HOURS_MS = 8 * 3600 * 1000


@dataclass
class ExecutionCostBreakdown:
    notional_usd: float
    quantity: float
    price: float
    commission_usd: float
    spread_cost_usd: float
    market_impact_usd: float
    timing_cost_usd: float
    total_cost_usd: float
    total_cost_bps: float  # coût total en basis points du notional

    def summary(self) -> str:
        return (
            f"Notional=${self.notional_usd:,.0f} | Total cost=${self.total_cost_usd:,.2f} "
            f"({self.total_cost_bps:.1f} bps) | "
            f"Comm=${self.commission_usd:.2f} Spread=${self.spread_cost_usd:.2f} "
            f"Impact=${self.market_impact_usd:.2f} Timing=${self.timing_cost_usd:.2f}"
        )


# ---------------------------------------------------------------------------
# Sous-modèles
# ---------------------------------------------------------------------------


def _commission(quantity: float, notional: float) -> float:
    """Interactive Brokers-like tiered pricing."""
    per_share_model = max(MIN_COMMISSION_USD, COMMISSION_PER_SHARE * abs(quantity))
    pct_model = abs(notional) * COMMISSION_PCT_FALLBACK
    # IB cap : commission ≤ 1% du notional (protection small trades)
    return min(per_share_model, max(pct_model, MIN_COMMISSION_USD))


def _half_spread_bps(adv_volume: float) -> float:
    """
    Approxime le spread half-spread en bps selon la liquidité.
    Marché très liquide (ADV=100M actions) → 0.5 bps
    Small cap (ADV=100K actions) → 20+ bps
    """
    if adv_volume <= 0:
        return SPREAD_CEILING_BPS
    # log-scale : log10(ADV)=8 (100M) → 0.5 bp ; log10(ADV)=5 (100K) → 20 bp
    log_vol = math.log10(max(adv_volume, 1e3))
    # interpolation linéaire
    bps = max(SPREAD_FLOOR_BPS, min(SPREAD_CEILING_BPS, 30.0 - 3.5 * log_vol))
    return bps


def _market_impact_pct(
    quantity: float,
    adv_volume: float,
    sigma_daily: float,
) -> float:
    """
    Almgren-Chriss 2001 : impact = λ · σ · sqrt(Q / ADV)
    Retourne un % du prix (ex: 0.002 = 20 bps).
    """
    if adv_volume <= 0 or quantity <= 0:
        return 0.0
    participation = abs(quantity) / adv_volume
    return IMPACT_LAMBDA * sigma_daily * math.sqrt(participation)


def _timing_cost_pct(delay_ms: float, sigma_daily: float) -> float:
    """
    Kissell 2006 : le délai entre signal et exécution coûte en moyenne
    α · σ · sqrt(T). T est exprimé en fraction de jour boursier.
    """
    if delay_ms <= 0:
        return 0.0
    fraction_of_day = delay_ms / TRADING_HOURS_MS
    return TIMING_ALPHA * sigma_daily * math.sqrt(fraction_of_day)


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------


def compute_execution_costs(
    quantity: float,
    price: float,
    adv_volume: float = 0.0,
    sigma_daily: float = ANNUALIZED_DAILY_VOL_DEFAULT,
    delay_ms: float = 0.0,
    use_almgren_chriss: bool = False,
) -> ExecutionCostBreakdown:
    """
    Coût d'exécution total pour un ordre market de taille `quantity` au prix `price`.

    MODE PAR DÉFAUT (cheap) — commissions IB + half-spread uniquement.
    Ignorer `adv_volume`, `sigma_daily`, `delay_ms` sauf si use_almgren_chriss=True.

    MODE AVANCÉ — passer use_almgren_chriss=True pour ajouter :
        + market_impact(Q, ADV, σ)   Almgren-Chriss 2001
        + timing_cost(delay_ms)      Kissell 2006
    À réserver aux ordres qui représentent > 0.5% de l'ADV.

    Args:
        quantity           : nb actions (positif = achat, la magnitude compte)
        price              : prix exécution
        adv_volume         : Average Daily Volume — utilisé seulement en mode avancé
        sigma_daily        : volatilité journalière — utilisé seulement en mode avancé
        delay_ms           : latence signal → exécution — utilisé seulement en mode avancé
        use_almgren_chriss : True pour gros ordres (>0.5% ADV). Default False.

    Returns:
        ExecutionCostBreakdown (détail par composante + total).
    """
    notional = abs(quantity) * price
    if notional == 0:
        return ExecutionCostBreakdown(
            notional_usd=0.0,
            quantity=0.0,
            price=price,
            commission_usd=0.0,
            spread_cost_usd=0.0,
            market_impact_usd=0.0,
            timing_cost_usd=0.0,
            total_cost_usd=0.0,
            total_cost_bps=0.0,
        )

    # Toujours : commission IB + half-spread
    comm = _commission(quantity, notional)
    spread_bps = _half_spread_bps(adv_volume) if adv_volume > 0 else SPREAD_FLOOR_BPS
    spread = notional * spread_bps / 10_000.0

    # Opt-in : Almgren-Chriss + Kissell timing
    impact = 0.0
    timing = 0.0
    if use_almgren_chriss:
        impact_pct = _market_impact_pct(quantity, adv_volume, sigma_daily)
        impact = notional * impact_pct
        timing_pct = _timing_cost_pct(delay_ms, sigma_daily)
        timing = notional * timing_pct

    total = comm + spread + impact + timing
    bps = (total / notional) * 10_000.0 if notional > 0 else 0.0

    return ExecutionCostBreakdown(
        notional_usd=round(notional, 2),
        quantity=quantity,
        price=price,
        commission_usd=round(comm, 4),
        spread_cost_usd=round(spread, 4),
        market_impact_usd=round(impact, 4),
        timing_cost_usd=round(timing, 4),
        total_cost_usd=round(total, 4),
        total_cost_bps=round(bps, 3),
    )


# ---------------------------------------------------------------------------
# Net-of-cost return — à brancher dans le Kelly
# ---------------------------------------------------------------------------


def net_expected_return(
    p_win: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    cost_bps: float,
) -> float:
    """
    Rendement espéré net de coûts, par trade.
      E[R_net] = p · TP - (1-p) · SL - cost_bps / 10000 · 2
                                         ^^ aller-retour (entry + exit)
    """
    c = cost_bps / 10_000.0 * 2.0
    return p_win * take_profit_pct - (1 - p_win) * abs(stop_loss_pct) - c


def break_even_accuracy(
    take_profit_pct: float,
    stop_loss_pct: float,
    cost_bps: float,
) -> float:
    """
    Résout E[R_net] = 0 pour p :
      p* = (|SL| + c) / (TP + |SL|)
    """
    c = cost_bps / 10_000.0 * 2.0
    sl = abs(stop_loss_pct)
    tp = abs(take_profit_pct)
    if (tp + sl) <= 0:
        return 1.0
    return round((sl + c) / (tp + sl), 4)


def kelly_net_of_costs(
    p_win: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    cost_bps: float,
) -> float:
    """
    Kelly (1956) appliqué sur des gains/pertes NETS des frais :
        b_net = (TP - c) / (|SL| + c)
        f*    = (p · b_net - (1-p)) / b_net
    On retourne max(0, f*).
    """
    c = cost_bps / 10_000.0  # one-way already compressed into b
    tp_net = max(1e-6, take_profit_pct - c)
    sl_net = abs(stop_loss_pct) + c
    b = tp_net / sl_net
    if b <= 0:
        return 0.0
    q = 1.0 - p_win
    f = (p_win * b - q) / b
    return round(max(0.0, f), 6)


if __name__ == "__main__":
    logger.info("=== CHEAP MODE (default, retail-size) ===")
    br = compute_execution_costs(
        quantity=1000,
        price=180.0,
        adv_volume=50_000_000,
    )
    print(br.summary())

    logger.info("\n=== ADVANCED MODE (Almgren-Chriss for big orders) ===")
    br_adv = compute_execution_costs(
        quantity=1000,
        price=180.0,
        adv_volume=50_000_000,
        sigma_daily=0.015,
        delay_ms=2000,
        use_almgren_chriss=True,
    )
    print(br_adv.summary())

    logger.info("\n=== Break-even analysis ===")
    ba = break_even_accuracy(0.03, 0.02, cost_bps=br.total_cost_bps)
    logger.info(f"Break-even accuracy (cheap mode) : {ba * 100:.2f}%")

    f_net = kelly_net_of_costs(0.57, 0.03, 0.02, br.total_cost_bps)
    f_gross = (0.57 * (0.03 / 0.02) - 0.43) / (0.03 / 0.02)
    logger.info(f"Kelly net = {f_net:.4f} vs Kelly gross = {f_gross:.4f}")
