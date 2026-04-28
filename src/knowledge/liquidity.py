"""
liquidity.py — Récupération dynamique de la liquidité (ADV + sigma)

OBJECTIF
--------
Remplacer les défauts hardcodés (`adv_volume = 5_000_000`, `sigma_daily = 0.018`)
par des valeurs réelles issues de yfinance. Ces deux variables conditionnent :
  1. Le modèle d'impact Almgren-Chriss (si activé)
  2. Le half-spread log-scale dans execution_costs
  3. Le filtre régime vol-adjusted dans yolo_classifier

SANS ADV DYNAMIQUE
------------------
Pour une small-cap (ADV = 500 K actions), utiliser 5 M = sous-estime l'impact
de 10×. Un ordre de 50 k actions = 10 % de l'ADV réelle = impact énorme,
là où avec le default 5 M = 1 %, l'algo pense qu'il peut y aller. Bug latent.

CACHE
-----
TTL 24 h : la liquidité intra-day ne bouge quasi pas, inutile de re-fetch
par article. Invalidation possible via `clear_cache()`.

USAGE
-----
    from src.knowledge.liquidity import get_liquidity_profile
    prof = get_liquidity_profile("AAPL")
    print(prof.adv_volume, prof.sigma_daily)
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paramètres par défaut (fallbacks conservateurs)
# ---------------------------------------------------------------------------
DEFAULT_ADV_VOLUME = 5_000_000  # actions/jour (large-cap typique)
DEFAULT_SIGMA_DAILY = 0.018  # ~28 % annualisé (sp500 moyen)
LIQUIDITY_CACHE_TTL_SEC = 24 * 60 * 60  # 24 h
_LIQUIDITY_CACHE: dict[str, tuple[float, LiquidityProfile]] = {}


@dataclass
class LiquidityProfile:
    ticker: str
    adv_volume: float  # Average Daily Volume (shares)
    sigma_daily: float  # Daily log-return stdev (fraction)
    source: str  # "yfinance" | "default" | "cache"
    fetched_at: str


def get_liquidity_profile(
    ticker: str,
    force_refresh: bool = False,
) -> LiquidityProfile:
    """
    Retourne un LiquidityProfile pour `ticker`. yfinance si possible, sinon
    fallback aux defaults.

    Args:
        ticker        : symbole (ex: "AAPL")
        force_refresh : bypass le cache TTL

    Returns:
        LiquidityProfile — best-effort, ne lève jamais.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return LiquidityProfile(
            ticker="",
            adv_volume=DEFAULT_ADV_VOLUME,
            sigma_daily=DEFAULT_SIGMA_DAILY,
            source="default",
            fetched_at="N/A",
        )

    # Cache lookup
    now = time.time()
    if not force_refresh and ticker in _LIQUIDITY_CACHE:
        expires_at, cached = _LIQUIDITY_CACHE[ticker]
        if now < expires_at:
            logger.debug("[Liquidity] cache HIT %s (TTL %.0fs)", ticker, expires_at - now)
            return cached

    # Fetch yfinance
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)

        # 1) ADV — info['averageVolume'] (10j moyen) ou recalcul depuis history
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        adv = info.get("averageVolume") or info.get("averageDailyVolume10Day")

        # 2) Sigma daily — calcul depuis les 3 derniers mois de returns
        hist = tk.history(period="3mo", auto_adjust=True)
        sigma = None
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) > 20:
                # log-returns journaliers
                import numpy as np

                log_ret = np.log(closes / closes.shift(1)).dropna()
                if len(log_ret) > 10:
                    sigma = float(log_ret.std())

            # Fallback ADV si info.get échoue
            if adv is None and "Volume" in hist.columns:
                vols = hist["Volume"].dropna()
                if len(vols) > 5:
                    adv = float(vols.mean())

        adv = float(adv) if adv else DEFAULT_ADV_VOLUME
        sigma = float(sigma) if sigma and math.isfinite(sigma) else DEFAULT_SIGMA_DAILY

        # Garde-fous : clamp anti-valeurs aberrantes
        adv = max(10_000, min(adv, 5_000_000_000))
        sigma = max(0.001, min(sigma, 0.15))

        prof = LiquidityProfile(
            ticker=ticker,
            adv_volume=adv,
            sigma_daily=sigma,
            source="yfinance",
            fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        )
        logger.info("[Liquidity] %s -> ADV=%s, sigma_daily=%.4f", ticker, f"{adv:,.0f}", sigma)

    except Exception as exc:
        logger.warning("[Liquidity] yfinance échec pour %s : %s — fallback défaut", ticker, exc)
        prof = LiquidityProfile(
            ticker=ticker,
            adv_volume=DEFAULT_ADV_VOLUME,
            sigma_daily=DEFAULT_SIGMA_DAILY,
            source="default",
            fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        )

    _LIQUIDITY_CACHE[ticker] = (now + LIQUIDITY_CACHE_TTL_SEC, prof)
    return prof


def clear_cache() -> None:
    """Flush le cache en mémoire (utile pour tests)."""
    _LIQUIDITY_CACHE.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for t in ["AAPL", "MSFT", "GME", "XXXZZZZ"]:
        prof = get_liquidity_profile(t)
        logger.info(f"{prof.ticker:8s}  ADV={prof.adv_volume:>15,.0f}  sigma={prof.sigma_daily:.4f}  ({prof.source})")
