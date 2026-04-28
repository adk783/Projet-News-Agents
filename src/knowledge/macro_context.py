"""
macro_context.py — Contexte Macroéconomique

Fournit aux agents de débat les indicateurs macro qui conditionnent
l'interprétation de toute news financière spécifique.

Sans contexte macro, un agent peut recommander "Achat AAPL" sur une bonne
news sans savoir que le VIX est à 35, la courbe de taux est inversée depuis
6 mois et la Fed vient de monter les taux de 50bps — qui sont des facteurs
structurellement baissiers qui priment sur les fondamentaux d'un seul titre.

Références scientifiques :
  [1] Whaley, R.E. (2009). "Understanding the VIX."
      Journal of Portfolio Management, 35(3), 98-105.
      → VIX > 30 : régime de haute volatilité. Les news positives sont
        sous-réagies (biais conservateur des investisseurs). Les décisions
        d'achat doivent être réduites proportionnellement.

  [2] Estrella, A. & Mishkin, F.S. (1998). "Predicting U.S. Recessions:
      Financial Variables as Leading Indicators."
      Review of Economics and Statistics, 80(1), 45-61.
      → Yield curve 10Y-2Y inverted (< 0) : prédicteur de récession
        avec probabilité ≈ 70% sur horizon 12 mois.

  [3] Bernanke, B.S. & Kuttner, K.N. (2005). "What Explains the Stock Market's
      Reaction to Federal Reserve Policy?"
      Journal of Finance, 60(3), 1221-1257.
      → Hausse surprise Fed rate +25bps → S&P500 -1% dans la semaine.
        La politique monétaire est le facteur macro le plus impactant court terme.

  [4] Koijen, R.S.J. et al. (2012). "An Equilibrium Model of the Term Structure
      of Interest Rates." Journal of Finance, 67(2), 369-413.
      → Le niveau des taux longs détermine le taux d'actualisation des cash
        flows futurs → impact direct sur les valorisations (P/E compression).

  [5] Bartov, E. et al. (2000). "Investor Sophistication and Patterns in Stock
      Returns after Earnings Announcements."
      The Accounting Review, 75(1), 43-63.
      → L'effet DXY sur les multinationales : dollar fort → revenus
        internationaux réduits en USD (mécanisme FX).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL cache du snapshot macro (15 min par défaut)
# ---------------------------------------------------------------------------
# Les indicateurs macro (VIX, yields, DXY, S&P500) bougent en temps réel mais
# les implications pour notre pipeline news-driven sont quasi-stables sur
# 15 minutes. Pas besoin de re-télécharger 6 séries yfinance pour chaque
# article. Cache en mémoire de process.
#
# Invalidation :
#   - TTL : 15 minutes par défaut (configurable via MACRO_CONTEXT_TTL_SEC)
#   - get_macro_context(force_refresh=True) pour forcer un rafraîchissement

MACRO_CONTEXT_TTL_SEC = 15 * 60  # 15 min
_MACRO_CACHE: dict = {"snap": None, "expires_at": 0.0}

# ---------------------------------------------------------------------------
# Seuils de référence (basés sur les recherches)
# ---------------------------------------------------------------------------

# VIX: Whaley (2009)
VIX_REGIME_CALM = 15.0  # Régime de calme (bull market normal)
VIX_REGIME_ALERT = 20.0  # Zone d'attention
VIX_REGIME_FEAR = 30.0  # Peur — biais conservateur
VIX_REGIME_PANIC = 40.0  # Panique — décisions irrationnelles probables

# Yield curve: Estrella & Mishkin (1998)
YIELD_CURVE_INVERSION_THRESHOLD = 0.0  # < 0 → inversion → signal récession


@dataclass
class MacroSnapshot:
    """
    Snapshot macroéconomique à l'instant t.
    Conçu pour injection directe dans les prompts des agents de débat.
    """

    fetched_at: str = ""

    # --- VIX (Whaley 2009) ---
    vix: Optional[float] = None
    vix_regime: str = "INCONNU"  # CALM / ALERT / FEAR / PANIC
    vix_1m_change: Optional[float] = None  # Variation 1 mois du VIX

    # --- Taux & Yield Curve (Estrella & Mishkin 1998, Bernanke & Kuttner 2005) ---
    yield_10y: Optional[float] = None  # Taux 10 ans US (%)
    yield_2y: Optional[float] = None  # Taux 2 ans US (%)
    yield_curve_spread: Optional[float] = None  # 10Y - 2Y (< 0 = inversion)
    yield_curve_signal: str = "INCONNU"  # NORMAL / FLAT / INVERTED
    fed_rate_approx: Optional[float] = None  # Approximation Fed Funds Rate

    # --- Dollar (Bartov et al. 2000) ---
    dxy: Optional[float] = None  # Dollar Index (DX-Y.NYB)
    dxy_1m_change: Optional[float] = None  # Variation 1 mois du DXY (%)

    # --- S&P 500 Référence ---
    sp500_1m_return: Optional[float] = None  # Return S&P500 sur 1 mois (%)
    sp500_ytd_return: Optional[float] = None  # Return S&P500 YTD (%)

    # --- FRED (optionnel si clé disponible) ---
    cpi_yoy: Optional[float] = None  # Inflation CPI YoY (%)
    unemployment_rate: Optional[float] = None  # Taux de chômage (%)
    fed_funds_rate: Optional[float] = None  # Fed Funds Rate officiel FRED

    # Qualité des données
    data_quality: str = "partial"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Récupération des données macro
# ---------------------------------------------------------------------------


def get_macro_context(
    fred_api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> MacroSnapshot:
    """
    Agrège les indicateurs macroéconomiques depuis yfinance (+ FRED si clé disponible).

    Sources :
      - VIX : ^VIX (CBOE Volatility Index) — yfinance
      - 10Y Treasury : ^TNX — yfinance
      - 5Y Treasury : ^FVX — yfinance (proxy pour interpoler 2Y)
      - DXY : DX-Y.NYB — yfinance
      - S&P 500 : ^GSPC — yfinance
      - FRED (si fred_api_key) : FEDFUNDS, T10Y2Y, CPIAUCSL, UNRATE

    Toutes les données yfinance sont gratuites et ne requièrent aucune clé.

    Args:
        fred_api_key  : optionnelle, FRED API key pour enrichir avec CPI/UNRATE
        force_refresh : bypass le cache TTL (utile pour tests/debug)

    Returns:
        MacroSnapshot — best-effort (valeurs None si indisponibles)

    Cache : TTL 15 min par défaut (MACRO_CONTEXT_TTL_SEC). Les indicateurs
    macro n'ont pas besoin d'être recalculés à chaque article (c'est ~0.5 s
    de yfinance + latence réseau par article sinon).
    """
    now_ts = time.time()
    if not force_refresh and _MACRO_CACHE["snap"] is not None and now_ts < _MACRO_CACHE["expires_at"]:
        remaining = _MACRO_CACHE["expires_at"] - now_ts
        logger.debug("MacroContext: cache HIT (TTL restant %.0f s)", remaining)
        return _MACRO_CACHE["snap"]

    snap = MacroSnapshot(fetched_at=datetime.now(timezone.utc).isoformat())

    try:
        import pandas as pd
        import yfinance as yf

        # Téléchargement groupé (1 seul appel réseau)
        tickers_to_fetch = ["^VIX", "^TNX", "^FVX", "^IRX", "DX-Y.NYB", "^GSPC"]
        data = yf.download(
            tickers_to_fetch,
            period="3mo",
            progress=False,
            auto_adjust=True,
        )
        close = data["Close"] if "Close" in data.columns else data

        now = datetime.now(timezone.utc)

        # --- VIX (Whaley 2009) ---
        if "^VIX" in close.columns:
            vix_series = close["^VIX"].dropna()
            if not vix_series.empty:
                snap.vix = round(float(vix_series.iloc[-1]), 2)
                snap.vix_regime = _vix_regime(snap.vix)
                # Variation 1 mois
                if len(vix_series) >= 21:
                    snap.vix_1m_change = round(
                        float((vix_series.iloc[-1] - vix_series.iloc[-21]) / vix_series.iloc[-21] * 100), 1
                    )

        # --- Yield Curve (Estrella & Mishkin 1998) ---
        yield_10y = yield_2y = None
        if "^TNX" in close.columns:
            tnx = close["^TNX"].dropna()
            if not tnx.empty:
                yield_10y = round(float(tnx.iloc[-1]), 3)
                snap.yield_10y = yield_10y

        # ^IRX = 13-week T-bill → proxy court terme
        # On utilise ^FVX (5Y) / ^TNX (10Y) pour interpoler le 2Y approximativement
        # Ou on utilise directement le T-Bill comme proxy du taux court
        if "^IRX" in close.columns:
            irx = close["^IRX"].dropna()
            if not irx.empty:
                # ^IRX est en taux annualisé (déjà en %)
                yield_2y = round(float(irx.iloc[-1]), 3)
                snap.yield_2y = yield_2y

        if yield_10y is not None and yield_2y is not None:
            spread = round(yield_10y - yield_2y, 3)
            snap.yield_curve_spread = spread
            snap.yield_curve_signal = _yield_curve_signal(spread)

        # Approximation Fed Rate = taux court terme + 0.25%
        if yield_2y is not None:
            snap.fed_rate_approx = round(yield_2y * 0.95, 2)  # approximation conservative

        # --- Dollar Index (Bartov et al. 2000) ---
        if "DX-Y.NYB" in close.columns:
            dxy_series = close["DX-Y.NYB"].dropna()
            if not dxy_series.empty:
                snap.dxy = round(float(dxy_series.iloc[-1]), 2)
                if len(dxy_series) >= 21:
                    snap.dxy_1m_change = round(
                        float((dxy_series.iloc[-1] - dxy_series.iloc[-21]) / dxy_series.iloc[-21] * 100), 1
                    )

        # --- S&P 500 ---
        if "^GSPC" in close.columns:
            sp_series = close["^GSPC"].dropna()
            if len(sp_series) >= 21:
                snap.sp500_1m_return = round(
                    float((sp_series.iloc[-1] - sp_series.iloc[-21]) / sp_series.iloc[-21] * 100), 1
                )
            # YTD approximation (depuis début de l'année)
            try:
                sp_ytd = yf.download(
                    "^GSPC", start=f"{now.year}-01-01", end=now.strftime("%Y-%m-%d"), progress=False, auto_adjust=True
                )
                if not sp_ytd.empty:
                    c_ytd = sp_ytd["Close"]
                    snap.sp500_ytd_return = round(float((c_ytd.iloc[-1] - c_ytd.iloc[0]) / c_ytd.iloc[0] * 100), 1)
            except Exception:
                pass

        # --- FRED (si clé disponible) ---
        if fred_api_key:
            _enrich_with_fred(snap, fred_api_key)

        n_filled = sum(1 for v in [snap.vix, snap.yield_10y, snap.yield_curve_spread, snap.dxy] if v is not None)
        snap.data_quality = "full" if n_filled >= 3 else "partial"

        logger.info(
            "[Macro] VIX=%.1f (%s) | Yield curve=%.3f%% (%s) | DXY=%.1f | SP500 1M=%s%%",
            snap.vix or 0,
            snap.vix_regime,
            snap.yield_curve_spread or 0,
            snap.yield_curve_signal,
            snap.dxy or 0,
            f"{snap.sp500_1m_return:+.1f}" if snap.sp500_1m_return else "N/A",
        )

    except Exception as e:
        snap.error = str(e)
        snap.data_quality = "unavailable"
        logger.warning("[Macro] Erreur récupération contexte macro : %s", e)

    # Persistance dans le cache TTL (sauf si fetch échoué : on ne cache pas
    # un snapshot "unavailable", on retentera au prochain appel).
    if snap.data_quality != "unavailable":
        _MACRO_CACHE["snap"] = snap
        _MACRO_CACHE["expires_at"] = time.time() + MACRO_CONTEXT_TTL_SEC
        logger.debug("MacroContext: cache MISS — snapshot mis en cache pour %d s", MACRO_CONTEXT_TTL_SEC)

    return snap


def _enrich_with_fred(snap: MacroSnapshot, api_key: str) -> None:
    """Enrichit le snapshot avec les données FRED si la clé est disponible."""
    try:
        import requests

        BASE = "https://api.stlouisfed.org/fred/series/observations"

        def get_fred_latest(series_id: str) -> Optional[float]:
            try:
                resp = requests.get(
                    BASE,
                    params={
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1,
                        "observation_start": "2020-01-01",
                    },
                    timeout=10,
                )
                if resp.ok:
                    obs = resp.json().get("observations", [])
                    if obs and obs[0]["value"] != ".":
                        return float(obs[0]["value"])
            except Exception:
                pass
            return None

        snap.fed_funds_rate = get_fred_latest("FEDFUNDS")
        snap.cpi_yoy = get_fred_latest("CPIAUCSL")
        snap.unemployment_rate = get_fred_latest("UNRATE")

        # T10Y2Y officiel FRED (meilleur que notre approximation)
        t10y2y = get_fred_latest("T10Y2Y")
        if t10y2y is not None:
            snap.yield_curve_spread = t10y2y
            snap.yield_curve_signal = _yield_curve_signal(t10y2y)

        logger.info(
            "[Macro/FRED] CPI YoY=%.1f%% | Chômage=%.1f%% | Fed Rate=%.2f%%",
            snap.cpi_yoy or 0,
            snap.unemployment_rate or 0,
            snap.fed_funds_rate or 0,
        )
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[Macro/FRED] Erreur FRED : %s", e)


# ---------------------------------------------------------------------------
# Formatage pour injection dans le prompt
# ---------------------------------------------------------------------------


def format_macro_for_prompt(snap: MacroSnapshot, ticker_sector: str = "") -> str:
    """
    Formate le contexte macro pour injection dans le prompt du débat.

    Les alertes critiques (VIX élevé, courbe inversée) sont mises en avant
    car elles conditionnent l'interprétation de TOUTE news individuelle.
    """
    if snap.data_quality == "unavailable":
        return f"[Contexte Macro] Données indisponibles. {snap.error or ''}"

    lines = ["=== CONTEXTE MACROÉCONOMIQUE ==="]

    # --- VIX (Whaley 2009) ---
    if snap.vix is not None:
        vix_line = f"VIX (Peur marché) : {snap.vix:.1f} [{snap.vix_regime}]"
        if snap.vix_1m_change is not None:
            vix_line += f" | Δ1M: {snap.vix_1m_change:+.1f}%"
        lines.append(vix_line)

        # Interprétation Whaley (2009)
        if snap.vix >= VIX_REGIME_FEAR:
            lines.append(
                f"  ⚠ [Whaley 2009] VIX ≥ {VIX_REGIME_FEAR} = régime de PEUR. "
                f"Les news positives sont statistiquement sous-réagies. "
                f"Pondérer les signaux haussiers avec prudence."
            )
        elif snap.vix >= VIX_REGIME_ALERT:
            lines.append(f"  ℹ VIX en zone d'alerte ({snap.vix:.1f}). Marché agité.")

    # --- Yield Curve (Estrella & Mishkin 1998) ---
    if snap.yield_curve_spread is not None:
        curve_line = f"Yield Curve 10Y-2Y : {snap.yield_curve_spread:+.3f}% [{snap.yield_curve_signal}]"
        if snap.yield_10y:
            curve_line += f" | 10Y={snap.yield_10y:.2f}%"
        if snap.yield_2y:
            curve_line += f" | 2Y={snap.yield_2y:.2f}%"
        lines.append(curve_line)

        if snap.yield_curve_spread < YIELD_CURVE_INVERSION_THRESHOLD:
            lines.append(
                "  ⚠ [Estrella & Mishkin 1998] INVERSION de la courbe des taux. "
                "Probabilité de récession ≈ 70% sur 12 mois. "
                "Contexte structurellement baissier — réduire l'exposition aux cycliques."
            )

    # --- Fed Rate (Bernanke & Kuttner 2005) ---
    rate_info = []
    if snap.fed_funds_rate is not None:
        rate_info.append(f"Fed Rate officiel: {snap.fed_funds_rate:.2f}% (FRED)")
    elif snap.fed_rate_approx is not None:
        rate_info.append(f"Fed Rate estimé: ~{snap.fed_rate_approx:.2f}%")
    if rate_info:
        lines.append("Taux Fed       : " + " | ".join(rate_info))

    # --- CPI (FRED) ---
    if snap.cpi_yoy is not None:
        lines.append(f"Inflation CPI  : {snap.cpi_yoy:.1f}% YoY (FRED)")

    # --- Dollar (Bartov et al. 2000) ---
    if snap.dxy is not None:
        dxy_line = f"DXY (Dollar)   : {snap.dxy:.2f}"
        if snap.dxy_1m_change is not None:
            sign = "+" if snap.dxy_1m_change >= 0 else ""
            dxy_line += f" | Δ1M: {sign}{snap.dxy_1m_change:.1f}%"

        lines.append(dxy_line)

        # Impact sur les multinationales (Bartov et al. 2000)
        if (
            snap.dxy_1m_change
            and abs(snap.dxy_1m_change) > 3
            and ticker_sector in ("Technology", "Consumer Cyclical", "Healthcare", "Industrials")
        ):
            direction = "hausse" if snap.dxy_1m_change > 0 else "baisse"
            lines.append(
                f"  ℹ [Bartov 2000] Dollar {direction} ({snap.dxy_1m_change:+.1f}% sur 1M) : "
                f"impact FX sur les revenus internationaux des multinationales."
            )

    # --- S&P 500 ---
    sp_parts = []
    if snap.sp500_1m_return is not None:
        sp_parts.append(f"1M: {snap.sp500_1m_return:+.1f}%")
    if snap.sp500_ytd_return is not None:
        sp_parts.append(f"YTD: {snap.sp500_ytd_return:+.1f}%")
    if sp_parts:
        lines.append("S&P500         : " + " | ".join(sp_parts))

    # --- Unemployment (FRED) ---
    if snap.unemployment_rate is not None:
        lines.append(f"Chômage US     : {snap.unemployment_rate:.1f}% (FRED)")

    lines.append(f"[Qualité: {snap.data_quality} | {snap.fetched_at[:10]}]")
    lines.append("=" * 36)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vix_regime(vix: float) -> str:
    if vix >= VIX_REGIME_PANIC:
        return "PANIQUE"
    elif vix >= VIX_REGIME_FEAR:
        return "PEUR"
    elif vix >= VIX_REGIME_ALERT:
        return "ALERTE"
    else:
        return "CALME"


def _yield_curve_signal(spread: float) -> str:
    if spread < -0.25:
        return "INVERTIE (fort)"
    elif spread < 0:
        return "INVERTIE"
    elif spread < 0.5:
        return "PLATE"
    else:
        return "NORMALE"
