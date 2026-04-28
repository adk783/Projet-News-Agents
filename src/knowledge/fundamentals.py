"""
fundamentals.py — Données Fondamentales Structurées

Injecte dans le débat les données fondamentales publiques les plus pertinentes,
permettant aux agents de contextualiser la news dans son environnement
financier réel (valorisation, attentes, catalysts à venir).

Références scientifiques :
  [1] Fama, E.F. (1970). "Efficient Capital Markets: A Review of Theory and
      Empirical Work." Journal of Finance, 25(2), 383-417.
      → La valeur ajoutée vient de la VITESSE et de la COMPLÉTUDE du traitement
        de l'information publique — même EMH suppose que les prix reflètent toute
        l'information disponible, un agent qui l'agrège plus vite a un avantage.

  [2] Ball, R. & Brown, P. (1968). "An Empirical Evaluation of Accounting
      Income Numbers." Journal of Accounting Research, 6(2), 159-178.
      → PEAD (Post-Earnings Announcement Drift) : les surprises EPS génèrent
        des drifts de cours de +2% à +4% sur 60 jours après l'annonce.
        Connaître le consensus EPS avant la news est crucial.

  [3] Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling
      Losers: Implications for Stock Market Efficiency."
      Journal of Finance, 48(1), 65-91.
      → Momentum : les recommandations d'analystes capturent une partie du
        momentum informationnel à 3-12 mois.

  [4] Graham, B. & Dodd, D. (1934). Security Analysis. McGraw-Hill.
      → Le P/E ratio comme mesure fondamentale de valorisation relative.
        P/E forward vs trailing : le forward P/E est plus prédictif (Shiller 2000).

  [5] Lim, T. (2001). "Rationality and Analysts' Forecast Bias."
      Journal of Finance, 56(1), 369-385.
      → Le consensus analystes présente un biais haussier systématique de ~5%.
        Ce biais est corrigé en normalisant le consensus vs historique du secteur.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass des fondamentaux
# ---------------------------------------------------------------------------


@dataclass
class FundamentalsData:
    """
    Snapshot des données fondamentales d'un titre à un instant t.
    Toutes les valeurs sont None si non disponibles (best-effort).
    """

    ticker: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    fetched_at: str = ""

    # --- Valorisation (Graham & Dodd 1934, Shiller 2000) ---
    pe_trailing: Optional[float] = None  # P/E sur les 12 derniers mois
    pe_forward: Optional[float] = None  # P/E sur les 12 prochains mois (Shiller)
    peg_ratio: Optional[float] = None  # Price/Earnings/Growth (Lynch)
    price_to_book: Optional[float] = None  # P/Book Value
    market_cap: Optional[float] = None  # Capitalisation boursière
    enterprise_value: Optional[float] = None  # EV (mieux que market cap)
    ev_to_ebitda: Optional[float] = None  # EV/EBITDA (preferred by practitioners)

    # --- Revenus & Marges ---
    revenue_ttm: Optional[float] = None  # Revenus TTM
    revenue_growth_yoy: Optional[float] = None  # Croissance YoY
    gross_margin: Optional[float] = None  # Marge brute
    operating_margin: Optional[float] = None  # Marge opérationnelle
    net_margin: Optional[float] = None  # Marge nette
    return_on_equity: Optional[float] = None  # ROE

    # --- EPS & Earnings (Ball & Brown 1968 — PEAD) ---
    eps_trailing: Optional[float] = None  # EPS TTM
    eps_forward: Optional[float] = None  # EPS estimé prochain exercice
    eps_surprise_last: Optional[float] = None  # Surprise EPS dernier trimestre (%)

    # --- Earnings Calendar ---
    next_earnings_date: Optional[str] = None  # ISO 8601
    days_to_earnings: Optional[int] = None  # Jours avant les prochains résultats

    # --- Analyst Consensus (Jegadeesh & Titman 1993, Lim 2001) ---
    analyst_consensus: Optional[str] = None  # "Strong Buy" / "Buy" / "Hold" / "Sell"
    analyst_mean_target: Optional[float] = None  # Prix cible moyen
    analyst_upside: Optional[float] = None  # Upside vs prix actuel (%)
    n_analysts: Optional[int] = None  # Nombre d'analystes couvrant la valeur

    # --- Bilan & Liquidité ---
    debt_to_equity: Optional[float] = None  # Levier financier
    current_ratio: Optional[float] = None  # Liquidité court terme
    free_cash_flow: Optional[float] = None  # FCF TTM

    # --- Dividende ---
    dividend_yield: Optional[float] = None  # Rendement dividende

    # Erreur éventuelle
    error: Optional[str] = None
    data_quality: str = "partial"  # "full" / "partial" / "unavailable"


# ---------------------------------------------------------------------------
# Récupération des fondamentaux
# ---------------------------------------------------------------------------


def get_fundamentals(ticker: str, current_price: float = 0.0) -> FundamentalsData:
    """
    Récupère les données fondamentales d'un ticker via yfinance.

    Toutes les données sont publiques et gratuites. Aucune clé API requise.
    Ref: Fama (1970) — intégrer rapidement l'information publique est la source
    de l'avantage informationnel même dans les marchés semi-forts.

    Args:
        ticker        : symbole boursier
        current_price : prix actuel (pour calcul upside)

    Returns:
        FundamentalsData (best-effort, valeurs None si indisponibles)
    """
    try:
        import yfinance as yf
    except ImportError:
        return FundamentalsData(ticker=ticker, error="yfinance non installé", data_quality="unavailable")

    fund = FundamentalsData(
        ticker=ticker,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        fund.company_name = info.get("shortName") or info.get("longName", ticker)
        fund.sector = info.get("sector", "")
        fund.industry = info.get("industry", "")

        # --- Valorisation ---
        fund.pe_trailing = _safe_float(info.get("trailingPE"))
        fund.pe_forward = _safe_float(info.get("forwardPE"))
        fund.peg_ratio = _safe_float(info.get("pegRatio"))
        fund.price_to_book = _safe_float(info.get("priceToBook"))
        fund.market_cap = _safe_float(info.get("marketCap"))
        fund.enterprise_value = _safe_float(info.get("enterpriseValue"))
        fund.ev_to_ebitda = _safe_float(info.get("enterpriseToEbitda"))

        # --- Revenus & Marges ---
        fund.revenue_ttm = _safe_float(info.get("totalRevenue"))
        fund.revenue_growth_yoy = _safe_float(info.get("revenueGrowth"))
        fund.gross_margin = _safe_float(info.get("grossMargins"))
        fund.operating_margin = _safe_float(info.get("operatingMargins"))
        fund.net_margin = _safe_float(info.get("profitMargins"))
        fund.return_on_equity = _safe_float(info.get("returnOnEquity"))

        # --- EPS ---
        fund.eps_trailing = _safe_float(info.get("trailingEps"))
        fund.eps_forward = _safe_float(info.get("forwardEps"))

        # Surprise EPS du dernier trimestre
        try:
            earnings_hist = stock.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                last = earnings_hist.iloc[-1]
                eps_est = _safe_float(last.get("epsEstimate"))
                eps_act = _safe_float(last.get("epsActual"))
                if eps_est and eps_act and eps_est != 0:
                    fund.eps_surprise_last = round((eps_act - eps_est) / abs(eps_est) * 100, 2)
        except Exception:
            pass

        # --- Earnings Calendar (Ball & Brown 1968 — PEAD) ---
        try:
            cal = stock.calendar
            if cal is not None:
                # yfinance renvoie un dict avec "Earnings Date" comme liste ou datetime
                earnings_dates = cal.get("Earnings Date") or cal.get("earningsDate")
                if earnings_dates:
                    if hasattr(earnings_dates, "__iter__") and not isinstance(earnings_dates, str):
                        next_ed = next(iter(earnings_dates), None)
                    else:
                        next_ed = earnings_dates

                    if next_ed is not None:
                        if hasattr(next_ed, "isoformat"):
                            fund.next_earnings_date = next_ed.isoformat()
                            fund.days_to_earnings = max(0, (next_ed.date() - datetime.now().date()).days)
                        else:
                            fund.next_earnings_date = str(next_ed)
        except Exception:
            pass

        # --- Analyst Consensus (Jegadeesh & Titman 1993) ---
        fund.analyst_mean_target = _safe_float(info.get("targetMeanPrice"))
        fund.n_analysts = info.get("numberOfAnalystOpinions")

        # Calcul upside vs prix actuel
        price_ref = (
            current_price or _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))
        )
        if fund.analyst_mean_target and price_ref and price_ref > 0:
            fund.analyst_upside = round((fund.analyst_mean_target - price_ref) / price_ref * 100, 1)

        # Consensus texte depuis recommandationKey
        rec_key = info.get("recommendationKey", "")
        fund.analyst_consensus = _map_recommendation(rec_key)

        # --- Bilan ---
        fund.debt_to_equity = _safe_float(info.get("debtToEquity"))
        fund.current_ratio = _safe_float(info.get("currentRatio"))
        fund.free_cash_flow = _safe_float(info.get("freeCashflow"))
        fund.dividend_yield = _safe_float(info.get("dividendYield"))

        # Évaluation qualité des données
        n_fields_filled = sum(
            1
            for v in [
                fund.pe_forward,
                fund.eps_forward,
                fund.analyst_consensus,
                fund.next_earnings_date,
                fund.gross_margin,
                fund.revenue_growth_yoy,
            ]
            if v is not None
        )
        fund.data_quality = "full" if n_fields_filled >= 5 else "partial"

        logger.info(
            "[Fundamentals] %s | P/E fwd: %.1f | Consensus: %s | Earnings dans: %s | Qualité: %s",
            ticker,
            fund.pe_forward or 0,
            fund.analyst_consensus or "N/A",
            f"{fund.days_to_earnings}j" if fund.days_to_earnings is not None else "N/A",
            fund.data_quality,
        )

    except Exception as e:
        fund.error = str(e)
        fund.data_quality = "unavailable"
        logger.warning("[Fundamentals] Erreur pour %s : %s", ticker, e)

    return fund


# ---------------------------------------------------------------------------
# Formatage pour injection dans le prompt
# ---------------------------------------------------------------------------


def format_fundamentals_for_prompt(fund: FundamentalsData) -> str:
    """
    Formate les fondamentaux en bloc texte structuré pour injection dans le débat.

    Le format est conçu pour maximiser la densité informationnelle par token
    (Lewis et al. 2020 RAG : le contexte structuré > contexte narratif).
    """
    if fund.data_quality == "unavailable":
        return f"[Fondamentaux {fund.ticker}] Données indisponibles. {fund.error or ''}"

    lines = [
        f"=== DONNÉES FONDAMENTALES — {fund.ticker} ({fund.company_name}) ===",
        f"Secteur: {fund.sector or 'N/A'} | Industrie: {fund.industry or 'N/A'}",
        "",
    ]

    # Valorisation
    val_parts = []
    if fund.pe_trailing is not None:
        val_parts.append(f"P/E trailing: {fund.pe_trailing:.1f}x")
    if fund.pe_forward is not None:
        val_parts.append(f"P/E forward: {fund.pe_forward:.1f}x")
    if fund.peg_ratio is not None:
        val_parts.append(f"PEG: {fund.peg_ratio:.2f}")
    if fund.ev_to_ebitda is not None:
        val_parts.append(f"EV/EBITDA: {fund.ev_to_ebitda:.1f}x")
    if val_parts:
        lines.append("Valorisation : " + " | ".join(val_parts))

    # EPS & Earnings
    eps_parts = []
    if fund.eps_trailing is not None:
        eps_parts.append(f"EPS TTM: {fund.eps_trailing:.2f}$")
    if fund.eps_forward is not None:
        eps_parts.append(f"EPS fwd: {fund.eps_forward:.2f}$")
    if fund.eps_surprise_last is not None:
        sign = "+" if fund.eps_surprise_last >= 0 else ""
        eps_parts.append(f"Surprise Q-1: {sign}{fund.eps_surprise_last:.1f}%")
    if eps_parts:
        lines.append("EPS         : " + " | ".join(eps_parts))

    # Earnings Calendar — CRITIQUE pour contexte PEAD (Ball & Brown 1968)
    if fund.next_earnings_date:
        urgence = ""
        if fund.days_to_earnings is not None:
            if fund.days_to_earnings <= 7:
                urgence = " ⚠ IMMINENT"
            elif fund.days_to_earnings <= 30:
                urgence = " (proche)"
        lines.append(
            f"Earnings    : {fund.next_earnings_date}"
            f"{f' (dans {fund.days_to_earnings}j)' if fund.days_to_earnings is not None else ''}"
            f"{urgence}"
        )

    # Analyst Consensus (Jegadeesh & Titman 1993 — momentum informationnel)
    cons_parts = []
    if fund.analyst_consensus:
        cons_parts.append(f"Consensus: {fund.analyst_consensus}")
    if fund.analyst_mean_target is not None:
        cons_parts.append(f"Cible moy: {fund.analyst_mean_target:.2f}$")
    if fund.analyst_upside is not None:
        sign = "+" if fund.analyst_upside >= 0 else ""
        cons_parts.append(f"Upside: {sign}{fund.analyst_upside:.1f}%")
    if fund.n_analysts is not None:
        cons_parts.append(f"N analystes: {fund.n_analysts}")
    if cons_parts:
        lines.append("Analystes   : " + " | ".join(cons_parts))
    # Note sur le biais analyste (Lim 2001)
    if fund.analyst_upside and fund.analyst_upside > 20:
        lines.append(
            "  ℹ [Lim 2001] Biais haussier systématique des analystes (~5%). "
            "Upside élevé → interpréter avec précaution."
        )

    # Marges & Croissance
    margin_parts = []
    if fund.gross_margin is not None:
        margin_parts.append(f"Brute: {fund.gross_margin:.1%}")
    if fund.operating_margin is not None:
        margin_parts.append(f"Opér: {fund.operating_margin:.1%}")
    if fund.net_margin is not None:
        margin_parts.append(f"Nette: {fund.net_margin:.1%}")
    if fund.revenue_growth_yoy is not None:
        margin_parts.append(f"Croiss. rev: {fund.revenue_growth_yoy:.1%} YoY")
    if margin_parts:
        lines.append("Marges      : " + " | ".join(margin_parts))

    # Bilan
    bilan_parts = []
    if fund.debt_to_equity is not None:
        bilan_parts.append(f"D/E: {fund.debt_to_equity:.2f}")
    if fund.current_ratio is not None:
        bilan_parts.append(f"Liquidité: {fund.current_ratio:.2f}")
    if fund.free_cash_flow is not None:
        fcf_b = fund.free_cash_flow / 1e9
        bilan_parts.append(f"FCF: {fcf_b:.1f}Md$")
    if fund.dividend_yield is not None and fund.dividend_yield > 0:
        bilan_parts.append(f"Dividende: {fund.dividend_yield:.2%}")
    if bilan_parts:
        lines.append("Bilan       : " + " | ".join(bilan_parts))

    lines.append(f"[Qualité données: {fund.data_quality} | {fund.fetched_at[:10]}]")
    lines.append("=" * 52)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _safe_float(v) -> Optional[float]:
    """Convertit en float sans lever d'exception."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _map_recommendation(key: str) -> Optional[str]:
    """Convertit la clé yfinance en texte lisible (Jegadeesh & Titman 1993)."""
    mapping = {
        "strong_buy": "Strong Buy ⭐⭐⭐⭐⭐",
        "buy": "Buy ⭐⭐⭐⭐",
        "hold": "Hold ⭐⭐⭐",
        "underperform": "Underperform ⭐⭐",
        "sell": "Sell ⭐",
        "strong_sell": "Strong Sell ⭐",
    }
    return mapping.get(key.lower(), key if key else None)
