"""
portfolio_constraints.py — Contraintes portefeuille (sectorielle + corrélation)

OBJECTIF
--------
Empêcher les concentrations de risque :
  - sectorielle (5 positions Tech qui chutent ensemble),
  - cross-sectionnelle (AAPL + MSFT = ρ~0.85 = le même trade dupliqué).

RÈGLES IMPLÉMENTÉES
-------------------
  1. `check_sector_concentration()` : refuse si l'exposition sectorielle
     projetée dépasse MAX_SECTOR_EXPOSURE_PCT (défaut 30 %). Référence
     DeMiguel et al. 2009 — au-delà de 40 %, la diversification
     n'est plus efficace.
  2. `check_pairwise_correlation()` : refuse si la nouvelle position a
     une corrélation Pearson > MAX_PAIRWISE_CORRELATION (défaut 0.80) avec
     au moins une position déjà ouverte, calculée sur les log-returns
     quotidiens des CORRELATION_LOOKBACK_DAYS derniers jours (yfinance).
     Référence Ang-Chen 2002, Rankin-Jegadeesh 1993.

RÉFÉRENCES
----------
  Markowitz (1952). Portfolio Selection. JoF.
  Rankin & Jegadeesh (1993). "Returns to trading strategies based on
      price ratios." JoF 48(5), 1681-1713.
  Ang & Chen (2002). "Asymmetric correlations of equity portfolios."
      J. Financial Economics 63(3), 443-494.
  DeMiguel, Garlappi, Uppal (2009). "Optimal versus Naive Diversification."
      Review of Financial Studies 22(5), 1915-1953.

USAGE
-----
    from src.strategy.portfolio_constraints import (
        check_sector_concentration, check_pairwise_correlation,
    )

    sector = check_sector_concentration(portfolio, "Technology", 8000.0)
    corr   = check_pairwise_correlation(portfolio, "AAPL")
    if not (sector.allowed and corr.allowed):
        # refuser ouverture + logger
        ...
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from src.config import (
    CORRELATION_LOOKBACK_DAYS,
    MAX_PAIRWISE_CORRELATION,
    MAX_SECTOR_EXPOSURE_PCT,
)

logger = logging.getLogger(__name__)


@dataclass
class ConstraintResult:
    allowed: bool
    reason: str
    sector: str
    current_pct: float  # exposition secteur AVANT ouverture
    projected_pct: float  # exposition secteur APRÈS ouverture hypothétique
    cap_pct: float  # seuil MAX_SECTOR_EXPOSURE_PCT effectif

    def summary(self) -> str:
        verdict = "OK" if self.allowed else "REFUSED"
        return (
            f"[{verdict}] Secteur {self.sector} : "
            f"{self.current_pct:.1%} -> {self.projected_pct:.1%} "
            f"(cap {self.cap_pct:.0%}). {self.reason}"
        )


def check_sector_concentration(
    portfolio_state,
    secteur: str,
    montant_propose_euros: float,
    cap_pct: Optional[float] = None,
) -> ConstraintResult:
    """
    Vérifie qu'ouvrir une nouvelle position de `montant_propose_euros` dans
    `secteur` ne dépasse pas `cap_pct` (ou MAX_SECTOR_EXPOSURE_PCT par défaut).

    Args:
        portfolio_state       : PortfolioState courant
        secteur               : secteur de la position envisagée
        montant_propose_euros : taille nominale de l'ordre (valeur absolue)
        cap_pct               : override du cap (None => config default)

    Returns:
        ConstraintResult — consultable via .allowed et .summary().
    """
    cap = cap_pct if cap_pct is not None else MAX_SECTOR_EXPOSURE_PCT
    secteur_key = secteur or "Inconnu"

    # On récupère le capital total du portefeuille. PortfolioState expose
    # `valeur_totale()` (méthode qui somme cash + MtM des positions). On
    # tolère aussi un attribut `capital_total` pour compatibilité ascendante
    # (mock tests), et on fallback sur `capital_initial` si rien d'autre.
    capital_total = 0.0
    if hasattr(portfolio_state, "valeur_totale"):
        try:
            capital_total = float(portfolio_state.valeur_totale() or 0.0)
        except Exception:
            capital_total = 0.0
    if capital_total <= 0:
        capital_total = float(getattr(portfolio_state, "capital_total", 0.0) or 0.0)
    if capital_total <= 0:
        capital_total = float(getattr(portfolio_state, "capital_initial", 0.0) or 0.0)

    if capital_total <= 0:
        # Portefeuille vide ou non initialisé : on ne peut pas calculer de ratio,
        # on autorise l'ouverture (ce sera la première position).
        return ConstraintResult(
            allowed=True,
            reason="capital_total=0, pas de contrainte applicable",
            sector=secteur_key,
            current_pct=0.0,
            projected_pct=0.0,
            cap_pct=cap,
        )

    # Exposition actuelle par secteur
    try:
        exposures = portfolio_state.exposition_sectorielle() or {}
    except Exception as exc:
        logger.warning("[Constraints] exposition_sectorielle() échec : %s", exc)
        exposures = {}

    current_pct = float(exposures.get(secteur_key, 0.0))
    montant = abs(float(montant_propose_euros or 0.0))
    projected_pct = current_pct + (montant / capital_total)

    if projected_pct > cap:
        return ConstraintResult(
            allowed=False,
            reason=(
                f"Ouverture refusée : dépassement du cap sectoriel "
                f"({projected_pct:.1%} > {cap:.0%}). Réduire la taille ou "
                f"attendre la clôture d'une position existante."
            ),
            sector=secteur_key,
            current_pct=current_pct,
            projected_pct=projected_pct,
            cap_pct=cap,
        )

    return ConstraintResult(
        allowed=True,
        reason=f"OK ({projected_pct:.1%} <= {cap:.0%})",
        sector=secteur_key,
        current_pct=current_pct,
        projected_pct=projected_pct,
        cap_pct=cap,
    )


# ---------------------------------------------------------------------------
# Cap de corrélation cross-sectionnelle
# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    allowed: bool
    reason: str
    ticker_proposed: str
    worst_partner: Optional[str]  # ticker existant le plus corrélé
    worst_rho: float  # valeur du pire ρ (peut être < 0 en valeur absolue)
    cap_rho: float  # seuil MAX_PAIRWISE_CORRELATION effectif

    def summary(self) -> str:
        verdict = "OK" if self.allowed else "REFUSED"
        partner = self.worst_partner or "—"
        return (
            f"[{verdict}] Corrélation {self.ticker_proposed} vs portfolio : "
            f"max |ρ| = {abs(self.worst_rho):.2f} avec {partner} "
            f"(cap {self.cap_rho:.2f}). {self.reason}"
        )


# Cache des log-returns yfinance (ticker → (fetched_at_epoch, list[float]))
# TTL 24h — la corrélation ne change pas dans la journée, inutile de recharger
# pour chaque nouvel article qui cite le même ticker.
_CORR_CACHE: dict[str, tuple[float, list[float]]] = {}
_CORR_CACHE_TTL_SEC = 24 * 3600


def _fetch_log_returns(
    ticker: str,
    lookback_days: int,
) -> list[float]:
    """
    Télécharge les log-returns quotidiens des `lookback_days` derniers jours
    via yfinance. Retourne [] si ticker invalide / pas de données.

    Cache TTL 24h — évite 20+ appels yfinance redondants dans une session.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return []

    now = time.time()
    cached = _CORR_CACHE.get(ticker)
    if cached is not None and (now - cached[0]) < _CORR_CACHE_TTL_SEC:
        return cached[1]

    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance indispo — pas de corrélation calculable.")
        return []

    try:
        period = f"{max(lookback_days + 10, 30)}d"
        hist = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.debug("[Corr] yfinance fail %s : %s", ticker, exc)
        return []

    if hist is None or hist.empty or "Close" not in hist.columns:
        return []

    closes = hist["Close"].dropna().to_numpy().flatten()
    if len(closes) < 10:
        return []

    # Log-returns (log(P_t / P_{t-1})) — plus stable numériquement que pct
    # pour les titres volatils et plus cohérent en CAPM/FF3.
    returns: list[float] = []
    for i in range(1, len(closes)):
        p_prev, p_curr = float(closes[i - 1]), float(closes[i])
        if p_prev > 0 and p_curr > 0:
            returns.append(math.log(p_curr / p_prev))

    # On garde les `lookback_days` derniers points exploitables.
    returns = returns[-lookback_days:] if len(returns) > lookback_days else returns

    _CORR_CACHE[ticker] = (now, returns)
    return returns


def _pearson(xs: list[float], ys: list[float]) -> float:
    """
    Pearson ρ sur séries alignées (min length). Retourne 0.0 si non calculable.
    """
    n = min(len(xs), len(ys))
    if n < 10:
        return 0.0
    xs = xs[-n:]
    ys = ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    if dx2 <= 0.0 or dy2 <= 0.0:
        return 0.0
    denom = math.sqrt(dx2 * dy2)
    return num / denom if denom > 0 else 0.0


def check_pairwise_correlation(
    portfolio_state,
    ticker_propose: str,
    cap_rho: Optional[float] = None,
    lookback_days: Optional[int] = None,
) -> CorrelationResult:
    """
    Vérifie que la position proposée (`ticker_propose`) n'est pas fortement
    corrélée à une position déjà ouverte. Si max(|ρ|) > cap_rho, refus.

    Args:
        portfolio_state : PortfolioState courant (pour lister les positions)
        ticker_propose  : ticker visé par l'ouverture
        cap_rho         : override du seuil (None → MAX_PAIRWISE_CORRELATION)
        lookback_days   : fenêtre de calcul (None → CORRELATION_LOOKBACK_DAYS)

    Returns:
        CorrelationResult — `.allowed` et `.summary()` pour logger.

    Règle de décision (signe de ρ) :
      On refuse si |ρ| > cap. Un ρ négatif -0.85 n'est pas non plus souhaité :
      l'algo pourrait "hedger" mais en réalité on double le pari sur le même
      facteur systémique (qui bouge de façon opposée). Kacperczyk-Sialm-Zheng
      (2005) montrent que les vrais gestionnaires actifs évitent les deux.
    """
    cap = cap_rho if cap_rho is not None else MAX_PAIRWISE_CORRELATION
    look = lookback_days if lookback_days is not None else CORRELATION_LOOKBACK_DAYS
    ticker_propose = (ticker_propose or "").upper().strip()

    positions = getattr(portfolio_state, "positions", {}) or {}
    # Filtrer les tickers à considérer (exclure celui que l'on ouvre / renforce)
    existing = [t for t in positions.keys() if t and t.upper() != ticker_propose]
    if not existing:
        return CorrelationResult(
            allowed=True,
            reason="Aucune position existante — pas de risque de corrélation.",
            ticker_proposed=ticker_propose,
            worst_partner=None,
            worst_rho=0.0,
            cap_rho=cap,
        )

    # Fetch log-returns du ticker proposé
    r_new = _fetch_log_returns(ticker_propose, look)
    if len(r_new) < 10:
        # Pas assez de données pour juger → on autorise (conservatisme
        # "faute de preuve, on laisse passer", sinon on bloque les IPO).
        return CorrelationResult(
            allowed=True,
            reason=f"Pas assez d'historique pour {ticker_propose} (<10 j).",
            ticker_proposed=ticker_propose,
            worst_partner=None,
            worst_rho=0.0,
            cap_rho=cap,
        )

    worst_partner: Optional[str] = None
    worst_rho: float = 0.0

    for t in existing:
        r_t = _fetch_log_returns(t, look)
        if len(r_t) < 10:
            continue
        rho = _pearson(r_new, r_t)
        if abs(rho) > abs(worst_rho):
            worst_rho = rho
            worst_partner = t

    if abs(worst_rho) > cap:
        return CorrelationResult(
            allowed=False,
            reason=(
                f"Ouverture refusée : |ρ({ticker_propose}, {worst_partner})| = "
                f"{abs(worst_rho):.2f} > cap {cap:.2f}. Risque dupliqué "
                f"(Ang-Chen 2002 : les corrélations explosent en régime stressé)."
            ),
            ticker_proposed=ticker_propose,
            worst_partner=worst_partner,
            worst_rho=worst_rho,
            cap_rho=cap,
        )

    return CorrelationResult(
        allowed=True,
        reason=f"OK — pire |ρ| = {abs(worst_rho):.2f} <= cap {cap:.2f}",
        ticker_proposed=ticker_propose,
        worst_partner=worst_partner,
        worst_rho=worst_rho,
        cap_rho=cap,
    )


if __name__ == "__main__":
    # Smoke test 1 : concentration sectorielle (ne touche pas le réseau)
    class _MockPortfolio:
        def valeur_totale(self):  # type: ignore[no-untyped-def]
            return 100_000.0

        def exposition_sectorielle(self):  # type: ignore[no-untyped-def]
            return {"Technology": 0.25, "Finance": 0.10}

    r1 = check_sector_concentration(_MockPortfolio(), "Technology", 4_000)
    print(r1.summary())  # 0.25 + 0.04 = 0.29 < 0.30 => OK
    r2 = check_sector_concentration(_MockPortfolio(), "Technology", 10_000)
    print(r2.summary())  # 0.25 + 0.10 = 0.35 > 0.30 => REFUSED
    r3 = check_sector_concentration(_MockPortfolio(), "Energy", 20_000)
    print(r3.summary())  # 0.00 + 0.20 = 0.20 < 0.30 => OK

    # Smoke test 2 : corrélation (hors réseau → portfolio vide)
    class _EmptyPortfolio:
        positions: dict = {}

    r4 = check_pairwise_correlation(_EmptyPortfolio(), "AAPL")
    print(r4.summary())  # OK, aucune position existante
