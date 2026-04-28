"""
portfolio_state.py — État du Portefeuille en Temps Réel

Références :
  [1] Markowitz, H. (1952). "Portfolio Selection."
      Journal of Finance, 7(1), 77-91.
      → Exposition sectorielle, diversification, frontière efficiente

  [2] Shefrin, H. & Statman, M. (1985). "The Disposition to Sell Winners Too
      Early and Ride Losers Too Long: Theory and Evidence."
      Journal of Finance, 40(3), 777-790.
      → Disposition Effect : imposer un suivi discipliné du PnL par position

  [3] DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive
      Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
      Review of Financial Studies, 22(5), 1915-1953.
      → La concentration sectorielle > 40% dégrade significativement le Sharpe

  [4] Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). "Drawdown Measure
      in Portfolio Optimization." International Journal of Theoretical and
      Applied Finance, 8(1), 13-58.
      → Maximum Drawdown comme mesure de risque de portefeuille
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = Path("data/portfolio_state.json")


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """
    Une position ouverte dans le portefeuille.
    Le PnL est calculé à la demande pour refléter le prix actuel.
    """

    ticker: str
    nb_actions: float
    prix_entree: float  # Prix moyen d'achat (VWAP si entrées multiples)
    secteur: str = "Inconnu"
    industrie: str = "Inconnu"
    date_entree: str = ""  # ISO 8601

    # Prix actuel (mis à jour via yfinance)
    prix_actuel: float = 0.0

    @property
    def valeur_entree(self) -> float:
        return self.nb_actions * self.prix_entree

    @property
    def valeur_actuelle(self) -> float:
        return self.nb_actions * self.prix_actuel if self.prix_actuel > 0 else self.valeur_entree

    @property
    def pnl_absolu(self) -> float:
        return self.valeur_actuelle - self.valeur_entree

    @property
    def pnl_pct(self) -> float:
        if self.prix_entree == 0:
            return 0.0
        ref = self.prix_actuel if self.prix_actuel > 0 else self.prix_entree
        return (ref - self.prix_entree) / self.prix_entree

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "nb_actions": self.nb_actions,
            "prix_entree": self.prix_entree,
            "secteur": self.secteur,
            "industrie": self.industrie,
            "date_entree": self.date_entree,
            "prix_actuel": self.prix_actuel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(
            ticker=d["ticker"],
            nb_actions=d["nb_actions"],
            prix_entree=d["prix_entree"],
            secteur=d.get("secteur", "Inconnu"),
            industrie=d.get("industrie", "Inconnu"),
            date_entree=d.get("date_entree", ""),
            prix_actuel=d.get("prix_actuel", 0.0),
        )


# ---------------------------------------------------------------------------
# PortfolioState
# ---------------------------------------------------------------------------


@dataclass
class PortfolioState:
    """
    État complet du portefeuille à un instant t.

    Basé sur Markowitz (1952) : le portefeuille est caractérisé par son
    exposition aux actifs, la diversification sectorielle et le capital libre.
    """

    positions: dict[str, Position] = field(default_factory=dict)
    cash_disponible: float = 10_000.0
    capital_initial: float = 10_000.0  # Pour calcul drawdown (Chekhlov et al. 2005)
    peak_valeur: float = 10_000.0  # High-water mark pour MDD
    historique_trades: list[dict] = field(default_factory=list)
    derniere_maj: str = ""

    # --- Métriques agrégées ---

    def valeur_totale(self) -> float:
        """Valeur totale : positions ouvertes + cash."""
        return self.valeur_positions() + self.cash_disponible

    def valeur_positions(self) -> float:
        return sum(p.valeur_actuelle for p in self.positions.values())

    def drawdown_actuel(self) -> float:
        """
        Drawdown vs capital initial.
        Ref: Chekhlov et al. (2005) — mesure standard de risque de portefeuille.
        Négatif = perte, positif = gain.
        """
        total = self.valeur_totale()
        return (total - self.capital_initial) / self.capital_initial

    def max_drawdown_vs_peak(self) -> float:
        """MDD vs high-water mark (Chekhlov et al. 2005)."""
        total = self.valeur_totale()
        if self.peak_valeur > 0:
            self.peak_valeur = max(self.peak_valeur, total)
            return (total - self.peak_valeur) / self.peak_valeur
        return 0.0

    def exposition_sectorielle(self) -> dict[str, float]:
        """
        Fraction du portefeuille investie par secteur.
        Ref: DeMiguel et al. (2009) — seuil d'alerte à 40% par secteur.
        """
        total = self.valeur_totale()
        if total == 0:
            return {}
        secteurs: dict[str, float] = {}
        for pos in self.positions.values():
            secteurs[pos.secteur] = secteurs.get(pos.secteur, 0.0) + pos.valeur_actuelle
        return {s: round(v / total, 4) for s, v in secteurs.items()}

    def pnl_total(self) -> dict[str, float]:
        """PnL global : absolu (€) et relatif (%)."""
        pnl_abs = sum(p.pnl_absolu for p in self.positions.values())
        pnl_pct = pnl_abs / self.capital_initial if self.capital_initial > 0 else 0.0
        return {"absolu": round(pnl_abs, 2), "pct": round(pnl_pct, 4)}

    def exposition_ticker(self, ticker: str) -> float:
        """Fraction du portefeuille investie dans un ticker spécifique."""
        total = self.valeur_totale()
        if total == 0 or ticker not in self.positions:
            return 0.0
        return self.positions[ticker].valeur_actuelle / total

    def cash_investissable(self, cash_reserve_min: float) -> float:
        """
        Cash réellement disponible après déduction de la réserve de liquidités.
        Ref: MiFID II — maintien d'une réserve de liquidité suffisante.
        """
        reserve = cash_reserve_min * self.capital_initial
        return max(0.0, self.cash_disponible - reserve)

    def nb_positions_ouvertes(self) -> int:
        return len(self.positions)

    def get_position_pnl(self, ticker: str) -> Optional[dict]:
        """Retourne le PnL détaillé d'une position."""
        if ticker not in self.positions:
            return None
        pos = self.positions[ticker]
        return {
            "ticker": ticker,
            "nb_actions": pos.nb_actions,
            "prix_entree": pos.prix_entree,
            "prix_actuel": pos.prix_actuel,
            "pnl_absolu": round(pos.pnl_absolu, 2),
            "pnl_pct": round(pos.pnl_pct, 4),
            "valeur_entree": round(pos.valeur_entree, 2),
            "valeur_actuelle": round(pos.valeur_actuelle, 2),
        }

    # --- Gestion des positions ---

    def enregistrer_achat(
        self,
        ticker: str,
        nb_actions: float,
        prix: float,
        secteur: str = "Inconnu",
        industrie: str = "Inconnu",
    ) -> None:
        """
        Enregistre un achat. Calcule le VWAP si position existante.
        Ref: Shefrin & Statman (1985) — suivi discipliné du prix de revient.
        """
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_actions = pos.nb_actions + nb_actions
            # Prix moyen pondéré (VWAP)
            pos.prix_entree = (pos.prix_entree * pos.nb_actions + prix * nb_actions) / total_actions
            pos.nb_actions = total_actions
            pos.prix_actuel = prix
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                nb_actions=nb_actions,
                prix_entree=prix,
                secteur=secteur,
                industrie=industrie,
                date_entree=datetime.now(timezone.utc).isoformat(),
                prix_actuel=prix,
            )

        cout = nb_actions * prix
        self.cash_disponible = max(0.0, self.cash_disponible - cout)
        self._log_trade("ACHAT", ticker, nb_actions, prix)
        logger.info(
            "[Portfolio] ACHAT %s: %.2f actions @ %.2f€ | Coût: %.2f€ | Cash restant: %.2f€",
            ticker,
            nb_actions,
            prix,
            cout,
            self.cash_disponible,
        )

    def enregistrer_vente(self, ticker: str, prix: float) -> Optional[float]:
        """
        Clôture une position et crédite le cash.
        Retourne le PnL réalisé en euros.
        """
        if ticker not in self.positions:
            logger.warning("[Portfolio] Tentative de vente de %s mais position inexistante.", ticker)
            return None
        pos = self.positions[ticker]
        pos.prix_actuel = prix
        pnl = pos.pnl_absolu
        recette = pos.valeur_actuelle
        self._log_trade("VENTE", ticker, pos.nb_actions, prix, pnl_realise=pnl)
        del self.positions[ticker]
        self.cash_disponible += recette
        logger.info(
            "[Portfolio] VENTE %s: %.2f€ encaissés | PnL réalisé: %+.2f€ | Cash: %.2f€",
            ticker,
            recette,
            pnl,
            self.cash_disponible,
        )
        return pnl

    def _log_trade(
        self,
        action: str,
        ticker: str,
        nb_actions: float,
        prix: float,
        pnl_realise: float = 0.0,
    ) -> None:
        self.historique_trades.append(
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "ticker": ticker,
                "nb_actions": nb_actions,
                "prix": prix,
                "pnl_realise": round(pnl_realise, 2),
            }
        )

    def to_dict(self) -> dict:
        return {
            "positions": {t: p.to_dict() for t, p in self.positions.items()},
            "cash_disponible": self.cash_disponible,
            "capital_initial": self.capital_initial,
            "peak_valeur": self.peak_valeur,
            "historique_trades": self.historique_trades[-100:],  # Garde les 100 derniers
            "derniere_maj": self.derniere_maj,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        ps = cls()
        ps.positions = {t: Position.from_dict(p) for t, p in d.get("positions", {}).items()}
        ps.cash_disponible = d.get("cash_disponible", 10_000.0)
        ps.capital_initial = d.get("capital_initial", 10_000.0)
        ps.peak_valeur = d.get("peak_valeur", ps.capital_initial)
        ps.historique_trades = d.get("historique_trades", [])
        ps.derniere_maj = d.get("derniere_maj", "")
        return ps

    def log_summary(self) -> None:
        """Log un résumé du portefeuille."""
        total = self.valeur_totale()
        pnl = self.pnl_total()
        sect = self.exposition_sectorielle()
        logger.info(
            "[Portfolio] Valeur totale: %.2f€ | PnL: %+.2f€ (%+.1f%%) | "
            "Positions: %d | Cash: %.2f€ | Drawdown: %+.1f%%",
            total,
            pnl["absolu"],
            pnl["pct"] * 100,
            self.nb_positions_ouvertes(),
            self.cash_disponible,
            self.drawdown_actuel() * 100,
        )
        for secteur, expo in sect.items():
            logger.info("  Secteur %s: %.0f%%", secteur, expo * 100)


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------


def load_portfolio_state() -> PortfolioState:
    """Charge l'état du portefeuille depuis data/portfolio_state.json."""
    if not PORTFOLIO_PATH.exists():
        state = PortfolioState()
        save_portfolio_state(state)
        logger.info("[Portfolio] Portefeuille vide initialisé → %s", PORTFOLIO_PATH)
        return state
    try:
        data = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        state = PortfolioState.from_dict(data)
        state.log_summary()
        return state
    except Exception as e:
        logger.error("[Portfolio] Erreur chargement : %s — portefeuille vide.", e)
        return PortfolioState()


def save_portfolio_state(state: PortfolioState) -> None:
    """Persiste l'état du portefeuille."""
    state.derniere_maj = datetime.now(timezone.utc).isoformat()
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_PATH.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def refresh_portfolio_prices(state: PortfolioState) -> None:
    """
    Met à jour les prix actuels via yfinance (best-effort, non bloquant).
    Ref: Markowitz (1952) — la valorisation temps réel est nécessaire pour
    calculer correctement l'exposition du portefeuille.
    """
    if not state.positions:
        return
    try:
        import yfinance as yf

        tickers = list(state.positions.keys())
        data = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        close = data["Close"] if "Close" in data.columns else None
        if close is None:
            return
        for ticker in tickers:
            try:
                if ticker in close.columns:
                    price = float(close[ticker].dropna().iloc[-1])
                else:
                    price = float(close.dropna().iloc[-1])
                state.positions[ticker].prix_actuel = price
            except Exception:
                pass
        logger.info("[Portfolio] Prix mis à jour pour: %s", tickers)
    except Exception as e:
        logger.warning("[Portfolio] Mise à jour prix yfinance échouée : %s", e)
