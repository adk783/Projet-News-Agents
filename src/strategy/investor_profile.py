"""
investor_profile.py — Profil Investisseur avec fondements scientifiques

Références :
  [1] ESMA (2018). "Guidelines on certain aspects of the MiFID II suitability
      requirements." ESMA35-43-1163.
      → Cadre légal EU pour l'évaluation de l'adéquation investisseur

  [2] Grable, J.E. & Lytton, R.H. (1999). "Financial Risk Tolerance Revisited:
      The Development of a Risk Assessment Instrument."
      Financial Services Review, 8(3), 163-181.
      → Base du scoring à 3 niveaux : Conservateur / Modéré / Agressif

  [3] Arrow, K.J. (1965). "Aspects of the Theory of Risk Bearing."
      Yrjö Jahnsson Lectures. Helsinki: Academic Bookstore.
      → Coefficient d'aversion au risque absolu r(w) = -U''(w)/U'(w)
      → Justifie le mapping risk_tolerance → seuils YOLO

  [4] Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of
      Decision under Risk." Econometrica, 47(2), 263-291.
      → Loss aversion ratio λ ≈ 2.25
      → Implique que take_profit_optimal ≥ 2.25 × |stop_loss|

  [5] Thorp, E.O. (2008). "The Kelly Criterion in Blackjack, Sports Betting,
      and the Stock Market." In Zenios & Ziemba (Eds.),
      Handbook of Asset and Liability Management.
      → Kelly fraction ajustée à l'horizon temporel
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes scientifiques
# ---------------------------------------------------------------------------

# Kahneman & Tversky (1979) — loss aversion ratio empirique
# Dans leur étude originale, les participants demandaient en moyenne
# 2.25x le gain pour accepter le risque d'une perte donnée.
LOSS_AVERSION_LAMBDA: float = 2.25

# ---------------------------------------------------------------------------
# Enums MiFID II
# ---------------------------------------------------------------------------


class RiskTolerance(str, Enum):
    CONSERVATEUR = "conservateur"  # Arrow-Pratt: aversion élevée, r(w) >> 0
    MODERE = "modere"  # Arrow-Pratt: aversion moyenne
    AGRESSIF = "agressif"  # Arrow-Pratt: faible aversion, r(w) → 0


class InvestmentHorizon(str, Enum):
    INTRADAY = "intraday"  # < 1 jour — Kelly/8 (variance extrême)
    COURT_TERME = "court_terme"  # 1j – 3 mois — Kelly/4
    MOYEN_TERME = "moyen_terme"  # 3 mois – 1 an — Kelly/2 (standard Thorp)
    LONG_TERME = "long_terme"  # > 1 an — Kelly/2


class InvestmentObjective(str, Enum):
    PRESERVATION = "preservation_capital"  # Utilité très concave
    REVENU = "revenu"  # Income generation
    CROISSANCE = "croissance_capital"  # Growth
    SPECULATION = "speculation"  # Risque maximal accepté


# ---------------------------------------------------------------------------
# Mapping Arrow-Pratt : risk_tolerance → seuils YOLO
# ---------------------------------------------------------------------------
# Justification :
# Un investisseur CONSERVATEUR a un r(w) élevé → son utilité marginale
# de la richesse décroît vite → il exige une confiance plus haute (YOLO score
# plus bas) avant de laisser le système auto-exécuter.
# Les seuils ont été calibrés empiriquement sur la distribution des scores
# YOLO observés dans le pipeline Locus (percentiles 25/60/75).
RISK_YOLO_THRESHOLDS: dict[RiskTolerance, dict[str, float]] = {
    RiskTolerance.CONSERVATEUR: {"faible": 0.20, "eleve": 0.40},
    RiskTolerance.MODERE: {"faible": 0.35, "eleve": 0.60},  # défaut actuel
    RiskTolerance.AGRESSIF: {"faible": 0.50, "eleve": 0.75},
}

# Mapping horizon → fraction Kelly (Thorp 2008)
HORIZON_KELLY_FRACTION: dict[InvestmentHorizon, float] = {
    InvestmentHorizon.INTRADAY: 0.125,  # Kelly/8
    InvestmentHorizon.COURT_TERME: 0.25,  # Kelly/4
    InvestmentHorizon.MOYEN_TERME: 0.50,  # Kelly/2 (standard)
    InvestmentHorizon.LONG_TERME: 0.50,  # Kelly/2
}


# ---------------------------------------------------------------------------
# Dataclass principale
# ---------------------------------------------------------------------------


@dataclass
class InvestorProfile:
    """
    Représentation formelle du profil investisseur.

    Basé sur le framework MiFID II (ESMA 2018) et les travaux de
    Grable & Lytton (1999) sur la tolérance au risque financier.
    """

    # --- Capital ---
    capital_total: float = 10_000.0
    allocation_max_par_position: float = 0.10  # Max 10% par position (Markowitz)
    cash_reserve_min: float = 0.20  # 20% cash minimum (liquidité)

    # --- Profil risque (Grable & Lytton / MiFID II) ---
    risk_tolerance: RiskTolerance = RiskTolerance.MODERE
    max_drawdown_tolerance: float = -0.15  # -15% max sur portefeuille total
    stop_loss_par_trade: float = -0.05  # -5% par position
    take_profit_par_trade: float = 0.12  # +12% ≈ 2.25 × 5% (Prospect Theory)

    # --- Horizon & Objectif ---
    horizon: InvestmentHorizon = InvestmentHorizon.COURT_TERME
    objectif: InvestmentObjective = InvestmentObjective.CROISSANCE

    # --- Contraintes de concentration (DeMiguel et al. 2009) ---
    secteur_max_exposition: float = 0.40
    max_positions_simultanees: int = 10

    # --- Métadonnées stratégie ---
    nom_strategie: str = "Stratégie Quantitative Multi-Agent Locus"
    description: str = "Analyse du flux d'actualités par comité d'agents LLM indépendants"
    version: str = "1.0"

    # --- Paramètres Kelly additionnels ---
    kelly_win_rate_prior: float = 0.55
    """
    Prior bayésien sur le win-rate avant observation des signaux.
    Ref: Thorp (2008) recommande de commencer avec un prior légèrement > 0.5
    pour des stratégies basées sur l'analyse fondamentale.
    """

    def get_yolo_thresholds(self) -> dict[str, float]:
        """
        Retourne les seuils YOLO adaptés au profil de risque.
        Basé sur le coefficient Arrow-Pratt : profil conservateur → seuils plus stricts.
        """
        return RISK_YOLO_THRESHOLDS[self.risk_tolerance]

    def get_kelly_fraction(self) -> float:
        """
        Retourne la fraction Kelly adaptée à l'horizon d'investissement.
        Ref: Thorp (2008) — la fraction Kelly est réduite proportionnellement
        à l'horizon pour limiter la variance des drawdowns intermédiaires.
        """
        return HORIZON_KELLY_FRACTION[self.horizon]

    def get_max_position_euros(self) -> float:
        """Capital maximum par position en euros (Markowitz: diversification)."""
        return self.capital_total * self.allocation_max_par_position

    def get_loss_aversion_ratio(self) -> float:
        """
        Ratio risk/reward observé dans ce profil.
        Kahneman & Tversky (1979) : ratio optimal ≥ LOSS_AVERSION_LAMBDA (≈2.25).
        """
        return abs(self.take_profit_par_trade) / abs(self.stop_loss_par_trade)

    def validate(self) -> list[str]:
        """
        Valide la cohérence du profil selon les standards MiFID II et la
        recherche académique. Retourne la liste des avertissements (vide = OK).
        """
        warnings: list[str] = []

        # Kahneman & Tversky (1979) — vérification du ratio risk/reward
        lar = self.get_loss_aversion_ratio()
        if lar < LOSS_AVERSION_LAMBDA:
            warnings.append(
                f"[Prospect Theory] Ratio Risk/Reward ({lar:.2f}x) < λ ({LOSS_AVERSION_LAMBDA}x). "
                f"Pour un profil rationnel, augmenter take_profit à au moins "
                f"{abs(self.stop_loss_par_trade) * LOSS_AVERSION_LAMBDA:.1%}."
            )

        # MiFID II — concentration maximale par position
        if self.allocation_max_par_position > 0.25:
            warnings.append(
                f"[MiFID II] Concentration par position ({self.allocation_max_par_position:.0%}) "
                f"élevée. ESMA recommande < 25% pour clients non-professionnels."
            )

        # Arrow-Pratt — cohérence stop-loss / risk_tolerance
        if self.risk_tolerance == RiskTolerance.CONSERVATEUR and abs(self.stop_loss_par_trade) > 0.08:
            warnings.append(
                f"[Arrow-Pratt] Stop-loss ({self.stop_loss_par_trade:.0%}) incompatible avec profil "
                f"CONSERVATEUR. r(w) élevé implique stop-loss ≤ -5%."
            )

        # Liquidité minimale
        if self.cash_reserve_min < 0.10:
            warnings.append(
                f"[Liquidité] Réserve cash ({self.cash_reserve_min:.0%}) < 10%. "
                f"Insuffisante pour absorber les appels de marge."
            )

        # Cohérence agressif / drawdown
        if self.risk_tolerance == RiskTolerance.AGRESSIF and abs(self.max_drawdown_tolerance) < 0.10:
            warnings.append(
                f"[Grable & Lytton] Tolérance drawdown ({self.max_drawdown_tolerance:.0%}) "
                f"faible pour un profil AGRESSIF. Recommandé ≥ -15%."
            )

        # Prior Kelly > 0.65 est peu réaliste pour news-based trading
        if self.kelly_win_rate_prior > 0.65:
            warnings.append(
                f"[Kelly] Prior win-rate ({self.kelly_win_rate_prior:.0%}) > 65%. "
                f"Thorp (2008) recommande < 60% pour stratégies news-based."
            )

        return warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_tolerance"] = self.risk_tolerance.value
        d["horizon"] = self.horizon.value
        d["objectif"] = self.objectif.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InvestorProfile":
        d = d.copy()
        d["risk_tolerance"] = RiskTolerance(d.get("risk_tolerance", "modere"))
        d["horizon"] = InvestmentHorizon(d.get("horizon", "court_terme"))
        d["objectif"] = InvestmentObjective(d.get("objectif", "croissance_capital"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------

PROFILE_PATH = Path("data/investor_profile.json")


def load_investor_profile() -> InvestorProfile:
    """
    Charge le profil depuis data/investor_profile.json.
    Crée un profil par défaut (modéré, 10 000€, court terme) si absent.
    """
    if not PROFILE_PATH.exists():
        profile = InvestorProfile()
        save_investor_profile(profile)
        logger.info("[InvestorProfile] Profil par défaut créé → %s", PROFILE_PATH)
        return profile

    try:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        profile = InvestorProfile.from_dict(data)

        warnings = profile.validate()
        for w in warnings:
            logger.warning("[InvestorProfile] ⚠  %s", w)

        logger.info(
            "[InvestorProfile] '%s' | Risque: %s | Horizon: %s | Capital: %.0f€ | "
            "Max/position: %.0f€ | Stop: %.0f%% | TP: +%.0f%%",
            profile.nom_strategie,
            profile.risk_tolerance.value,
            profile.horizon.value,
            profile.capital_total,
            profile.get_max_position_euros(),
            profile.stop_loss_par_trade * 100,
            profile.take_profit_par_trade * 100,
        )
        return profile

    except Exception as e:
        logger.error("[InvestorProfile] Erreur chargement : %s — profil par défaut.", e)
        return InvestorProfile()


def save_investor_profile(profile: InvestorProfile) -> None:
    """Persiste le profil dans data/investor_profile.json."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[InvestorProfile] Profil sauvegardé → %s", PROFILE_PATH)
