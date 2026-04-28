"""
position_sizer.py — Sizing de Position Optimal

Implémente 3 méthodes scientifiquement fondées, combinées en un résultat final
conservateur (minimum des méthodes ou cap de profil).

Références :
  [1] Kelly, J.L. (1956). "A New Interpretation of Information Rate."
      Bell System Technical Journal, 35(4), 917-926.
      → Formule de base: f* = (p·b - q) / b
      → Maximise le log-utility, donc la croissance géométrique à long terme

  [2] Thorp, E.O. (1997). "The Kelly Criterion: Implementation Issues."
      NBER Working Paper.
      Thorp, E.O. (2008). In Zenios & Ziemba (Eds.),
      Handbook of Asset and Liability Management.
      → Half-Kelly (f*/2) réduit la variance de ~75% pour ~0% de perte de rendement
      → La fraction Kelly est ajustée à l'horizon temporel

  [3] MacLean, L.C., Thorp, E.O., & Ziemba, W.T. (2010).
      "Good and Bad Properties of the Kelly Criterion."
      Quantitative Finance, 10(1), 3-9.
      → Justifie la réduction fractionnelle du Kelly selon le profil

  [4] Wilder, J.W. (1978). "New Concepts in Technical Trading Systems."
      Trend Research.
      → ATR (Average True Range) comme proxy de volatilité pour le sizing

  [5] Van Tharp (1998). "Trade Your Way to Financial Freedom."
      McGraw-Hill.
      → Fixed Fractional Position Sizing : f_ff = R × signal_strength

  [6] Browne, S. (1999). "Reaching Goals by a Deadline: Digital Options and
      Continuous-Time Active Portfolio Management."
      Advances in Applied Probability, 31(2), 551-577.
      → Sizing optimal dans un contexte d'objectif de rendement cible
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Résultat du sizing
# ---------------------------------------------------------------------------


@dataclass
class PositionSizeResult:
    """
    Résultat du calcul de taille de position.
    Expose la méthode utilisée et la justification pour la traçabilité.
    """

    ticker: str

    # Résultat final
    montant_euros: float  # Montant à investir en €
    nb_actions: float  # Nombre d'actions
    action_type: str  # ACHETER / RENFORCER / TENIR / RÉDUIRE / COUPER

    # Détail du calcul Kelly
    kelly_f_star: float  # Fraction Kelly brute f*
    kelly_fraction_applied: float  # Fraction Kelly après ajustement horizon (f*/N)
    kelly_montant: float  # Montant selon Kelly ajusté
    kelly_win_prob: float  # Probabilité de gain estimée
    kelly_odds_ratio: float  # Ratio gain/perte (b dans la formule)

    # Fixed Fractional (Van Tharp)
    ff_montant: float  # Montant selon Fixed Fractional

    # Contraintes appliquées
    capped_by_profile: bool  # Le résultat a été plafonné par le profil
    capped_by_cash: bool  # Le résultat a été limité par le cash disponible
    capped_by_concentration: bool  # Limité par la concentration sectorielle
    methode_finale: str  # "kelly_half" / "fixed_fractional" / "capped_profile"

    # Contexte pour le prompt
    justification: str  # Explication lisible pour injection dans le débat


# ---------------------------------------------------------------------------
# Estimation des paramètres Kelly
# ---------------------------------------------------------------------------


def _estimate_win_probability(
    impact_strength: float,
    trust_trade_score: float,
    prior_win_rate: float = 0.55,
) -> float:
    """
    Estime la probabilité de gain à partir des métriques du pipeline.

    Approche bayésienne naïve :
    - Prior : kelly_win_rate_prior du profil (défaut 55%)
    - Likelihood : impact_strength × trust_trade_score (signal composite)

    La mise à jour bayésienne est simplifiée par un mélange pondéré
    entre le prior et l'évidence signal, pondéré par la confiance (trust).

    Ref: Thorp (2008) — estimer p depuis les caractéristiques du signal
    avant d'appliquer Kelly.

    Args:
        impact_strength    : force du signal [0,1] — produit de l'ABSA et FinBERT
        trust_trade_score  : fiabilité croisée numérique/sémantique [0,1]
        prior_win_rate     : prior bayésien du profil (Grable & Lytton 1999)

    Returns:
        p_estimated : probabilité de gain [0.45, 0.75] (clampée pour réalisme)
    """
    # Signal composite = produit des deux métriques indépendantes
    signal_composite = (impact_strength + trust_trade_score) / 2.0

    # Mélange prior / evidence (Bayes naïf) — trust_trade pondère la confiance
    # qu'on a dans le signal par rapport au prior
    alpha = trust_trade_score  # Plus trust est haut, plus on fait confiance au signal
    p_posterior = (1 - alpha) * prior_win_rate + alpha * (0.40 + signal_composite * 0.35)

    # Garde-fou faible : on autorise [0.15, 0.85] désormais (signal Vente possible).
    # L'ancien clamp [0.45, 0.75] masquait les signaux très forts et empêchait
    # Kelly de descendre à zéro quand l'edge est négligeable. Le scaling
    # bayésien (via kelly_scale) gère la sur-confidence maintenant.
    return round(max(0.15, min(0.85, p_posterior)), 4)


def _compute_kelly_f_star(
    win_prob: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    cost_bps: float = 0.0,
) -> float:
    """
    Calcule la fraction Kelly optimale f*, **nette des coûts** si fournis.

    Formule Kelly (1956) : f* = (p·b - q) / b
    où :
      p = probabilité de gain
      q = 1 - p (probabilité de perte)
      b = ratio gain/perte = (TP - c) / (|SL| + c)   ← inclut frais A/R

    Ref: Kelly (1956), Bell System Technical Journal.
         Almgren & Chriss (2001), Optimal execution — pour la correction nets.

    Returns:
        f* ∈ [0, 1] (clampé à 0 si négatif = edge négatif après frais)
    """
    c = cost_bps / 10_000.0
    tp_net = max(1e-6, abs(take_profit_pct) - c)
    sl_net = abs(stop_loss_pct) + c
    b = tp_net / sl_net
    if b <= 0:
        return 0.0
    q = 1.0 - win_prob
    f_star = (win_prob * b - q) / b
    return round(max(0.0, min(1.0, f_star)), 4)


def _apply_bayesian_uncertainty_penalty(
    f_star: float,
    p_var: float,
    kappa: float = 4.0,
) -> float:
    """
    Corrige Kelly pour l'incertitude paramétrique sur p (Medo & Pignatti 2013,
    Browne & Whitt 1996). Si la variance du posterior de p est grande, Kelly
    doit être sous-dimensionné :
        f_effective = f* / (1 + κ · Var[p])

    Args:
        f_star : fraction Kelly brute
        p_var  : variance totale du posterior Beta sur p (epistemic + aleatoric)
        kappa  : sensibilité (4.0 par défaut, aligné avec bayesian_aggregator)
    """
    denom = 1.0 + kappa * max(0.0, p_var)
    return round(f_star / denom, 6)


def _compute_fixed_fractional(
    capital_disponible: float,
    allocation_max_par_position: float,
    impact_strength: float,
    trust_trade_score: float,
) -> float:
    """
    Position sizing par fraction fixe ajustée à la force du signal.

    Ref: Van Tharp (1998) — Fixed Fractional Position Sizing.
    f_ff = allocation_max × signal_force_composite

    La allocation_max est le plafond ; la force du signal réduit la taille
    pour les signaux faibles.

    Returns:
        Montant en € à investir selon Fixed Fractional.
    """
    signal_force = (impact_strength + trust_trade_score) / 2.0
    # Linéaire entre 30% et 100% de l'allocation max selon la force du signal
    fraction_du_max = 0.30 + signal_force * 0.70
    return round(capital_disponible * allocation_max_par_position * fraction_du_max, 2)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def calculate_position_size(
    ticker: str,
    prix_actuel: float,
    signal_final: str,
    impact_strength: float,
    trust_trade_score: float,
    profile,  # InvestorProfile
    portfolio,  # PortfolioState
    secteur: str = "Inconnu",
    bayesian_consensus=None,  # Optional[BayesianConsensus] — Module ②
    adv_volume: float = 5_000_000.0,  # Average Daily Volume (actions), pour impact
    sigma_daily: float = 0.018,  # volatilité journalière estimée
    delay_ms: float = 0.0,  # délai signal-exécution
) -> PositionSizeResult:
    """
    Calcule la taille de position optimale selon Kelly Half + Fixed Fractional.

    Méthodologie (par ordre de priorité) :
    1. Calcul de p (win_prob) via Bayes naïf sur impact_strength + trust
    2. Calcul de f* via Kelly (1956)
    3. Application de la fraction ajustée à l'horizon (Thorp 2008)
    4. Comparaison avec Fixed Fractional (Van Tharp 1998)
    5. Prise du minimum des deux (approche conservatrice — MacLean et al. 2010)
    6. Application des contraintes du profil : cap allocation, cash, concentration

    Args:
        ticker          : symbole boursier
        prix_actuel     : prix actuel du titre
        signal_final    : "Achat" | "Vente" | "Neutre"
        impact_strength : [0,1] force du signal ABSA/FinBERT
        trust_trade_score : [0,1] score TrustTrade du pipeline
        profile         : InvestorProfile
        portfolio       : PortfolioState
        secteur         : secteur du ticker (pour contrôle concentration)

    Returns:
        PositionSizeResult avec montant, nb_actions, méthode, justification
    """
    from .investor_profile import InvestorProfile
    from .portfolio_state import PortfolioState

    assert isinstance(profile, InvestorProfile)
    assert isinstance(portfolio, PortfolioState)

    # --- Cas "HOLD" systémique / sectoriel / corrélation : aucun sizing ---
    if signal_final in ("HOLD_SYSTEMIC", "HOLD_SECTOR_CAP", "HOLD_CORR_CAP"):
        return _make_hold_result(
            ticker,
            prix_actuel,
            f"Signal override : {signal_final} — prise de position refusée par garde-fou.",
        )

    # --- Cas Neutre / Vente : pas de nouvelle position ---
    if signal_final == "Neutre":
        return _make_hold_result(ticker, prix_actuel, "Signal Neutre — aucune action suggérée.")

    if signal_final == "Vente" and ticker not in portfolio.positions:
        return _make_hold_result(ticker, prix_actuel, "Signal Vente mais aucune position ouverte à clôturer.")

    # --- Signal Vente sur position existante → COUPER ---
    if signal_final == "Vente" and ticker in portfolio.positions:
        return _make_exit_result(ticker, prix_actuel, portfolio, profile, trust_trade_score)

    # --- Signal Achat (nouvelle position ou renforcement) ---

    # Contrôle drawdown (Chekhlov et al. 2005)
    drawdown = portfolio.drawdown_actuel()
    if drawdown < profile.max_drawdown_tolerance:
        return _make_hold_result(
            ticker,
            prix_actuel,
            f"Drawdown actuel ({drawdown:.1%}) < limite tolérance ({profile.max_drawdown_tolerance:.1%}). "
            f"Aucun nouvel achat autorisé.",
        )

    # Contrôle nombre de positions (Markowitz 1952 — diversification)
    if portfolio.nb_positions_ouvertes() >= profile.max_positions_simultanees:
        return _make_hold_result(
            ticker,
            prix_actuel,
            f"Nombre de positions max atteint ({profile.max_positions_simultanees}). "
            f"Clôturer une position avant d'ouvrir une nouvelle.",
        )

    # Contrôle concentration sectorielle (DeMiguel et al. 2009)
    capped_concentration = False
    expo_secteur = portfolio.exposition_sectorielle().get(secteur, 0.0)
    if expo_secteur >= profile.secteur_max_exposition:
        return _make_hold_result(
            ticker,
            prix_actuel,
            f"Concentration sectorielle '{secteur}' ({expo_secteur:.0%}) ≥ limite "
            f"({profile.secteur_max_exposition:.0%}). Nouvelle position refusée.",
        )

    capital_dispo = portfolio.cash_investissable(profile.cash_reserve_min)
    if capital_dispo <= 0:
        return _make_hold_result(ticker, prix_actuel, "Cash investissable insuffisant après réserve de liquidités.")

    # --- Étape 1 : Estimation de p ---
    # Si le consensus bayésien est fourni (Module ②), on l'utilise directement
    # (posterior hiérarchique plutôt qu'heuristique naïve).
    if bayesian_consensus is not None and getattr(bayesian_consensus, "n_agents", 0) > 0:
        p = float(bayesian_consensus.p_mean)
        p_var = float(bayesian_consensus.p_var_total)
    else:
        p = _estimate_win_probability(impact_strength, trust_trade_score, profile.kelly_win_rate_prior)
        p_var = 0.0  # pas de propagation d'incertitude en mode legacy

    # --- Étape 1.5 : Estimation des coûts d'exécution (Almgren-Chriss + spread)
    # La taille exacte n'est pas connue avant d'avoir fini Kelly ; on pré-estime
    # sur la base de l'allocation moyenne du profil pour le coût marginal.
    try:
        from src.utils.execution_costs import (
            break_even_accuracy,
            compute_execution_costs,
        )

        notional_estim = profile.capital_total * profile.allocation_max_par_position * 0.5
        qty_estim = notional_estim / max(prix_actuel, 1e-3)
        cost_br = compute_execution_costs(
            quantity=qty_estim,
            price=prix_actuel,
            adv_volume=adv_volume,
            sigma_daily=sigma_daily,
            delay_ms=delay_ms,
        )
        cost_bps = cost_br.total_cost_bps
        be_acc = break_even_accuracy(profile.take_profit_par_trade, profile.stop_loss_par_trade, cost_bps)
    except Exception as exc:
        logger.debug("[PositionSizer] execution_costs disabled: %s", exc)
        cost_bps = 0.0
        be_acc = None

    # Si p est sous le seuil break-even → refuse d'ouvrir
    if be_acc is not None and p < be_acc and signal_final == "Achat":
        return _make_hold_result(
            ticker,
            prix_actuel,
            f"p estimée ({p:.1%}) < break-even accuracy ({be_acc:.1%}) après "
            f"coûts ({cost_bps:.1f} bps). Trade refusé (edge négatif net).",
        )

    # --- Étape 2 : Kelly f* net des frais ---
    f_star = _compute_kelly_f_star(
        win_prob=p,
        take_profit_pct=profile.take_profit_par_trade,
        stop_loss_pct=profile.stop_loss_par_trade,
        cost_bps=cost_bps,
    )

    # --- Étape 2.5 : Pénalité d'incertitude paramétrique (Medo & Pignatti 2013)
    f_star_bayes = _apply_bayesian_uncertainty_penalty(f_star, p_var, kappa=4.0)

    # --- Étape 3 : Fraction ajustée à l'horizon (Thorp 2008) ---
    kelly_fraction = profile.get_kelly_fraction()
    f_applied = f_star_bayes * kelly_fraction

    # Si un kelly_scale bayésien est fourni, on l'applique additionnellement
    if bayesian_consensus is not None:
        f_applied *= float(getattr(bayesian_consensus, "kelly_scale", 1.0))

    kelly_montant = round(profile.capital_total * f_applied, 2)
    odds_ratio = abs(profile.take_profit_par_trade) / abs(profile.stop_loss_par_trade)

    # --- Étape 4 : Fixed Fractional (Van Tharp 1998) ---
    ff_montant = _compute_fixed_fractional(
        profile.capital_total,
        profile.allocation_max_par_position,
        impact_strength,
        trust_trade_score,
    )

    # --- Étape 5 : Minimum conservateur (MacLean et al. 2010) ---
    montant_brut = min(kelly_montant, ff_montant)
    methode = "kelly_half" if kelly_montant <= ff_montant else "fixed_fractional"

    # --- Étape 6 : Contraintes du profil ---
    max_position = profile.get_max_position_euros()
    capped_by_profile = False
    capped_by_cash = False

    if montant_brut > max_position:
        montant_brut = max_position
        capped_by_profile = True
        methode = "capped_profile"

    if montant_brut > capital_dispo:
        montant_brut = capital_dispo
        capped_by_cash = True

    # --- Nombre d'actions ---
    if prix_actuel <= 0:
        nb_actions = 0.0
    else:
        nb_actions = math.floor(montant_brut / prix_actuel * 100) / 100  # 2 décimales

    montant_final = round(nb_actions * prix_actuel, 2)

    # --- Type d'action ---
    action_type = "RENFORCER" if ticker in portfolio.positions else "ACHETER"

    # --- Justification ---
    kelly_pct = round(f_applied * 100, 2)
    justification = (
        f"{action_type} {ticker} | {nb_actions} actions @ {prix_actuel:.2f}€ = {montant_final:.2f}€\n"
        f"  • p(gain) estimée: {p:.0%} (prior {profile.kelly_win_rate_prior:.0%} + signal {impact_strength:.2f})\n"
        f"  • Kelly f*: {f_star:.4f} × fraction {kelly_fraction} (horizon {profile.horizon.value}) = {kelly_pct:.1f}% du capital\n"
        f"  • Méthode: {methode} | Kelly: {kelly_montant:.0f}€ | Fixed-Frac: {ff_montant:.0f}€\n"
        f"  • Contraintes: cap profil={'OUI' if capped_by_profile else 'NON'} | "
        f"cash={'OUI' if capped_by_cash else 'NON'} | "
        f"concentration={'OUI' if capped_concentration else 'NON'}\n"
        f"  • Stop-loss: {profile.stop_loss_par_trade:.0%} ({montant_final * abs(profile.stop_loss_par_trade):.2f}€ max perte)\n"
        f"  • Take-profit: +{profile.take_profit_par_trade:.0%} ({montant_final * profile.take_profit_par_trade:.2f}€ objectif)"
    )

    logger.info("[PositionSizer] %s", justification.replace("\n", " | "))

    return PositionSizeResult(
        ticker=ticker,
        montant_euros=montant_final,
        nb_actions=nb_actions,
        action_type=action_type,
        kelly_f_star=f_star,
        kelly_fraction_applied=f_applied,
        kelly_montant=kelly_montant,
        kelly_win_prob=p,
        kelly_odds_ratio=odds_ratio,
        ff_montant=ff_montant,
        capped_by_profile=capped_by_profile,
        capped_by_cash=capped_by_cash,
        capped_by_concentration=capped_concentration,
        methode_finale=methode,
        justification=justification,
    )


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _make_hold_result(ticker: str, prix: float, raison: str) -> PositionSizeResult:
    return PositionSizeResult(
        ticker=ticker,
        montant_euros=0.0,
        nb_actions=0.0,
        action_type="TENIR",
        kelly_f_star=0.0,
        kelly_fraction_applied=0.0,
        kelly_montant=0.0,
        kelly_win_prob=0.0,
        kelly_odds_ratio=0.0,
        ff_montant=0.0,
        capped_by_profile=False,
        capped_by_cash=False,
        capped_by_concentration=False,
        methode_finale="hold",
        justification=f"TENIR {ticker} | {raison}",
    )


def _make_exit_result(ticker: str, prix: float, portfolio, profile, trust_score: float) -> PositionSizeResult:
    pos = portfolio.positions[ticker]
    raison = (
        f"Signal Vente confirmé (TrustTrade={trust_score:.2f}). "
        f"Clôturer {pos.nb_actions} actions @ {prix:.2f}€. "
        f"PnL estimé: {pos.pnl_pct:.1%} ({pos.pnl_absolu:+.2f}€)."
    )
    return PositionSizeResult(
        ticker=ticker,
        montant_euros=pos.valeur_actuelle,
        nb_actions=pos.nb_actions,
        action_type="COUPER",
        kelly_f_star=0.0,
        kelly_fraction_applied=0.0,
        kelly_montant=pos.valeur_actuelle,
        kelly_win_prob=trust_score,
        kelly_odds_ratio=0.0,
        ff_montant=pos.valeur_actuelle,
        capped_by_profile=False,
        capped_by_cash=False,
        capped_by_concentration=False,
        methode_finale="exit_signal",
        justification=f"COUPER {ticker} | {raison}",
    )
