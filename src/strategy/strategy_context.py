"""
strategy_context.py — Construction du Contexte Stratégique pour Injection LLM

Transforme le profil investisseur + état du portefeuille en un bloc de texte
structuré injecté dans les prompts du débat multi-agent.

Cela ancre les décisions des agents dans la réalité financière de l'investisseur,
passant d'une analyse générique ("AAPL est-elle bullish ?") à une décision
personnalisée ("Dois-JE, avec mon budget et ma stratégie, agir sur AAPL ?").

Références :
  [1] Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks." NeurIPS 2020.
      → L'injection de contexte structuré améliore la précision décisionnelle
      des LLMs en ancrant les réponses dans des faits externes vérifiables

  [2] Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and
      Chatbot Arena." NeurIPS 2023.
      → La qualité du contexte injecté détermine la qualité de la décision
      (contexte structuré > contexte verbeux non structuré)

  [3] Shefrin, H. & Statman, M. (1985). Disposition Effect.
      → Exposer le PnL actuel force les agents à traiter le biais de disposition
      (tendance à vendre trop tôt les gagnants, garder trop long les perdants)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------


def build_strategy_context(
    ticker: str,
    signal_candidat: Optional[str],  # Signal du filtre DistilRoBERTa (hint)
    profile,  # InvestorProfile
    portfolio,  # PortfolioState
    prix_actuel: float = 0.0,
    secteur: str = "Inconnu",
) -> str:
    """
    Construit le bloc de contexte stratégique à injecter dans le prompt du débat.

    Le bloc est conçu selon les principes RAG (Lewis et al. 2020) :
    - Structuré et concis (pas verbeux)
    - Contient uniquement les informations décisionnellement pertinentes
    - Formule la question de décision réelle à résoudre

    Args:
        ticker          : symbole boursier analysé
        signal_candidat : signal FinBERT préliminaire ("positive"/"negative")
        profile         : InvestorProfile chargé
        portfolio       : PortfolioState mis à jour
        prix_actuel     : prix du titre (€)
        secteur         : secteur du ticker

    Returns:
        Bloc texte formaté pour injection dans le prompt de débat.
    """
    from .investor_profile import InvestorProfile
    from .portfolio_state import PortfolioState

    assert isinstance(profile, InvestorProfile)
    assert isinstance(portfolio, PortfolioState)

    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║          CONTEXTE STRATÉGIQUE INVESTISSEUR (LOCUS)          ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")

    # --- Profil ---
    lines.append("\n▶ PROFIL STRATÉGIQUE")
    lines.append(f"  Stratégie      : {profile.nom_strategie}")
    lines.append(
        f"  Tolérance      : {profile.risk_tolerance.value.upper()} "
        f"(stop-loss {profile.stop_loss_par_trade:.0%} | take-profit +{profile.take_profit_par_trade:.0%})"
    )
    lines.append(f"  Horizon        : {profile.horizon.value.replace('_', ' ')}")
    lines.append(f"  Objectif       : {profile.objectif.value.replace('_', ' ')}")
    lines.append(f"  Capital total  : {profile.capital_total:,.0f}€")
    lines.append(
        f"  Max/position   : {profile.get_max_position_euros():,.0f}€ "
        f"({profile.allocation_max_par_position:.0%} du capital)"
    )
    lines.append(
        f"  Ratio R/R      : {profile.get_loss_aversion_ratio():.2f}x "
        f"(recommandé ≥ {2.25:.2f}x — Kahneman & Tversky 1979)"
    )

    # --- État du portefeuille ---
    total = portfolio.valeur_totale()
    pnl = portfolio.pnl_total()
    drawdown = portfolio.drawdown_actuel()
    cash_dispo = portfolio.cash_investissable(profile.cash_reserve_min)
    expo_secteur = portfolio.exposition_sectorielle()

    lines.append("\n▶ ÉTAT DU PORTEFEUILLE")
    lines.append(f"  Valeur totale  : {total:,.2f}€ | PnL: {pnl['pct']:+.1%} ({pnl['absolu']:+.2f}€)")
    lines.append(f"  Cash libre     : {cash_dispo:,.2f}€ (après réserve {profile.cash_reserve_min:.0%})")
    lines.append(f"  Positions      : {portfolio.nb_positions_ouvertes()} / {profile.max_positions_simultanees} max")

    # Alerte drawdown
    if drawdown < 0:
        alerte_dd = " ⚠ ALERTE" if drawdown < profile.max_drawdown_tolerance else ""
        lines.append(f"  Drawdown actuel: {drawdown:+.1%}{alerte_dd} (limite: {profile.max_drawdown_tolerance:.0%})")

    # Expositions sectorielles
    if expo_secteur:
        lines.append("  Expositions sectorielles:")
        for sect, pct in sorted(expo_secteur.items(), key=lambda x: -x[1]):
            alerte = " ⚠ ÉLEVÉ" if pct >= profile.secteur_max_exposition else ""
            lines.append(f"    • {sect}: {pct:.0%}{alerte}")

    # --- Position existante sur ce ticker ---
    lines.append(f"\n▶ POSITION ACTUELLE SUR {ticker}")
    if ticker in portfolio.positions:
        pos = portfolio.positions[ticker]
        pnl_pos = pos.pnl_pct
        pnl_abs = pos.pnl_absolu

        # Disposition Effect warning (Shefrin & Statman 1985)
        disposition_warning = ""
        if pnl_pos >= abs(profile.take_profit_par_trade):
            disposition_warning = (
                f"\n    ⚠ [Disposition Effect — Shefrin & Statman 1985] "
                f"PnL +{pnl_pos:.1%} ≥ take-profit. Risque de clôture prématurée."
            )
        elif pnl_pos <= profile.stop_loss_par_trade:
            disposition_warning = (
                f"\n    ⚠ [Stop-Loss] PnL {pnl_pos:.1%} ≤ stop-loss "
                f"({profile.stop_loss_par_trade:.0%}). Clôture disciplinée recommandée."
            )

        lines.append(f"  ↗ LONG {pos.nb_actions} actions @ {pos.prix_entree:.2f}€")
        if prix_actuel > 0:
            lines.append(f"  Prix actuel    : {prix_actuel:.2f}€")
        lines.append(f"  PnL position   : {pnl_pos:+.1%} ({pnl_abs:+.2f}€){disposition_warning}")
        lines.append(f"  Stop-loss à    : {pos.prix_entree * (1 + profile.stop_loss_par_trade):.2f}€")
        lines.append(f"  Take-profit à  : {pos.prix_entree * (1 + profile.take_profit_par_trade):.2f}€")
    else:
        lines.append(f"  Aucune position ouverte sur {ticker}.")
        if prix_actuel > 0:
            lines.append(f"  Prix actuel    : {prix_actuel:.2f}€")
            max_pos = profile.get_max_position_euros()
            nb_max = int(max_pos / prix_actuel) if prix_actuel > 0 else 0
            lines.append(f"  Max achetable  : ~{nb_max} actions ({max_pos:,.0f}€ max)")

    # Concentration sectorielle sur ce ticker
    expo_ticker_sect = expo_secteur.get(secteur, 0.0)
    if secteur != "Inconnu":
        alerte_sect = (
            f" ⚠ SATURÉ (limite {profile.secteur_max_exposition:.0%})"
            if expo_ticker_sect >= profile.secteur_max_exposition
            else ""
        )
        lines.append(f"  Secteur        : {secteur} (exposition actuelle: {expo_ticker_sect:.0%}){alerte_sect}")

    # --- Question de décision ---
    lines.append("\n▶ QUESTION DE DÉCISION POUR CE DÉBAT")

    if drawdown < profile.max_drawdown_tolerance:
        lines.append(
            "  ⛔ DRAWDOWN LIMITE ATTEINT — Aucun achat autorisé jusqu'à récupération."
            " Débattez uniquement de la gestion des positions existantes."
        )
    elif portfolio.nb_positions_ouvertes() >= profile.max_positions_simultanees:
        lines.append(
            f"  ⛔ PORTEFEUILLE PLEIN ({portfolio.nb_positions_ouvertes()} positions). "
            f"Évaluer si une position existante doit être clôturée pour libérer du capital."
        )
    elif expo_ticker_sect >= profile.secteur_max_exposition:
        lines.append(f"  ⛔ SECTEUR SATURÉ ({secteur}: {expo_ticker_sect:.0%}). Aucun renforcement sectoriel autorisé.")
    else:
        if ticker in portfolio.positions:
            lines.append(
                "  Compte tenu de la position existante et du PnL actuel, "
                "la décision à prendre est : RENFORCER / TENIR / RÉDUIRE / COUPER ?"
            )
        else:
            lines.append(
                f"  Compte tenu du budget disponible ({cash_dispo:,.0f}€) et du profil {profile.risk_tolerance.value}, "
                f"la décision à prendre est : ACHETER / NE PAS ACHETER ?"
            )
        lines.append(
            f"  Votre réponse doit spécifier l'action ET justifier au regard "
            f"du stop-loss ({profile.stop_loss_par_trade:.0%}) et du take-profit (+{profile.take_profit_par_trade:.0%})."
        )

    lines.append("═" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Résumé court pour le Consensus Agent
# ---------------------------------------------------------------------------


def build_strategy_summary_for_consensus(
    ticker: str,
    sizing_result,  # PositionSizeResult
    profile,  # InvestorProfile
    portfolio,  # PortfolioState
) -> str:
    """
    Version condensée du contexte pour l'agent Consensus.
    Contient uniquement la recommandation sizing et les contraintes actives.

    Ref: Zheng et al. (2023) — le contexte pour l'arbitre doit être plus
    concis que pour les débatteurs pour éviter la dilution d'attention.
    """
    from .investor_profile import InvestorProfile
    from .portfolio_state import PortfolioState

    lines = [
        "\n=== RECOMMANDATION SIZING (Kelly/Half-Kelly — Thorp 2008) ===",
        f"Action suggérée : {sizing_result.action_type} {ticker}",
        f"Montant         : {sizing_result.montant_euros:.2f}€ ({sizing_result.nb_actions} actions)",
        f"Méthode         : {sizing_result.methode_finale}",
        f"Win prob estimée: {sizing_result.kelly_win_prob:.0%}",
        f"Kelly f*        : {sizing_result.kelly_f_star:.4f} → fraction appliquée: {sizing_result.kelly_fraction_applied:.4f}",
        f"Stop-loss       : {profile.stop_loss_par_trade:.0%} | Take-profit: +{profile.take_profit_par_trade:.0%}",
        f"Drawdown portif : {portfolio.drawdown_actuel():+.1%} (limite: {profile.max_drawdown_tolerance:.0%})",
        "=" * 52,
    ]
    return "\n".join(lines)
