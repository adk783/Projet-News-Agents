"""
evaluate_execution_costs.py — Couche 10 : Rentabilité nette après frais réels
=============================================================================
OBJECTIF SCIENTIFIQUE :
  Répondre à la question : "La stratégie reste-t-elle rentable après prise
  en compte des contraintes d'exécution réelles ?"

  Un signal peut être correct à 60% mais générer un alpha NÉGATIF si les
  frais de transaction dépassent le gain brut. Cette couche modélise
  l'ensemble des frictions de marché.

MODÈLE DE FRICTION (référence : Kissell & Glantz 2003, Almgren & Chriss 2001)
  Les frais d'exécution réels décomposés en 4 composantes :

  1. COMMISSIONS (Explicit Costs)
     Frais de courtage fixes par trade.
     Référence : Interactive Brokers (0.005$/action, min 1$), Binance (0.1%)
     -> modèle : max(MIN_COMMISSION, prix x n_actions x COMMISSION_PCT)

  2. SPREAD BID-ASK (Half-Spread Cost)
     On achète au Ask, on vend au Bid. Le spread est le coût immédiat.
     Modèle empirique : spread ~= 0.01% à 0.10% selon liquidité (Huang & Stoll 1997)
     -> proxy via volume normalisé (plus le volume est élevé, plus le spread est faible)

  3. SLIPPAGE / MARKET IMPACT (Price Impact)
     Pour des ordres larges, l'exécution déplace les cours.
     Modèle linéaire simplifié (Bertsimas & Lo 1998) :
     impact = sigma x sqrt(Q/V) où Q = taille ordre, V = volume journalier, sigma = volatilité
     -> Pour notre système, Q << V donc impact ~= 0.02% à 0.08%

  4. TIMING SLIPPAGE (Signal-to-Execution Delay)
     Le délai entre la publication de la news et l'exécution de l'ordre.
     Si l'article est publié à 10h00 et traité à 10h05, le cours a peut-être
     déjà bougé. Ce coût est modélisé séparément dans evaluate_latency.py.

MÉTRIQUES DE SORTIE :
  - Return brut (gross P&L) vs Return net (net P&L) par trade
  - Coût moyen par trade (absolu et en %)
  - Turnover annualisé (combien de fois on retourne le portefeuille)
  - Break-even accuracy : l'accuracy minimale pour rester positif net de frais
  - Comparaison des 3 régimes de courtage (retail / semi-pro / pro)

Lancé via : python eval/run_eval.py --layer 10 --sub execution_costs
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ExecutionCosts")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"

# ---------------------------------------------------------------------------
# Modèles de courtage — 3 niveaux de réalisme
# ---------------------------------------------------------------------------

# Chaque broker_profile définit les frais appliqués à chaque trade
BROKER_PROFILES = {
    "RETAIL": {
        "label": "Retail (ex: Degiro, Trading 212)",
        "commission_pct": 0.0010,  # 0.10% par trade
        "min_commission": 1.00,  # 1EUR minimum
        "half_spread_pct": 0.0005,  # 0.05% demi-spread moyen (action liquide)
        "market_impact_pct": 0.0003,  # 0.03% impact marché (petit compte)
        "slippage_pct": 0.0005,  # 0.05% slippage estimation
    },
    "SEMI_PRO": {
        "label": "Semi-Pro (ex: Interactive Brokers Lite)",
        "commission_pct": 0.0005,  # 0.05%
        "min_commission": 0.35,
        "half_spread_pct": 0.0003,
        "market_impact_pct": 0.0002,
        "slippage_pct": 0.0003,
    },
    "PRO": {
        "label": "Pro (ex: IB Pro / DMA direct access)",
        "commission_pct": 0.00015,  # 0.015% — coût institutionnel
        "min_commission": 0.10,
        "half_spread_pct": 0.0001,
        "market_impact_pct": 0.0001,
        "slippage_pct": 0.0001,
    },
}

# Taille d'ordre hypothétique pour le calcul (en euros)
DEFAULT_POSITION_SIZE_EUR = 5_000.0  # 5 000EUR par position

# Nombre de jours de bourse par an
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Calcul du coût d'exécution pour un trade
# ---------------------------------------------------------------------------


def compute_execution_cost(
    price: float,
    position_eur: float,
    broker: dict,
    signal: str,
) -> dict:
    """
    Modélise le coût total d'exécution d'un trade aller-retour.

    Un trade complet = entrée + sortie (deux legs).
    Chaque leg supporte : commission + demi-spread + slippage.

    Args:
        price         : Cours d'entrée (EUR)
        position_eur  : Montant investi (EUR)
        broker        : Profil de courtage (dict BROKER_PROFILES)
        signal        : "Achat" ou "Vente"

    Returns:
        dict avec le détail des frais et le coût total en %
    """
    if price <= 0 or position_eur <= 0 or signal == "Neutre":
        return {
            "commission_pct": 0.0,
            "spread_cost_pct": 0.0,
            "market_impact_pct": 0.0,
            "slippage_pct": 0.0,
            "total_roundtrip_pct": 0.0,
            "total_eur": 0.0,
        }

    n_actions = position_eur / price

    # --- Commission (aller + retour = 2 legs) ---
    commission_per_leg = max(broker["min_commission"], position_eur * broker["commission_pct"])
    commission_total_pct = (commission_per_leg * 2) / position_eur

    # --- Spread bid-ask (aller + retour) ---
    spread_total_pct = broker["half_spread_pct"] * 2

    # --- Market impact (aller + retour) ---
    impact_total_pct = broker["market_impact_pct"] * 2

    # --- Slippage (entrée uniquement, généralement) ---
    slippage_pct = broker["slippage_pct"]

    total_pct = commission_total_pct + spread_total_pct + impact_total_pct + slippage_pct
    total_eur = total_pct * position_eur

    return {
        "commission_pct": round(commission_total_pct * 100, 4),  # En %
        "spread_cost_pct": round(spread_total_pct * 100, 4),
        "market_impact_pct": round(impact_total_pct * 100, 4),
        "slippage_pct": round(slippage_pct * 100, 4),
        "total_roundtrip_pct": round(total_pct * 100, 4),
        "total_eur": round(total_eur, 2),
        "n_actions": round(n_actions, 4),
    }


# ---------------------------------------------------------------------------
# Calcul du Turnover annualisé
# ---------------------------------------------------------------------------


def compute_annualized_turnover(
    n_trades: int,
    avg_holding_days: float,
    portfolio_eur: float = DEFAULT_POSITION_SIZE_EUR * 10,
) -> dict:
    """
    Turnover = combien de fois le portefeuille est retourné par an.
    Un turnover élevé -> frais massifs même si chaque trade est rentable.

    Formule : Turnover = (n_trades x position_size) / (portfolio_value x horizon_ans)

    Args:
        n_trades          : Nombre de trades sur la période observée
        avg_holding_days  : Durée moyenne de détention d'une position
        portfolio_eur     : Valeur totale du portefeuille supposée

    Returns:
        dict avec turnover annualisé et fréquence de trading
    """
    if n_trades <= 0 or avg_holding_days <= 0:
        return {"turnover_annual": None, "trades_per_year": None}

    # Extrapolation à 1 an
    observation_days = n_trades * avg_holding_days
    if observation_days == 0:
        return {"turnover_annual": None, "trades_per_year": None}

    annualized_factor = TRADING_DAYS_PER_YEAR / observation_days
    n_trades_per_year = round(n_trades * annualized_factor, 1)

    # Valeur négociée par an
    capital_traded_per_year = n_trades_per_year * DEFAULT_POSITION_SIZE_EUR
    turnover_ratio = capital_traded_per_year / portfolio_eur if portfolio_eur > 0 else None

    return {
        "turnover_annual": round(turnover_ratio, 2) if turnover_ratio else None,
        "trades_per_year": n_trades_per_year,
        "avg_holding_days": round(avg_holding_days, 1),
        "capital_traded_yr_eur": round(capital_traded_per_year, 0),
    }


# ---------------------------------------------------------------------------
# Break-even accuracy
# ---------------------------------------------------------------------------


def compute_breakeven_accuracy(
    avg_win_pct: float,
    avg_loss_pct: float,
    total_cost_pct: float,
) -> dict:
    """
    Calcule l'accuracy minimale pour que la stratégie soit rentable NET de frais.

    Kelly / Break-even classique :
      E[P&L net] = acc x (win - cost) + (1-acc) x (-loss - cost) = 0
      -> acc_be = (loss + cost) / (win + loss)

    C'est le seuil critique : en dessous, le système détruit de la valeur même
    si les signaux ont une information edge.

    Args:
        avg_win_pct     : Gain moyen brut sur les trades gagnants (en %)
        avg_loss_pct    : Perte moyenne (abs) sur les trades perdants (en %)
        total_cost_pct  : Coût total aller-retour en % (commission + spread + impact)

    Returns:
        dict avec le break-even accuracy et l'interprétation
    """
    if avg_win_pct + avg_loss_pct == 0:
        return {"breakeven_accuracy": None, "profitable": None}

    # Net du coût
    net_win = avg_win_pct - total_cost_pct
    net_loss = avg_loss_pct + total_cost_pct  # La perte est augmentée des frais

    if net_win + net_loss == 0:
        return {"breakeven_accuracy": None, "profitable": None}

    acc_be = net_loss / (net_win + net_loss)

    return {
        "breakeven_accuracy": round(acc_be, 3),
        "net_win_per_trade": round(net_win, 4),
        "net_loss_per_trade": round(-net_loss, 4),
        "payoff_ratio_net": round(net_win / net_loss, 3) if net_loss > 0 else None,
    }


# ---------------------------------------------------------------------------
# Chargement des trades depuis L3
# ---------------------------------------------------------------------------


def _load_l3_trades() -> list[dict]:
    """Charge les trades depuis le dernier run de la Couche 3."""
    if not EVAL_RESULTS_DIR.exists():
        return []
    l3_runs = sorted([d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir() and "market_layer3" in d.name], reverse=True)
    if not l3_runs:
        return []
    trades_file = l3_runs[0] / "trades.json"
    if not trades_file.exists():
        return []
    with open(trades_file, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------------


def _analyse_net_performance(
    trades: list[dict],
    broker_key: str,
    position_eur: float = DEFAULT_POSITION_SIZE_EUR,
) -> dict:
    """
    Pour chaque trade, calcule le return NET des frais d'exécution.
    """
    broker = BROKER_PROFILES[broker_key]
    active = [
        t
        for t in trades
        if t.get("signal") != "Neutre" and t.get("return_pct") is not None and t.get("price_t0") is not None
    ]

    if not active:
        return {"broker": broker_key, "n_trades": 0, "error": "Pas de trades actifs"}

    net_returns = []
    gross_returns = []
    costs_detail = []

    for t in active:
        cost = compute_execution_cost(
            price=t["price_t0"],
            position_eur=position_eur,
            broker=broker,
            signal=t["signal"],
        )
        gross = t["return_pct"]
        net = round(gross - cost["total_roundtrip_pct"], 4)

        gross_returns.append(gross)
        net_returns.append(net)
        costs_detail.append(
            {
                "id": t.get("id", "?"),
                "signal": t["signal"],
                "gross_pct": gross,
                "cost_pct": cost["total_roundtrip_pct"],
                "net_pct": net,
            }
        )

    # --- Métriques agrégées ---
    n = len(net_returns)
    gross_arr = np.array(gross_returns)
    net_arr = np.array(net_returns)

    avg_gross = float(gross_arr.mean())
    avg_net = float(net_arr.mean())
    avg_cost = avg_gross - avg_net

    # Sharpe net
    def _sharpe(arr):
        if len(arr) < 2 or arr.std(ddof=1) == 0:
            return None
        return round(float(arr.mean() / arr.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)), 3)

    # Taux de gain brut vs net
    gross_win_rate = sum(1 for r in gross_returns if r > 0) / n
    net_win_rate = sum(1 for r in net_returns if r > 0) / n

    # Break-even
    wins = [r for r in gross_returns if r > 0]
    losses = [abs(r) for r in gross_returns if r < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    be = compute_breakeven_accuracy(avg_win, avg_loss, avg_cost)

    # Turnover (horizon moyen supposé = 20 jours ouvres)
    # On récupère les dates si disponibles pour calculer le vrai horizon
    turnover = compute_annualized_turnover(n_trades=n, avg_holding_days=20)

    return {
        "broker_key": broker_key,
        "broker_label": broker["label"],
        "n_trades": n,
        "gross": {
            "total_return": round(float(gross_arr.sum()), 2),
            "mean_return": round(avg_gross, 4),
            "win_rate": round(gross_win_rate, 3),
            "sharpe": _sharpe(gross_arr),
        },
        "net": {
            "total_return": round(float(net_arr.sum()), 2),
            "mean_return": round(avg_net, 4),
            "win_rate": round(net_win_rate, 3),
            "sharpe": _sharpe(net_arr),
        },
        "costs": {
            "avg_total_cost_pct": round(avg_cost, 4),
            "avg_commission_pct": round(broker["commission_pct"] * 200, 4),  # aller-retour
            "avg_spread_cost_pct": round(broker["half_spread_pct"] * 200, 4),
            "avg_market_impact_pct": round(broker["market_impact_pct"] * 200, 4),
            "avg_slippage_pct": round(broker["slippage_pct"] * 100, 4),
            "total_cost_eur": round(avg_cost / 100 * position_eur * n, 2),
        },
        "break_even": be,
        "turnover": turnover,
        "trade_details": costs_detail[:20],  # Les 20 premiers pour inspection
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------


def _print_report(result: dict) -> None:
    """Affiche les résultats pour un profil de courtage."""
    if result.get("error"):
        print(f"\n  [{result['broker_key']}] {result['error']}")
        return

    print(f"\n  +{'-' * 65}")
    print(f"  | PROFIL : {result['broker_label']}")
    print(f"  | Trades actifs : {result['n_trades']}")
    print(f"  +{'-' * 65}")

    g = result["gross"]
    n = result["net"]
    c = result["costs"]

    print(f"  | {'Métrique':<35} {'BRUT':>10} {'NET':>10}")
    print(f"  +{'-' * 65}")
    print(f"  | {'Return cumulé':<35} {g['total_return']:>+10.2f}% {n['total_return']:>+10.2f}%")
    print(f"  | {'Return moyen / trade':<35} {g['mean_return']:>+10.4f}% {n['mean_return']:>+10.4f}%")
    print(f"  | {'Taux de gain':<35} {g['win_rate']:>10.1%} {n['win_rate']:>10.1%}")
    print(f"  | {'Sharpe ratio annualisé':<35} {g['sharpe'] or 'N/A':>10} {n['sharpe'] or 'N/A':>10}")
    print(f"  +{'-' * 65}")
    print("  | DÉTAIL DES FRAIS (aller-retour par trade) :")
    print(f"  |   Commission         : {c['avg_commission_pct']:>+8.4f}%")
    print(f"  |   Spread bid-ask     : {c['avg_spread_cost_pct']:>+8.4f}%")
    print(f"  |   Market impact      : {c['avg_market_impact_pct']:>+8.4f}%")
    print(f"  |   Slippage           : {c['avg_slippage_pct']:>+8.4f}%")
    print("  |   -------------------------------------")
    print(f"  |   COÛT TOTAL         : {c['avg_total_cost_pct']:>+8.4f}%  (~{c['total_cost_eur']:.0f}EUR total)")

    be = result.get("break_even", {})
    be_acc = be.get("breakeven_accuracy")
    if be_acc is not None:
        print(f"  +{'-' * 65}")
        print("  | SEUIL DE RENTABILITÉ :")
        print(f"  |   Accuracy min. requise  : {be_acc:.1%}")
        print(f"  |   Payoff net (W/L)       : {be.get('payoff_ratio_net', 'N/A')}")
        if g["win_rate"] > be_acc:
            print(f"  |   [OK] La stratégie dépasse le seuil ({g['win_rate']:.1%} > {be_acc:.1%})")
        else:
            print(f"  |   [!!] La stratégie EST EN DESSOUS du seuil ({g['win_rate']:.1%} < {be_acc:.1%})")

    to = result.get("turnover", {})
    if to.get("turnover_annual") is not None:
        print(f"  +{'-' * 65}")
        print("  | TURNOVER ANNUALISÉ :")
        print(f"  |   Taux de rotation    : {to['turnover_annual']:.1f}x/an")
        print(f"  |   Trades / an (est.)  : {to['trades_per_year']:.0f}")
        print(f"  |   Durée moy. position : {to['avg_holding_days']:.0f}j")

    print(f"  +{'-' * 65}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_execution_costs_analysis(
    position_eur: float = DEFAULT_POSITION_SIZE_EUR,
) -> dict:
    """
    Couche 10 : Analyse de la rentabilité nette après frais d'exécution réels.
    Compare 3 profils de courtage : Retail, Semi-Pro, Pro.

    Returns:
        dict structuré avec les métriques brutes/nettes par profil.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 10 : Rentabilité nette après frais d'exécution (Market Frictions)")
    print(f"{'=' * 70}")
    print("\n  Modèle : Commission + Spread bid-ask + Market impact + Slippage")
    print(f"  Position hypothétique par trade : {position_eur:,.0f}EUR")
    print("  Référence : Kissell & Glantz (2003), Almgren & Chriss (2001)")

    trades = _load_l3_trades()
    if not trades:
        print("\n  [INFO] Aucun trade disponible. Lancez d'abord L3 :")
        print("  python eval/run_eval.py --layer 3")
        return {}

    active = [t for t in trades if t.get("signal") != "Neutre"]
    print(f"\n  {len(trades)} trades chargés ({len(active)} actifs Achat/Vente).")

    results = {}
    for broker_key in ["RETAIL", "SEMI_PRO", "PRO"]:
        r = _analyse_net_performance(trades, broker_key, position_eur)
        results[broker_key] = r
        _print_report(r)

    # --- Résumé comparatif ---
    print(f"\n  {'=' * 70}")
    print("  RÉSUMÉ COMPARATIF — RETURN NET PAR PROFIL DE COURTAGE")
    print(f"  {'=' * 70}")
    print(f"\n  {'Profil':<15} {'Return brut':>12} {'Coût total':>12} {'Return net':>12} {'Sharpe net':>12}")
    print(f"  {'-' * 63}")
    for k, r in results.items():
        if "error" not in r:
            g = r["gross"]
            n_ = r["net"]
            c = r["costs"]
            print(
                f"  {k:<15} "
                f"{g['total_return']:>+12.2f}% "
                f"{c['avg_total_cost_pct']:>+12.4f}% "
                f"{n_['total_return']:>+12.2f}% "
                f"{str(n_['sharpe'] or 'N/A'):>12}"
            )

    # --- Conclusion ---
    print(f"\n  {'=' * 70}")
    print("  CONCLUSIONS :")
    for k, r in results.items():
        if "error" in r:
            continue
        g = r["gross"]
        n_ = r["net"]
        be = r.get("break_even", {})
        be_acc = be.get("breakeven_accuracy")
        if n_["total_return"] > 0:
            status = "[OK] RENTABLE net de frais"
        elif g["total_return"] > 0:
            status = "[!] Rentable brut, NÉGATIF net (frais > alpha)"
        else:
            status = "[!!] Négatif brut et net"

        print(f"  [{k}] {status}")
        if be_acc:
            print(f"       -> Accuracy actuelle {g['win_rate']:.1%} vs seuil {be_acc:.1%}")

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = EVAL_RESULTS_DIR / f"{timestamp}_execution_costs_layer10"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sérialisation (supprimer les trade_details pour alléger)
    save_results = {}
    for k, r in results.items():
        sr = {kk: vv for kk, vv in r.items() if kk != "trade_details"}
        save_results[k] = sr

    with open(out_dir / "execution_costs.json", "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Résultats sauvegardés : {out_dir}/execution_costs.json")
    print("=" * 70)

    return {
        "layer": 10,
        "sub": "execution_costs",
        "n_trades_active": len(active),
        "results_by_broker": {
            k: {
                "gross_total": r.get("gross", {}).get("total_return"),
                "net_total": r.get("net", {}).get("total_return"),
                "avg_cost_pct": r.get("costs", {}).get("avg_total_cost_pct"),
                "net_sharpe": r.get("net", {}).get("sharpe"),
            }
            for k, r in results.items()
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--position_eur",
        type=float,
        default=DEFAULT_POSITION_SIZE_EUR,
        help=f"Taille de position par trade en EUR (défaut: {DEFAULT_POSITION_SIZE_EUR})",
    )
    args = parser.parse_args()
    run_execution_costs_analysis(position_eur=args.position_eur)
