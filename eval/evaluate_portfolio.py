"""
evaluate_portfolio.py -- Métriques de gestion de portefeuille
==============================================================
OBJECTIF :
  Mesurer si la stratégie pilotée par les signaux de l'architecture
  reste rentable sur la durée SANS prendre un risque excessif.

  Un système peut avoir 60% de précision mais être désastreux si ses
  10 erreurs sont des pertes de -15% chacune et ses 15 succès +1% chacun.
  Ces métriques capturent cette asymétrie.

MÉTRIQUES IMPLÉMENTÉES :

  1. RATIO DE SHARPE
     (Return moyen - Taux sans risque) / Ecart-type des returns
     Annualisé avec √252 (252 jours de bourse / an)
     → Mesure le rendement PAR UNITÉ de risque total
     → > 1.0 = bon, > 2.0 = excellent
     → Pénalise la volatilité dans les deux sens (hausse ET baisse)

  2. RATIO DE SORTINO
     (Return moyen - Return cible) / Écart-type des returns NÉGATIFS seuls
     → Ne pénalise que les pertes (downside deviation), pas les gains excessifs
     → Plus approprié pour notre cas : on accepte les fortes hausses
     → > 1.0 = bon, > 2.0 = excellent

  3. RATIO DE RACHEV (CVaR Ratio)
     E[Return | Return > VaR(1-β)] / |E[Return | Return < -VaR(α)]|
     → Rapport entre la moyenne des tops α% de gains
        et la moyenne des tops α% de pertes (CVaR)
     → > 1.0 = les gains extrêmes compensent les pertes extrêmes
     → Nécessite au moins 20 trades pour être fiable

  4. MAXIMUM DRAWDOWN (Max DD)
     Max sur toute la période de (Pic cumulé - Creux cumulé) / Pic
     → Pire perte contiguë depuis un sommet
     → Un Max DD > 20% est généralement inacceptable pour un portefeuille

  5. CALMAR RATIO
     Return annualisé / |Max Drawdown|
     → Combine performance et résistance aux crashes

Lancé via : python eval/run_eval.py --layer 5 --sub portfolio
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Portfolio")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"
RISK_FREE_RATE = 0.0  # Taux sans risque (simplifié à 0%)
TARGET_RETURN = 0.0  # Return cible pour Sortino (0% = "ne pas perdre")
RACHEV_ALPHA = 0.10  # Top 10% des gains pour le Rachev
RACHEV_BETA = 0.10  # Top 10% des pertes pour le Rachev
MIN_RACHEV_N = 20  # Minimum de trades pour Rachev fiable


# ---------------------------------------------------------------------------
# Métriques individuelles
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: list[float], risk_free: float = RISK_FREE_RATE, annualize: int = 252) -> Optional[float]:
    """Ratio de Sharpe annualisé."""
    if len(returns) < 2:
        return None
    arr = np.array(returns)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return None
    # Annualisation : on suppose une décision par jour ouvré
    # Si on a peu de trades, on annualise quand même pour comparabilité
    return round(float((mean - risk_free) / std * math.sqrt(annualize)), 3)


def sortino_ratio(returns: list[float], target: float = TARGET_RETURN, annualize: int = 252) -> Optional[float]:
    """Ratio de Sortino annualisé (downside deviation uniquement)."""
    if len(returns) < 2:
        return None
    arr = np.array(returns)
    mean = arr.mean()
    # Downside deviation : seulement les returns < target
    downside = arr[arr < target]
    if len(downside) == 0:
        return float("inf")  # Aucune perte → Sortino infini
    dd = math.sqrt(np.mean((downside - target) ** 2))
    if dd == 0:
        return None
    return round(float((mean - target) / dd * math.sqrt(annualize)), 3)


def rachev_ratio(returns: list[float], alpha: float = RACHEV_ALPHA, beta: float = RACHEV_BETA) -> Optional[float]:
    """
    Ratio de Rachev : espérance des top α% gains / espérance des top β% pertes.
    Retourne None si l'échantillon est trop petit.
    """
    if len(returns) < MIN_RACHEV_N:
        return None
    arr = np.array(returns)
    # Top α gains (valeurs au-dessus du (1-α) quantile)
    up_threshold = np.quantile(arr, 1 - alpha)
    top_gains = arr[arr >= up_threshold]
    # Top β worst losses (valeurs en dessous du β quantile)
    down_threshold = np.quantile(arr, beta)
    worst_losses = arr[arr <= down_threshold]

    if len(top_gains) == 0 or len(worst_losses) == 0:
        return None
    mean_gain = float(top_gains.mean())
    mean_loss = float(worst_losses.mean())

    if mean_loss == 0:
        return None
    return round(mean_gain / abs(mean_loss), 3)


def max_drawdown(returns: list[float]) -> dict:
    """
    Calcule le Maximum Drawdown sur la séquence de returns.
    Retourne le Max DD, la position du pic et du creux.
    """
    if not returns:
        return {"max_dd": None, "peak_idx": None, "trough_idx": None}

    cumulative = []
    cum = 0.0
    for r in returns:
        cum += r
        cumulative.append(cum)

    arr = np.array(cumulative)
    peak = arr[0]
    max_dd = 0.0
    peak_idx = 0
    trough_idx = 0
    best_peak_idx = 0

    for i, val in enumerate(arr):
        if val > peak:
            peak = val
            best_peak_idx = i
        dd = (peak - val) / (1 + peak / 100)  # En % relatif
        if dd > max_dd:
            max_dd = dd
            peak_idx = best_peak_idx
            trough_idx = i

    return {
        "max_dd": round(max_dd * 100, 2),  # En %
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "cum_return": round(float(arr[-1]), 2),
    }


def calmar_ratio(returns: list[float], annualize: int = 252) -> Optional[float]:
    """Return annualisé / |Max Drawdown|."""
    dd = max_drawdown(returns)
    if dd["max_dd"] is None or dd["max_dd"] == 0:
        return None
    arr = np.array(returns)
    ann_return = float(arr.mean() * annualize)
    return round(ann_return / dd["max_dd"], 3)


def value_at_risk(returns: list[float], alpha: float = 0.05) -> Optional[float]:
    """VaR paramétrique au niveau alpha (5% par défaut)."""
    if len(returns) < 5:
        return None
    arr = np.array(returns)
    return round(float(np.quantile(arr, alpha)), 4)


def expected_shortfall(returns: list[float], alpha: float = 0.05) -> Optional[float]:
    """CVaR / Expected Shortfall : moyenne des pertes au-delà du VaR."""
    if len(returns) < 5:
        return None
    arr = np.array(returns)
    var = np.quantile(arr, alpha)
    tail = arr[arr <= var]
    return round(float(tail.mean()), 4) if len(tail) > 0 else None


# ---------------------------------------------------------------------------
# Analyse d'un portefeuille complet
# ---------------------------------------------------------------------------


def analyse_portfolio(returns: list[float], label: str = "Portefeuille") -> dict:
    """
    Calcule toutes les métriques pour un portefeuille de returns (en %).

    Args:
        returns : Liste des returns individuels en % (ex: [+2.3, -1.1, +5.0, ...])
        label   : Nom du portefeuille pour l'affichage

    Returns:
        Dict complet avec toutes les métriques.
    """
    if not returns:
        return {"label": label, "error": "Aucun trade"}

    arr = np.array(returns)
    dd = max_drawdown(returns)

    metrics = {
        "label": label,
        "n_trades": len(returns),
        "total_return": round(float(arr.sum()), 2),
        "mean_return": round(float(arr.mean()), 4),
        "std_return": round(float(arr.std(ddof=1 if len(arr) > 1 else 0)), 4),
        "win_rate": round(sum(1 for r in returns if r > 0) / len(returns), 3),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "rachev": rachev_ratio(returns),
        "max_drawdown": dd["max_dd"],
        "calmar": calmar_ratio(returns),
        "var_5pct": value_at_risk(returns),
        "cvar_5pct": expected_shortfall(returns),
        "best_trade": round(float(arr.max()), 2),
        "worst_trade": round(float(arr.min()), 2),
    }
    return metrics


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


def _ascii_bar(value: float, max_val: float = 20.0, width: int = 30, positive: bool = True) -> str:
    """Génère une barre ASCII proportionnelle."""
    ratio = min(abs(value) / max_val, 1.0)
    n_chars = int(ratio * width)
    char = "+" if (positive and value >= 0) or (not positive) else "-"
    return char * n_chars + "." * (width - n_chars)


def _print_metrics_table(metrics: dict) -> None:
    """Affiche les métriques de façon lisible."""

    def fmt(v, suffix=""):
        if v is None:
            return "N/A (insuffisant)"
        if isinstance(v, float):
            if abs(v) < 10:
                return f"{v:+.3f}{suffix}"
            return f"{v:+.1f}{suffix}"
        return str(v)

    print(f"\n  {metrics['label']}")
    print(f"  {'─' * 55}")
    print(f"  Trades                    : {metrics['n_trades']}")
    print(f"  Return cumule             : {fmt(metrics['total_return'], '%')}")
    print(f"  Return moyen / trade      : {fmt(metrics['mean_return'], '%')}")
    print(f"  Volatilite (std)          : {fmt(metrics['std_return'], '%')}")
    print(f"  Taux de gain              : {metrics['win_rate']:.0%}")
    print(f"  Meilleur trade            : {fmt(metrics['best_trade'], '%')}")
    print(f"  Pire trade                : {fmt(metrics['worst_trade'], '%')}")
    print("  ─── Métriques de risque ────────────────────────────────")
    print(f"  Ratio de Sharpe           : {fmt(metrics['sharpe'])}")
    print(f"  Ratio de Sortino          : {fmt(metrics['sortino'])}")
    print(
        f"  Ratio de Rachev           : {fmt(metrics['rachev'])} "
        f"{'(min 20 trades requis)' if metrics['rachev'] is None else ''}"
    )
    print(f"  Maximum Drawdown          : {fmt(metrics['max_drawdown'], '%')}")
    print(f"  Ratio de Calmar           : {fmt(metrics['calmar'])}")
    print(f"  VaR 5%                    : {fmt(metrics['var_5pct'], '%')}")
    print(f"  CVaR 5% (Expected Shortfall): {fmt(metrics['cvar_5pct'], '%')}")

    # Verdict
    sharpe = metrics.get("sharpe")
    max_dd = metrics.get("max_drawdown")
    verdict = ""
    if sharpe is not None and max_dd is not None:
        if sharpe > 1.5 and max_dd < 10:
            verdict = "EXCELLENT : Rendement eleve, risque faible."
        elif sharpe > 1.0 and max_dd < 20:
            verdict = "BON : Strategy viable. Continuer a accumuler des donnees."
        elif sharpe > 0:
            verdict = "PASSABLE : La strategy est positive mais le risque est eleve."
        else:
            verdict = "INSUFFISANT : Return/risque negatif. Revoir le pipeline."
    if verdict:
        print(f"\n  Verdict : {verdict}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_portfolio_analysis() -> dict:
    """
    Lance l'analyse de portefeuille complète depuis les trades de L3.
    Compare Portefeuille A (tous signaux) vs B (FAIBLE uniquement).
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 5c : Gestion de Portefeuille (Sharpe, Sortino, Rachev, MaxDD)")
    print(f"{'=' * 70}")

    trades = _load_l3_trades()
    if not trades:
        print("\n  [INFO] Aucun trade disponible. Lancez L2 puis L3:")
        print("  python eval/run_eval.py --layer 2 --limit 10")
        print("  python eval/run_eval.py --layer 3")
        return {}

    print(f"\n  {len(trades)} trades charges depuis la Couche 3.\n")

    # Portefeuille A : Tous les signaux Achat/Vente
    all_returns = [t["return_pct"] for t in trades if t.get("signal") != "Neutre" and t.get("return_pct") is not None]

    # Portefeuille B : Uniquement FAIBLE-risque
    faible_returns = [
        t["return_pct"]
        for t in trades
        if t.get("signal") != "Neutre" and t.get("risk_level") == "FAIBLE" and t.get("return_pct") is not None
    ]

    # Portefeuille M : Buy & Hold du marché (benchmark passif)
    # On calcule le return moyen du marché sur la même période (estimé)
    market_returns_sim = [0.04] * len(all_returns)  # Proxy S&P500 ~10% annuel / 252j ≈ 0.04%/j

    portf_all = analyse_portfolio(all_returns, "Portefeuille A — Suivre tous les signaux")
    portf_faible = analyse_portfolio(faible_returns, "Portefeuille B — Signaux FAIBLE-risque uniquement (YOLO)")
    portf_market = analyse_portfolio(market_returns_sim, "Portefeuille M — Buy & Hold marche (baseline passif)")

    _print_metrics_table(portf_all)
    _print_metrics_table(portf_faible)
    _print_metrics_table(portf_market)

    # Comparaison A vs B
    print(f"\n  {'=' * 55}")
    print("  COMPARAISON YOLO (B vs A) :")
    for key, label in [
        ("total_return", "Return cumule"),
        ("sharpe", "Sharpe ratio"),
        ("sortino", "Sortino ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Taux de gain"),
    ]:
        a = portf_all.get(key)
        b = portf_faible.get(key)
        if a is None or b is None:
            continue
        delta = b - a if isinstance(b, (int, float)) else None
        sign = "+" if delta and delta > 0 else ""
        print(f"  {label:<25} : A={a} | B={b} | delta={sign}{delta}")

    print(f"\n  {'=' * 55}")
    if portf_faible.get("sharpe") and portf_all.get("sharpe"):
        if portf_faible["sharpe"] > portf_all["sharpe"]:
            print("  Le YOLO classifier AMELIORE le Sharpe ratio.")
            print("  Les signaux FAIBLE sont plus rentables par unite de risque.")
        else:
            print("  Le YOLO classifier EST TROP CONSERVATEUR.")
            print("  Il ecarte des trades rentables. Ajuster les seuils.")

    print("=" * 70)

    return {
        "sub": "portfolio",
        "n_trades": len(trades),
        "portfolio_all": portf_all,
        "portfolio_faible": portf_faible,
    }


if __name__ == "__main__":
    run_portfolio_analysis()
