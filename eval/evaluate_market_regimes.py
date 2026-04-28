"""
evaluate_market_regimes.py — Couche 12 : Robustesse aux régimes de marché
==========================================================================
OBJECTIF SCIENTIFIQUE :
  Un système de trading doit rester performant dans TOUS les environnements
  de marché, pas seulement en tendance haussière.

  La question centrale est : "Mon architecture se comporte-t-elle différemment
  selon le régime de marché ? Est-elle robuste ?"

DÉFINITION DES RÉGIMES DE MARCHÉ :
  Nous définissons 4 régimes sur la base du comportement du SPY (S&P 500) :

  +-----------------?-------------------------------------------------------+
  | Régime          | Définition                                            |
  +-----------------+-------------------------------------------------------?
  | BULL            | SPY return sur 20j > +3% ET VIX < 20                 |
  | BEAR            | SPY return sur 20j < -3% ET VIX > 25                 |
  | HIGH_VOLATILITY | VIX > 30 (peu importe la direction)                  |
  | SIDEWAYS        | -3% <= SPY 20j <= +3% ET VIX < 25                    |
  +-----------------?-------------------------------------------------------+

CLASSIFICATION DU RÉGIME :
  Pour chaque trade, on récupère :
    1. Le VIX au moment de la décision (macro_snap.vix stocké en DB)
    2. Le return SPY sur les 20 jours précédant le trade
    3. On classifie le régime selon les seuils ci-dessus

MÉTRIQUES PAR RÉGIME :
  Pour chaque régime :
    - Sharpe ratio
    - Sortino ratio
    - Maximum Drawdown
    - Accuracy directionnelle
    - Alpha vs SPY buy-and-hold
    - Taux de signaux FAIBLE vs ELEVE (comportement du YOLO classifier)

ROBUSTESSE ANALYSIS :
  - Ratio de Sharpe normalisé : Sharpe_bull / Sharpe_bear > 1 = biais haussier
  - Régime le plus difficile (Sharpe minimal)
  - Stabilité du signal : variance de l'accuracy inter-régimes

RÉFÉRENCES :
  - Hamilton (1989) — Markov Switching Models
  - Lo (2004) — The Adaptive Markets Hypothesis
  - Ang & Bekaert (2002) — International Asset Allocation with Regime Shifts

Lancé via : python eval/run_eval.py --layer 12 --sub market_regimes
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from dotenv import load_dotenv

load_dotenv()

try:
    import yfinance as yf

    HAS_YF = True
except ImportError:
    HAS_YF = False

logger = logging.getLogger("MarketRegimes")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

DATABASE_PATH = "data/news_database.db"
EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"

# ---------------------------------------------------------------------------
# Seuils de classification des régimes
# ---------------------------------------------------------------------------

REGIME_THRESHOLDS = {
    "spy_bull_threshold": +3.0,  # SPY return 20j > +3% -> tendance haussière
    "spy_bear_threshold": -3.0,  # SPY return 20j < -3% -> tendance baissière
    "vix_low": 20.0,  # VIX < 20 -> faible volatilité
    "vix_medium": 25.0,  # VIX 20-25 -> volatilité modérée
    "vix_high": 30.0,  # VIX > 30 -> très forte volatilité
}

REGIME_LABELS = {
    "BULL": "Marché Haussier (Bull)",
    "BEAR": "Marché Baissier (Bear)",
    "HIGH_VOL": "Haute Volatilité (VIX > 30)",
    "SIDEWAYS": "Marché Lateral (Sideways)",
    "UNKNOWN": "Régime Inconnu",
}

REGIME_COLORS_ASCII = {
    "BULL": "^",
    "BEAR": "v",
    "HIGH_VOL": "[!]",
    "SIDEWAYS": "->",
    "UNKNOWN": "?",
}

# Cache des returns SPY pour éviter les appels répétés
_spy_cache: dict = {}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _get_spy_return_20d(date_str: str) -> Optional[float]:
    """
    Retourne le return SPY sur les 20 jours précédant date_str.
    Utilisé pour classifier le régime au moment du trade.
    """
    if not HAS_YF:
        return None

    cache_key = f"spy_20d_{date_str}"
    if cache_key in _spy_cache:
        return _spy_cache[cache_key]

    try:
        base = datetime.strptime(date_str[:10], "%Y-%m-%d")
        start = base - timedelta(days=30)  # Marge pour jours non-ouvrés
        end = base + timedelta(days=2)

        hist = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if len(hist) < 15:
            _spy_cache[cache_key] = None
            return None

        # Les 20 derniers jours ouvrés disponibles avant la date
        last_20 = hist["Close"].iloc[-20:] if len(hist) >= 20 else hist["Close"]
        ret = (float(last_20.iloc[-1]) - float(last_20.iloc[0])) / float(last_20.iloc[0]) * 100
        result = round(ret, 2)
        _spy_cache[cache_key] = result
        return result
    except Exception as e:
        logger.debug("SPY 20d return for %s: %s", date_str, e)
        _spy_cache[cache_key] = None
        return None


def _classify_regime(
    vix: Optional[float],
    spy_return_20d: Optional[float],
) -> str:
    """
    Classifie le régime de marché selon VIX et SPY return 20j.

    Priorité :
    1. HIGH_VOL si VIX > 30 (peu importe la direction)
    2. BULL si SPY > +3% et VIX < 25
    3. BEAR si SPY < -3% et VIX > 20
    4. SIDEWAYS sinon

    Returns:
        str : "BULL" | "BEAR" | "HIGH_VOL" | "SIDEWAYS" | "UNKNOWN"
    """
    if vix is None and spy_return_20d is None:
        return "UNKNOWN"

    t = REGIME_THRESHOLDS

    if vix is not None and vix > t["vix_high"]:
        return "HIGH_VOL"

    if spy_return_20d is not None:
        if spy_return_20d > t["spy_bull_threshold"] and (vix is None or vix < t["vix_medium"]):
            return "BULL"
        if spy_return_20d < t["spy_bear_threshold"] and (vix is None or vix > t["vix_low"]):
            return "BEAR"

    return "SIDEWAYS"


# ---------------------------------------------------------------------------
# Métriques par régime
# ---------------------------------------------------------------------------


def _compute_regime_metrics(trades: list[dict]) -> dict:
    """
    Calcule Sharpe, Sortino, MaxDD, accuracy et alpha pour un groupe de trades.
    """
    if not trades:
        return {"n": 0, "error": "Aucun trade"}

    active = [t for t in trades if t.get("signal") != "Neutre" and t.get("return_pct") is not None]

    if not active:
        return {"n": len(trades), "n_active": 0, "error": "Aucun trade actif"}

    returns = np.array([t["return_pct"] for t in active])
    n = len(returns)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1)) if n > 1 else 0.0

    # Sharpe (annualisé)
    sharpe = round(mean_ret / std_ret * math.sqrt(252), 3) if std_ret > 0 else None

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0:
        dd_std = math.sqrt(float(np.mean(downside**2)))
        sortino = round(mean_ret / dd_std * math.sqrt(252), 3) if dd_std > 0 else None
    else:
        sortino = float("inf")

    # Maximum Drawdown
    cumulative = np.cumsum(returns)
    peak = cumulative[0]
    max_dd = 0.0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = (peak - val) / (1 + peak / 100) if (1 + peak / 100) > 0 else 0
        max_dd = max(max_dd, dd)

    # Accuracy directionnelle
    signal_correct_count = sum(1 for t in active if t.get("signal_correct") is True)
    actionable_count = sum(1 for t in active if t.get("actionable") is True)
    accuracy = round(signal_correct_count / actionable_count, 3) if actionable_count > 0 else None

    # Alpha vs SPY
    spy_rets = [t.get("spy_return") for t in active if t.get("spy_return") is not None]
    alpha = round(mean_ret - float(np.mean(spy_rets)), 4) if spy_rets else None

    # Distribution des niveaux de risque YOLO
    risk_dist = defaultdict(int)
    for t in active:
        risk_dist[t.get("risk_level", "INCONNU")] += 1

    win_rate = round(sum(1 for r in returns if r > 0) / n, 3)

    return {
        "n": len(trades),
        "n_active": n,
        "mean_return": round(mean_ret, 4),
        "total_return": round(float(returns.sum()), 2),
        "win_rate": win_rate,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": round(max_dd * 100, 2),
        "accuracy": accuracy,
        "alpha_vs_spy": alpha,
        "risk_distribution": dict(risk_dist),
    }


# ---------------------------------------------------------------------------
# Chargement et classification des trades
# ---------------------------------------------------------------------------


def _load_and_classify_trades() -> dict[str, list[dict]]:
    """
    Charge les trades depuis L3, récupère les données de régime
    (VIX depuis SQLite + SPY 20j depuis yfinance), et classifie chaque trade.

    Returns:
        dict : régime -> liste de trades classifiés dans ce régime
    """
    # 1. Charge les trades L3
    l3_trades = _load_l3_trades()
    if not l3_trades:
        return {}

    # 2. Charge les VIX depuis SQLite (stocké par le pipeline)
    vix_by_id = _load_vix_from_db()

    # 3. Classification
    by_regime: dict[str, list[dict]] = defaultdict(list)

    for t in l3_trades:
        date_str = t.get("date_utc", "")
        ticker = t.get("ticker", "")

        # VIX au moment de la décision (depuis DB si disponible)
        vix = vix_by_id.get(f"{ticker}_{date_str[:10]}")

        # SPY return 20j (récupéré via yfinance)
        spy_20d = None
        if date_str:
            spy_20d = _get_spy_return_20d(date_str)
            if spy_20d is not None:
                logger.debug("SPY 20d @ %s = %.2f%%", date_str[:10], spy_20d)

        regime = _classify_regime(vix, spy_20d)

        t_enriched = {
            **t,
            "regime": regime,
            "vix_at_trade": vix,
            "spy_20d_ret": spy_20d,
        }
        by_regime[regime].append(t_enriched)

    return dict(by_regime)


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


def _load_vix_from_db() -> dict[str, Optional[float]]:
    """
    Récupère le VIX stocké à chaque décision depuis SQLite.
    Retourne un dict : 'TICKER_DATE' -> vix_value
    """
    vix_map = {}
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker, date_utc, vix_at_decision
            FROM articles
            WHERE vix_at_decision IS NOT NULL
              AND date_utc IS NOT NULL
        """)
        for row in cur.fetchall():
            key = f"{row['ticker']}_{row['date_utc'][:10]}"
            vix_map[key] = float(row["vix_at_decision"])
        conn.close()
    except Exception as e:
        logger.debug("VIX from DB: %s", e)
    return vix_map


# ---------------------------------------------------------------------------
# Analyse de robustesse inter-régimes
# ---------------------------------------------------------------------------


def _compute_robustness_metrics(regime_results: dict[str, dict]) -> dict:
    """
    Calcule les métriques de robustesse inter-régimes.
    Un système vraiment robuste a un Sharpe ratio relativement stable
    entre les régimes.

    Returns:
        dict avec métriques de stabilité et d'instabilité
    """
    sharpes = {k: v.get("sharpe") for k, v in regime_results.items() if v.get("sharpe") is not None and k != "UNKNOWN"}

    if len(sharpes) < 2:
        return {"error": "Insuffisant (< 2 régimes avec données)"}

    sharpe_values = list(sharpes.values())

    best_regime = max(sharpes, key=sharpes.get)
    worst_regime = min(sharpes, key=sharpes.get)
    sharpe_range = round(max(sharpe_values) - min(sharpe_values), 3)
    sharpe_std = round(float(np.std(sharpe_values)), 3)

    # Stabilité = 1 - (std/mean) si mean > 0 (coefficient de variation normé)
    sharpe_mean = float(np.mean(sharpe_values))
    cv = round(sharpe_std / abs(sharpe_mean), 3) if sharpe_mean != 0 else None

    # Bull/Bear ratio
    bull_sharpe = sharpes.get("BULL")
    bear_sharpe = sharpes.get("BEAR")
    bull_bear_ratio = None
    if bull_sharpe is not None and bear_sharpe is not None and bear_sharpe != 0:
        bull_bear_ratio = round(bull_sharpe / bear_sharpe, 2)

    return {
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "sharpe_range": sharpe_range,
        "sharpe_std": sharpe_std,
        "sharpe_mean": round(sharpe_mean, 3),
        "sharpe_cv": cv,
        "bull_bear_ratio": bull_bear_ratio,
        "sharpes": sharpes,
        "verdict": (
            "TRÈS ROBUSTE"
            if sharpe_std < 0.3
            else "ROBUSTE"
            if sharpe_std < 0.6
            else "PARTIELLEMENT"
            if sharpe_std < 1.0
            else "FRAGILE (biais de régime)"
        ),
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------


def _print_regime_report(
    by_regime: dict[str, list[dict]],
    regime_metrics: dict[str, dict],
    robustness: dict,
) -> None:
    """Affiche le rapport complet par régime."""

    print(f"\n  {'-' * 70}")
    print("  [A] DISTRIBUTION DES TRADES PAR RÉGIME")
    print(f"  {'-' * 70}")

    total = sum(len(v) for v in by_regime.values())
    for reg, trades in sorted(by_regime.items(), key=lambda x: -len(x[1])):
        icon = REGIME_COLORS_ASCII.get(reg, "?")
        label = REGIME_LABELS.get(reg, reg)
        pct = len(trades) / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {icon} {label:<30} {len(trades):>5} trades ({pct:.0f}%) [{bar}]")

    if total == 0:
        print("\n  [INFO] Aucun trade disponible pour l'analyse de régimes.")
        return

    print(f"\n  {'-' * 70}")
    print("  [B] MÉTRIQUES PAR RÉGIME DE MARCHÉ")
    print(f"  {'-' * 70}")

    print(
        f"\n  {'Régime':<20} {'N':>4} {'Ret moy':>9} {'Sharpe':>8} {'Sortino':>9} "
        f"{'MaxDD':>8} {'Accuracy':>10} {'Alpha':>8}"
    )
    print(f"  {'-' * 82}")

    for reg in ["BULL", "BEAR", "HIGH_VOL", "SIDEWAYS", "UNKNOWN"]:
        m = regime_metrics.get(reg, {})
        if m.get("n_active", 0) == 0:
            icon = REGIME_COLORS_ASCII.get(reg, "?")
            print(f"  {icon} {reg:<18} {'N/A':>4}")
            continue

        icon = REGIME_COLORS_ASCII.get(reg, "?")
        sharpe = f"{m['sharpe']:>+8.3f}" if m.get("sharpe") is not None else "     N/A"
        sortino = (
            f"{m['sortino']:>+9.3f}"
            if m.get("sortino") is not None and m["sortino"] != float("inf")
            else "      inf"
            if m.get("sortino") == float("inf")
            else "      N/A"
        )
        maxdd = f"{m['max_drawdown']:>+8.2f}%" if m.get("max_drawdown") is not None else "     N/A"
        acc = f"{m['accuracy']:>10.1%}" if m.get("accuracy") is not None else "       N/A"
        alpha = f"{m['alpha_vs_spy']:>+8.3f}%" if m.get("alpha_vs_spy") is not None else "     N/A"

        print(
            f"  {icon} {reg:<18} {m['n_active']:>4} {m['mean_return']:>+9.4f}% {sharpe} {sortino} {maxdd} {acc} {alpha}"
        )

    # Distribution YOLO par régime
    print(f"\n  {'-' * 70}")
    print("  [C] COMPORTEMENT DU YOLO CLASSIFIER PAR RÉGIME")
    print(f"  {'-' * 70}")
    print(f"\n  {'Régime':<20} {'FAIBLE%':>9} {'MOYEN%':>9} {'ELEVE%':>9}")
    print(f"  {'-' * 50}")

    for reg in ["BULL", "BEAR", "HIGH_VOL", "SIDEWAYS"]:
        m = regime_metrics.get(reg, {})
        rdist = m.get("risk_distribution", {})
        total_r = sum(rdist.values()) or 1
        faible = rdist.get("FAIBLE", 0) / total_r
        moyen = rdist.get("MOYEN", 0) / total_r
        eleve = rdist.get("ELEVE", 0) / total_r
        if m.get("n_active", 0) == 0:
            continue
        icon = REGIME_COLORS_ASCII.get(reg, "?")
        print(f"  {icon} {reg:<18} {faible:>9.1%} {moyen:>9.1%} {eleve:>9.1%}")

    print(f"\n  {'-' * 70}")
    print("  [D] ANALYSE DE ROBUSTESSE INTER-RÉGIMES")
    print(f"  {'-' * 70}")

    if robustness.get("error"):
        print(f"\n  [INFO] {robustness['error']}")
    else:
        print(
            f"\n  Meilleur régime   : {robustness['best_regime']} "
            f"(Sharpe = {robustness['sharpes'].get(robustness['best_regime'], 'N/A')})"
        )
        print(
            f"  Pire régime       : {robustness['worst_regime']} "
            f"(Sharpe = {robustness['sharpes'].get(robustness['worst_regime'], 'N/A')})"
        )
        print(f"  Range des Sharpes : {robustness['sharpe_range']}")
        print(f"  Écart-type Sharpe : {robustness['sharpe_std']}")
        if robustness.get("bull_bear_ratio") is not None:
            bb = robustness["bull_bear_ratio"]
            print(f"  Ratio Bull/Bear   : {bb:.2f}x")
            if bb > 1.5:
                print("  [!] Biais haussier fort — la stratégie surperforme surtout en Bull")
            elif bb < -1.0:
                print("  [!] Comportement défensif — la stratégie surperforme en Bear (short-biased)")
            else:
                print("  [OK] Ratio équilibré")

        verdict = robustness.get("verdict", "?")
        print(f"\n  VERDICT ROBUSTESSE : {verdict}")
        if "FRAGILE" in verdict:
            print("  -> Le système est sensible aux changements de régime.")
            print("  -> Recommandation : Ajouter un filtre de régime au YOLO classifier.")
        elif "ROBUSTE" in verdict:
            print("  -> La performance est relativement stable entre les régimes.")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_market_regime_analysis() -> dict:
    """
    Couche 12 : Analyse de la robustesse aux régimes de marché.
    Classifie chaque trade selon le régime (Bull/Bear/High-Vol/Sideways)
    et calcule les métriques spécifiques à chaque régime.

    Références : Hamilton (1989), Lo (2004), Ang & Bekaert (2002)
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 12 : Robustesse aux Régimes de Marché")
    print("            Bull / Bear / Haute Volatilité / Lateral")
    print(f"{'=' * 70}")
    print("\n  Seuils :")
    print(
        f"   BULL    : SPY 20j > +{REGIME_THRESHOLDS['spy_bull_threshold']}% && VIX < {REGIME_THRESHOLDS['vix_medium']}"
    )
    print(f"   BEAR    : SPY 20j < {REGIME_THRESHOLDS['spy_bear_threshold']}% && VIX > {REGIME_THRESHOLDS['vix_low']}")
    print(f"   HIGH_VOL: VIX > {REGIME_THRESHOLDS['vix_high']} (prioritaire)")
    print("   SIDEWAYS: Sinon")
    print("\n  Récupération des données de régime (VIX + SPY)...")

    by_regime = _load_and_classify_trades()

    if not by_regime:
        print("\n  [INFO] Aucun trade disponible. Lancez d'abord L3 :")
        print("  python eval/run_eval.py --layer 3")
        return {}

    total = sum(len(v) for v in by_regime.values())
    print(f"  {total} trades classifiés en {len(by_regime)} régimes.")

    # Calcul des métriques par régime
    regime_metrics = {}
    for reg in ["BULL", "BEAR", "HIGH_VOL", "SIDEWAYS", "UNKNOWN"]:
        trades_in_regime = by_regime.get(reg, [])
        regime_metrics[reg] = _compute_regime_metrics(trades_in_regime)

    # Analyse de robustesse
    robustness = _compute_robustness_metrics(regime_metrics)

    # Rapport
    _print_regime_report(by_regime, regime_metrics, robustness)

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = EVAL_RESULTS_DIR / f"{timestamp}_market_regimes_layer12"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "layer": 12,
        "sub": "market_regimes",
        "n_trades_total": total,
        "regime_counts": {k: len(v) for k, v in by_regime.items()},
        "regime_metrics": regime_metrics,
        "robustness": robustness,
        "thresholds": REGIME_THRESHOLDS,
    }

    with open(out_dir / "market_regimes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Résultats sauvegardés : {out_dir}/market_regimes.json")
    print("=" * 70)

    return output


if __name__ == "__main__":
    run_market_regime_analysis()
