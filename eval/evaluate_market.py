"""
evaluate_market.py — Couche 3 : Validation financière rétrospective (v3)
=========================================================================
Réponse à la question centrale : "Mes signaux sont-ils profitables ?"

DEUX SOURCES DE DONNÉES :
  1. Les résultats de la Couche 2 (evaluate_pipeline.py) sur le benchmark
     annoté — articles et dates CONNUES (2024-2025).
     → Récupère le cours via yfinance à la date de l'article et N jours après.
     → Compare le signal prédit au mouvement réel.
     → Calcule le P&L simulé d'un portefeuille qui suit les signaux FAIBLE.

  2. Les décisions stockées en SQLite (pipeline sur vraies news récentes).
     → Même logique, complète la source 1.

MÉTRIQUES CLÉS :
  - Précision globale des signaux
  - Précision par niveau de risque YOLO (FAIBLE / MOYEN / ELEVE)
    → Question : les signaux FAIBLE-risque sont-ils plus fiables ?
  - P&L simulé sur portefeuille "suivre FAIBLE uniquement"
  - P&L simulé sur portefeuille "suivre TOUT" (baseline)
  - Sharpe ratio simplifié
  - Benchmark vs S&P 500 (SPY) — ajout v3
    → Alpha = retour portefeuille − retour SPY buy-and-hold sur la même période
    → Valeur ajoutée réelle du système vs. simple suivi de l'indice

Lancé via : python eval/run_eval.py --layer 3 [--horizon 30]
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import yfinance as yf

logger = logging.getLogger("EvalMarket")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

DATABASE_PATH = "data/news_database.db"
EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"


# ---------------------------------------------------------------------------
# Prix réels via yfinance
# ---------------------------------------------------------------------------

_price_cache: dict[tuple, Optional[float]] = {}


def _get_close_price(ticker: str, date_str: str, offset_days: int = 0) -> Optional[float]:
    """
    Retourne le cours de clôture de 'ticker' à 'date_str + offset_days' jours.
    Retourne None si données indisponibles (weekend, données trop récentes...).
    """
    cache_key = (ticker, date_str, offset_days)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        pub = datetime.fromisoformat(date_str.replace("Z", "+00:00").replace("T00:00:00+00:00", ""))
        # On simplifie en parsant juste la date
        pub_date = datetime.strptime(str(pub)[:10], "%Y-%m-%d")
        target = pub_date + timedelta(days=offset_days)
        end = target + timedelta(days=5)  # Marge pour jours non-ouvrés

        hist = yf.Ticker(ticker).history(
            start=target.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True
        )
        if hist.empty:
            _price_cache[cache_key] = None
            return None

        price = round(float(hist["Close"].iloc[0]), 2)
        _price_cache[cache_key] = price
        return price

    except Exception as e:
        logger.debug("Prix %s @%s+%dd: %s", ticker, date_str, offset_days, e)
        _price_cache[cache_key] = None
        return None


# ---------------------------------------------------------------------------
# Benchmark S&P 500 (SPY) — ajout v3
# ---------------------------------------------------------------------------

_spy_cache: dict[tuple, Optional[float]] = {}


def _get_spy_return(date_str: str, horizon_days: int) -> Optional[float]:
    """
    Calcule le retour buy-and-hold du S&P 500 (ETF SPY) sur la même période
    que le trade évalué. Permet de calculer l'alpha du système.

    Returns:
        float : retour en % du SPY sur [date_str, date_str + horizon_days]
        None  : si données indisponibles
    """
    cache_key = (date_str, horizon_days)
    if cache_key in _spy_cache:
        return _spy_cache[cache_key]

    price_t0 = _get_close_price("SPY", date_str, 0)
    price_tn = _get_close_price("SPY", date_str, horizon_days)

    if price_t0 is None or price_tn is None or price_t0 == 0:
        _spy_cache[cache_key] = None
        return None

    spy_return = round((price_tn - price_t0) / price_t0 * 100, 2)
    _spy_cache[cache_key] = spy_return
    return spy_return


# ---------------------------------------------------------------------------
# Calcul du retour et validation du signal
# ---------------------------------------------------------------------------

SIGNAL_CORRECT_THRESHOLD_PCT = 1.0  # Un mouvement < 1% est considéré "bruit"


def _compute_trade_result(signal: str, price_t0: float, price_tn: float) -> dict:
    """
    Calcule le résultat d'un trade hypothétique.

    Si signal = "Achat"  : on simule un achat à T0, revente à TN.
    Si signal = "Vente"  : on simule une vente à découvert à T0, rachat à TN.
    Si signal = "Neutre" : pas de trade (return 0%).

    Retourne :
      return_pct    : retour en % du trade (positif = profitable)
      signal_correct: True si le signal prédit la bonne direction
      actionable    : False si mouvement trop faible (< threshold)
    """
    change_pct = (price_tn - price_t0) / price_t0 * 100
    actionable = abs(change_pct) >= SIGNAL_CORRECT_THRESHOLD_PCT

    SLIPPAGE = 0.05
    if signal == "Achat":
        return_pct = change_pct - SLIPPAGE
        correct = change_pct >= SIGNAL_CORRECT_THRESHOLD_PCT if actionable else None
    elif signal == "Vente":
        return_pct = -change_pct - SLIPPAGE  # Vente à découvert → profit si le cours baisse
        correct = change_pct <= -SIGNAL_CORRECT_THRESHOLD_PCT if actionable else None
    else:  # Neutre : pas de trade
        return_pct = 0.0
        correct = None

    return {
        "change_pct": round(change_pct, 2),
        "return_pct": round(return_pct, 2),
        "signal_correct": correct,
        "actionable": actionable,
    }


def _sharpe(returns: list[float]) -> float:
    """Sharpe ratio simplifié sans taux sans risque (annualisé sur base 252j)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)
    return round((mean / std) * math.sqrt(252), 2) if std > 0 else 0.0


# ---------------------------------------------------------------------------
# Source 1 : Résultats de la Couche 2 (benchmark annoté)
# ---------------------------------------------------------------------------


def _load_from_layer2() -> list[dict]:
    """
    Charge les prédictions du pipeline depuis le dernier run de la Couche 2.
    Retourne les entrées avec date, ticker, signal prédit, risk_level.
    """
    if not EVAL_RESULTS_DIR.exists():
        return []

    # Cherche le dernier dossier pipeline_layer2
    layer2_runs = sorted(
        [d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir() and "pipeline_layer2" in d.name], reverse=True
    )
    if not layer2_runs:
        return []

    latest = layer2_runs[0]
    details_file = latest / "details.json"
    if not details_file.exists():
        return []

    with open(details_file, encoding="utf-8") as f:
        results = json.load(f)

    entries = []
    for r in results:
        # Ignorer les hors_scope, les erreurs, et les articles sans date
        if r.get("filtrage_pred") == "hors_scope":
            continue
        if r.get("signal_pred") is None:
            continue

        gt = r.get("ground_truth", {})
        entries.append(
            {
                "source": "benchmark_L2",
                "id": r["id"],
                "ticker": r["ticker"],
                "date_utc": gt.get("date", ""),  # Date depuis le dataset
                "title": r["title"],
                "signal": r["signal_pred"],
                "risk_level": r.get("risk_level", "INCONNU"),
                "gt_signal": gt.get("signal"),  # Label annoté
            }
        )

    return entries


# ---------------------------------------------------------------------------
# Source 2 : SQLite (pipeline sur vraies news)
# ---------------------------------------------------------------------------


def _load_from_sqlite() -> list[dict]:
    """Charge les décisions depuis la DB (articles réellement traités)."""
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, date_utc, title, signal_final, risk_level
            FROM articles
            WHERE signal_final IN ('Achat', 'Vente', 'Neutre')
              AND date_utc IS NOT NULL
              AND date_utc != ''
            ORDER BY date_utc DESC
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.warning("Erreur lecture SQLite: %s", e)
        return []

    return [
        {
            "source": "sqlite_live",
            "id": f"{row['ticker']}_{row['date_utc'][:10]}",
            "ticker": row["ticker"],
            "date_utc": row["date_utc"],
            "title": (row["title"] or "")[:50],
            "signal": row["signal_final"],
            "risk_level": row["risk_level"] or "INCONNU",
            "gt_signal": None,  # Pas de label annoté pour les news live
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Analyse financière principale
# ---------------------------------------------------------------------------


def _analyse(entries: list[dict], horizon_days: int) -> dict:
    """
    Pour chaque entrée, récupère les cours et calcule le résultat du trade.
    Inclut désormais le benchmark SPY et le calcul d'alpha. (v3)
    Retourne un dict avec toutes les métriques financières.
    """
    validated = []
    skipped = 0

    for entry in entries:
        ticker = entry["ticker"]
        date_str = entry.get("date_utc") or entry.get("date", "")

        if not date_str:
            skipped += 1
            continue

        price_t0 = _get_close_price(ticker, date_str, 0)
        price_tn = _get_close_price(ticker, date_str, horizon_days)

        if price_t0 is None or price_tn is None:
            skipped += 1
            continue

        trade = _compute_trade_result(entry["signal"], price_t0, price_tn)

        # Benchmark SPY : retour buy-and-hold sur la même période
        spy_return = _get_spy_return(date_str, horizon_days)
        alpha = round(trade["return_pct"] - spy_return, 2) if spy_return is not None else None

        validated.append(
            {
                **entry,
                **trade,
                "price_t0": price_t0,
                "price_tn": price_tn,
                "spy_return": spy_return,  # Retour SPY sur la même période
                "alpha": alpha,  # Alpha = retour trade − retour SPY
            }
        )

    return {
        "trades": validated,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Rapport financier
# ---------------------------------------------------------------------------


def _print_financial_report(trades: list[dict], horizon: int) -> None:
    if not trades:
        print("\n  [INFO] Aucune trade validee - donnees de prix insuffisantes.")
        return

    # --- Tableau détaillé ---
    print(f"\n  {'ID':<14} {'Sig':>6} {'Risq':>7} {'T0':>8} {'T+N':>8} {'delta':>8} {'Ret':>8} {'OK':>5}")
    print(f"  {'-' * 66}")
    for t in trades:
        ok = "OK" if t["signal_correct"] else "FAIL" if t["signal_correct"] is False else "--"
        sig = t["signal"][:5]
        print(
            f"  {t['id']:<14} {sig:>6} {t['risk_level']:>7} "
            f"{t['price_t0']:>8.2f} {t['price_tn']:>8.2f} "
            f"{t['change_pct']:>+8.1f}% {t['return_pct']:>+8.1f}% {ok:>5}"
        )

    # --- Métriques globales ---
    all_actionable = [t for t in trades if t["actionable"] and t["signal"] != "Neutre"]
    overall_acc = sum(1 for t in all_actionable if t["signal_correct"]) / len(all_actionable) if all_actionable else 0.0

    print(
        f"\n  Precision globale (signaux actionnables): {overall_acc:.0%} "
        f"({sum(1 for t in all_actionable if t['signal_correct'])}/{len(all_actionable)})"
    )

    # --- Précision par niveau YOLO ---
    print("\n  Precision par niveau de risque YOLO :")
    print(f"  {'Niveau':<10} {'Precision':>10} {'N trades':>10} {'Correct':>10}")
    print(f"  {'-' * 42}")
    for level in ["FAIBLE", "MOYEN", "ELEVE", "INCONNU"]:
        lvl_trades = [t for t in all_actionable if t["risk_level"] == level]
        if not lvl_trades:
            print(f"  {level:<10} {'N/A':>10} {0:>10} {'—':>10}")
            continue
        n_correct = sum(1 for t in lvl_trades if t["signal_correct"])
        acc = n_correct / len(lvl_trades)
        print(f"  {level:<10} {acc:>10.0%} {len(lvl_trades):>10} {n_correct:>10}")

    # --- Portefeuille simulé ---
    print(f"\n  === SIMULATION DE PORTEFEUILLE (horizon = {horizon}j) ===")

    # Portefeuille 1 : Suivre TOUT (baseline)
    all_returns = [t["return_pct"] for t in trades if t["signal"] != "Neutre"]
    total_all = sum(all_returns)
    print("\n  Portefeuille A - Suivre tous les signaux (baseline) :")
    if all_returns:
        print(f"    Trades          : {len(all_returns)}")
        print(f"    Retour cumule   : {total_all:+.2f}%")
        print(f"    Retour moyen    : {total_all / len(all_returns):+.2f}% / trade")
        wins = sum(1 for r in all_returns if r > 0)
        print(f"    Taux de gain    : {wins}/{len(all_returns)} = {wins / len(all_returns):.0%}")
        print(f"    Sharpe ratio    : {_sharpe(all_returns):.2f}")
    else:
        print("    Aucune donnee.")

    # Portefeuille 2 : Suivre uniquement les signaux FAIBLE-risque
    faible_returns = [t["return_pct"] for t in trades if t["risk_level"] == "FAIBLE" and t["signal"] != "Neutre"]
    total_faible = sum(faible_returns)
    print("\n  Portefeuille B - Suivre uniquement les signaux FAIBLE (YOLO filtre) :")
    if faible_returns:
        print(f"    Trades          : {len(faible_returns)}")
        print(f"    Retour cumule   : {total_faible:+.2f}%")
        print(f"    Retour moyen    : {total_faible / len(faible_returns):+.2f}% / trade")
        wins_f = sum(1 for r in faible_returns if r > 0)
        print(f"    Taux de gain    : {wins_f}/{len(faible_returns)} = {wins_f / len(faible_returns):.0%}")
        print(f"    Sharpe ratio    : {_sharpe(faible_returns):.2f}")
        if all_returns:
            delta = total_faible - total_all
            print("\n  APPORT DU YOLO CLASSIFIER :")
            print(f"    Delta P&L      : {delta:+.2f}% (B vs A)")
            if total_faible > total_all:
                print("    Conclusion     : Le YOLO classifier AMELIORE la performance (filtre les mauvais signaux)")
            elif total_faible < total_all:
                print("    Conclusion     : Le YOLO classifier SOUS-PERFORME (trop conservateur)")
            else:
                print("    Conclusion     : Equivalent (pas encore assez de donnees)")
    else:
        print("    Aucun signal FAIBLE-risque dans les donnees.")
        print("    -> Cela signifie que le YOLO classifier est tres conservateur sur ce dataset.")
        print(f"       Distribution actuelle : { {t['risk_level'] for t in trades} }")

    # --- Benchmark vs SPY ---
    spy_returns_available = [t["spy_return"] for t in trades if t.get("spy_return") is not None]
    if spy_returns_available:
        spy_mean = sum(spy_returns_available) / len(spy_returns_available)
        if all_returns:
            portfolio_mean = sum(all_returns) / len(all_returns)
            mean_alpha = round(portfolio_mean - spy_mean, 2)
            print("\n  === BENCHMARK vs S&P 500 (SPY) ===")
            print(f"    Retour moyen SPY  : {spy_mean:+.2f}% / trade")
            print(f"    Retour moyen port.: {portfolio_mean:+.2f}% / trade")
            print(f"    Alpha moyen       : {mean_alpha:+.2f}% / trade")
            if mean_alpha > 0:
                print("    Conclusion        : Le systeme SURPERFORME le marche (alpha positif)")
            else:
                print("    Conclusion        : Le systeme SOUS-PERFORME le marche (alpha negatif)")

    # --- Courbe cumulative ASCII ---
    if faible_returns or all_returns:
        returns_to_show = faible_returns if faible_returns else all_returns
        cumulative = []
        cum = 0.0
        for r in returns_to_show:
            cum += r
            cumulative.append(cum)

        print(f"\n  Courbe cumulative (Portefeuille {'B - FAIBLE' if faible_returns else 'A - Tous'}) :")
        min_v = min(cumulative)
        max_v = max(cumulative)
        rng = max_v - min_v or 1
        height = 4
        for row in range(height, 0, -1):
            threshold = min_v + (row / height) * rng
            line = "".join("X" if v >= threshold else "." for v in cumulative)
            print(f"  {threshold:+6.1f}% | {line}")
        print(f"         +{'-' * len(cumulative)}")
        print("          Transactions (de gauche a droite)")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_market_validation(horizon_days: int = 5) -> dict:
    """
    Couche 3 : Validation financière rétrospective.
    Combine les résultats de Layer 2 (benchmark) et SQLite (live).
    """
    print(f"\n{'=' * 70}")
    print(f"COUCHE 3 : Validation financiere retrospective (horizon = {horizon_days}j)")
    print(f"{'=' * 70}")

    # Chargement des deux sources
    benchmark_entries = _load_from_layer2()
    sqlite_entries = _load_from_sqlite()

    print(f"\n  Source 1 - Benchmark L2 annote  : {len(benchmark_entries)} articles")
    print(f"  Source 2 - SQLite (pipeline live): {len(sqlite_entries)} articles")

    all_entries = benchmark_entries + sqlite_entries

    if not all_entries:
        print("\n  [INFO] Aucune donnee disponible.")
        print("  -> Lancez d'abord : python eval/run_eval.py --layer 2 --limit 10")
        return {}

    print("\n  Recuperation des prix de marche (yfinance)...\n")

    result = _analyse(all_entries, horizon_days)
    trades = result["trades"]

    if result["skipped"] > 0:
        print(f"  [INFO] {result['skipped']} articles ignores (prix non disponibles via yfinance)")

    if not trades:
        print("\n  [INFO] Aucune donnee de prix recuperable.")
        print("  -> Les articles du benchmark sont peut-etre trop recents ou hors-cote.")
        return {}

    # Rapport
    _print_financial_report(trades, horizon_days)

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = EVAL_RESULTS_DIR / f"{timestamp}_market_layer3"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "trades.json", "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultats sauvegardes dans : {out_dir}")
    print("=" * 70)

    faible_returns = [t["return_pct"] for t in trades if t["risk_level"] == "FAIBLE" and t["signal"] != "Neutre"]
    return {
        "layer": 3,
        "n_trades": len(trades),
        "n_skipped": result["skipped"],
        "global_accuracy": (sum(1 for t in trades if t["signal_correct"]) / len(trades) if trades else 0),
        "returns_faible": faible_returns,
        "total_return_faible": sum(faible_returns),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30, help="Horizon de validation en jours (defaut: 30)")
    args = parser.parse_args()
    run_market_validation(horizon_days=args.horizon)
