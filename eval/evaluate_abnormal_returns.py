"""
evaluate_abnormal_returns.py -- Étude d'événement : AR, CAR, tests de significativité
=======================================================================================
THÉORIE (Event Study Methodology - Fama et al., 1969) :

  Le cours d'un titre bouge pour deux raisons :
    1. Le marché en général monte ou descend (inutile pour nous).
    2. La news spécifique crée une valeur informationnelle (ce qu'on mesure).

  Pour isoler l'effet de la news, on retire le rendement "attendu" du titre
  estimé à partir de sa relation historique avec le marché (modèle de CAPM).

  Rendement Anormal (AR) = R_titre - (alpha + beta * R_marché)
  Rendement Anormal Cumulé (CAR) = Somme des AR sur la fenêtre d'événement

  H0: CAR = 0 (la news n'a pas eu d'effet anormal)
  H1: CAR ≠ 0 (la news a créé un rendement anormal — signal informatif)

  Si notre signal prédit "Achat" ET que le CAR est positif ET significatif :
  → La news contenait vraiment de l'information bullish
  → Notre décision était économiquement justifiée, pas le fruit du hasard

PARAMÈTRES :
  ESTIMATION_WINDOW = 120 jours AVANT l'événement (pour estimer alpha, beta)
  PRE_GAP           = 20 jours (tampon entre estimation et événement)
  EVENT_WINDOW      = (-1, +5) jours (fenêtre autour de la publication)
  MARKET_TICKER     = "^GSPC" (S&P 500 comme proxy du marché)

Lancé via : python eval/run_eval.py --layer 5 --sub ar
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import yfinance as yf
from scipy import stats

logger = logging.getLogger("AR_CAR")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paramètres de l'étude d'événement
# ---------------------------------------------------------------------------

ESTIMATION_DAYS = 120  # Fenêtre d'estimation avant l'événement
PRE_GAP_DAYS = 20  # Tampon entre estimation et événement
EVENT_WINDOW = (-1, 5)  # [-1 jour avant, +5 jours après] la publication
MARKET_TICKER = "^GSPC"  # S&P 500
SIGNIFICANCE = 0.05  # Seuil de significativité (p < 0.05)


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------


def _get_returns(ticker: str, start: str, end: str) -> Optional[np.ndarray]:
    """Retourne les rendements journaliers log-returns d'un ticker."""
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if len(hist) < 5:
            return None
        prices = hist["Close"].values
        returns = np.log(prices[1:] / prices[:-1])
        return returns
    except Exception as e:
        logger.warning("[AR] Erreur %s %s-%s: %s", ticker, start, end, e)
        return None


def _get_daily_returns_df(ticker: str, start: str, end: str):
    """Retourne un DataFrame date→return pour l'alignement temporel."""
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None
        hist.index = hist.index.tz_localize(None)
        prices = hist["Close"]
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns
    except Exception as e:
        logger.warning("[AR] Erreur DataFrame %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Étude d'événement principale
# ---------------------------------------------------------------------------


def run_event_study(
    ticker: str,
    event_date_str: str,
    signal_predicted: str,
) -> dict:
    """
    Exécute une étude d'événement complète pour un article.

    Args:
        ticker            : Symbole boursier (ex: "AAPL")
        event_date_str    : Date de publication de l'article (ISO format)
        signal_predicted  : Signal prédit par le pipeline ("Achat"/"Vente"/"Neutre")

    Returns:
        Dict avec AR, CAR, t-stat, p-value, significatif, conclusion.
    """
    result = {
        "ticker": ticker,
        "event_date": event_date_str,
        "signal_predicted": signal_predicted,
        "alpha": None,
        "beta": None,
        "r_squared": None,
        "ar": None,
        "car": None,
        "car_std": None,
        "t_stat": None,
        "p_value": None,
        "significant": False,
        "car_direction": None,  # "positif" ou "negatif"
        "signal_correct": None,  # True si signal aligné avec CAR significatif
        "error": None,
    }

    try:
        event_date = datetime.strptime(event_date_str[:10], "%Y-%m-%d")

        # ----- Fenêtre d'estimation -----
        est_end = event_date - timedelta(days=PRE_GAP_DAYS)
        est_start = est_end - timedelta(days=ESTIMATION_DAYS + 30)  # Marge weekends

        # ----- Fenêtre d'événement -----
        ev_start = event_date + timedelta(days=EVENT_WINDOW[0] - 1)
        ev_end = event_date + timedelta(days=EVENT_WINDOW[1] + 4)  # Marge weekends

        # Télécharger les données
        r_stock_est = _get_daily_returns_df(ticker, est_start.strftime("%Y-%m-%d"), est_end.strftime("%Y-%m-%d"))
        r_market_est = _get_daily_returns_df(
            MARKET_TICKER, est_start.strftime("%Y-%m-%d"), est_end.strftime("%Y-%m-%d")
        )
        r_stock_ev = _get_daily_returns_df(ticker, ev_start.strftime("%Y-%m-%d"), ev_end.strftime("%Y-%m-%d"))
        r_market_ev = _get_daily_returns_df(MARKET_TICKER, ev_start.strftime("%Y-%m-%d"), ev_end.strftime("%Y-%m-%d"))

        if any(x is None for x in [r_stock_est, r_market_est, r_stock_ev, r_market_ev]):
            result["error"] = "Données de prix insuffisantes"
            return result

        # ----- Régression OLS sur la fenêtre d'estimation -----
        # Aligner les dates (inner join)
        est_common = r_stock_est.index.intersection(r_market_est.index)
        if len(est_common) < 30:
            result["error"] = f"Seulement {len(est_common)} jours en commun estimation"
            return result

        y = r_stock_est[est_common].values
        x = r_market_est[est_common].values
        slope, intercept, r_value, _, std_err = stats.linregress(x, y)
        result["alpha"] = round(float(intercept), 6)
        result["beta"] = round(float(slope), 4)
        result["r_squared"] = round(float(r_value**2), 4)

        # ----- Calcul des rendements anormaux dans la fenêtre d'événement -----
        ev_common = r_stock_ev.index.intersection(r_market_ev.index)
        # Filtrer pour ne garder que EVENT_WINDOW[0]..EVENT_WINDOW[1] jours autour de l'événement
        ev_dates = sorted(
            [
                d
                for d in ev_common
                if event_date - timedelta(days=abs(EVENT_WINDOW[0]) + 1)
                <= d
                <= event_date + timedelta(days=EVENT_WINDOW[1] + 1)
            ]
        )[: abs(EVENT_WINDOW[0]) + EVENT_WINDOW[1] + 1]

        if len(ev_dates) < 3:
            result["error"] = f"Seulement {len(ev_dates)} jours dans la fenêtre d'événement"
            return result

        ar_values = []
        for d in ev_dates:
            r_stock_d = float(r_stock_ev[d])
            r_market_d = float(r_market_ev[d])
            expected = intercept + slope * r_market_d
            ar = r_stock_d - expected
            ar_values.append(ar)

        ar_array = np.array(ar_values)
        car = float(np.sum(ar_array))
        car_std = float(np.std(ar_array) * math.sqrt(len(ar_array)))

        # ----- Test t : H0: CAR = 0 -----
        sigma_est = float(np.std(y))  # Écart-type des résidus de l'estimation
        car_var = sigma_est**2 * len(ar_array)
        t_stat = car / math.sqrt(car_var) if car_var > 0 else 0.0
        # Degré de liberté = nb jours estimation - 2 (OLS)
        df = len(est_common) - 2
        p_value = float(2 * stats.t.sf(abs(t_stat), df))

        # ----- Conclusion -----
        significant = p_value < SIGNIFICANCE
        car_dir = "positif" if car > 0 else "negatif"

        # Le signal est "correct" si :
        #   - Signal Achat et CAR positif et significatif
        #   - Signal Vente et CAR négatif et significatif
        #   - Signal Neutre et CAR non significatif
        signal_correct = None
        if significant:
            if signal_predicted == "Achat":
                signal_correct = car > 0
            elif signal_predicted == "Vente":
                signal_correct = car < 0
        else:
            # CAR non significatif → la news n'avait pas d'effet économique mesurable
            # Un signal Neutre serait le plus approprié dans ce cas
            signal_correct = signal_predicted == "Neutre" or None

        result.update(
            {
                "ar": [round(float(v) * 100, 4) for v in ar_values],  # En %
                "car": round(car * 100, 4),  # En %
                "car_std": round(car_std * 100, 4),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": significant,
                "car_direction": car_dir,
                "signal_correct": signal_correct,
                "n_est_days": len(est_common),
                "n_ev_days": len(ev_dates),
            }
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        logger.error("[AR] %s %s: %s", ticker, event_date_str, e)

    return result


# ---------------------------------------------------------------------------
# Chargement des prédictions depuis L2 et/ou SQLite
# ---------------------------------------------------------------------------

EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"


def _load_predictions() -> list[dict]:
    """Charge les prédictions depuis le dernier run L2 + SQLite."""
    predictions = []

    # Depuis L2 (benchmark annoté)
    layer2_runs = (
        sorted([d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir() and "pipeline_layer2" in d.name], reverse=True)
        if EVAL_RESULTS_DIR.exists()
        else []
    )

    if layer2_runs:
        details_file = layer2_runs[0] / "details.json"
        if details_file.exists():
            with open(details_file, encoding="utf-8") as f:
                l2_data = json.load(f)
            for r in l2_data:
                if (
                    r.get("filtrage_pred") == "pertinent"
                    and r.get("signal_pred") is not None
                    and r.get("ground_truth", {}).get("date")
                ):
                    predictions.append(
                        {
                            "source": "benchmark_L2",
                            "id": r["id"],
                            "ticker": r["ticker"],
                            "date": r["ground_truth"]["date"],
                            "signal": r["signal_pred"],
                        }
                    )

    # Depuis SQLite (pipeline live)
    try:
        conn = sqlite3.connect("data/news_database.db", timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker, date_utc, signal_final
            FROM articles
            WHERE signal_final IN ('Achat', 'Vente', 'Neutre')
              AND date_utc IS NOT NULL
            ORDER BY date_utc DESC
        """)
        for row in cur.fetchall():
            predictions.append(
                {
                    "source": "sqlite_live",
                    "id": f"{row['ticker']}_{row['date_utc'][:10]}",
                    "ticker": row["ticker"],
                    "date": row["date_utc"],
                    "signal": row["signal_final"],
                }
            )
        conn.close()
    except Exception:
        pass

    return predictions


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def run_abnormal_returns_analysis(limit: int = 0) -> dict:
    """
    Lance l'analyse des rendements anormaux sur toutes les prédictions disponibles.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 5a : Rendements Anormaux (AR/CAR) — Étude d'événement")
    print(f"{'=' * 70}")
    print(f"  Fenetre estimation : {ESTIMATION_DAYS} jours avant publication")
    print(f"  Fenetre evenement  : [{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}] jours")
    print(f"  Marche reference   : {MARKET_TICKER} (S&P 500)")
    print(f"  Seuil significat.  : p < {SIGNIFICANCE}")

    predictions = _load_predictions()
    if limit > 0:
        predictions = predictions[:limit]

    if not predictions:
        print("\n  [INFO] Aucune prediction disponible. Lancez d'abord: python eval/run_eval.py --layer 2")
        return {}

    print(f"\n  {len(predictions)} predictions a analyser...\n")

    results = []
    n_significant = 0
    n_signal_correct = 0
    n_evaluated = 0

    print(f"  {'ID':<14} {'Signal':<8} {'CAR%':>8} {'p':>8} {'Sig':>6} {'OK?':>6} {'beta':>6}")
    print(f"  {'-' * 58}")

    for pred in predictions:
        study = run_event_study(pred["ticker"], pred["date"], pred["signal"])

        if study.get("error"):
            print(f"  {pred['id']:<14} {pred['signal']:<8} {'ERR':>8} {study['error'][:20]}")
            continue

        car_str = f"{study['car']:+.2f}"
        p_str = f"{study['p_value']:.3f}"
        sig = "YES" if study["significant"] else "no"
        ok_str = "OK" if study["signal_correct"] is True else "FAIL" if study["signal_correct"] is False else "--"
        beta = f"{study['beta']:.2f}" if study["beta"] is not None else "--"

        print(f"  {pred['id']:<14} {pred['signal']:<8} {car_str:>8} {p_str:>8} {sig:>6} {ok_str:>6} {beta:>6}")

        if study["significant"]:
            n_significant += 1
        if study["signal_correct"] is True:
            n_signal_correct += 1
        if study["signal_correct"] is not None:
            n_evaluated += 1

        results.append({**pred, **study})

    # Résumé
    n_results = len(results)
    print(f"\n  CAR significatifs : {n_significant}/{n_results} ({n_significant / n_results:.0%} si n>0)")
    print(
        f"  Signaux economiquement justifies : {n_signal_correct}/{n_evaluated} "
        f"({'N/A' if not n_evaluated else f'{n_signal_correct / n_evaluated:.0%}'})"
    )

    cars = [r["car"] for r in results if r.get("car") is not None]
    if cars:
        print(f"\n  CAR moyen (toutes decisions)         : {sum(cars) / len(cars):+.2f}%")
        sig_cars = [r["car"] for r in results if r.get("significant") and r.get("car") is not None]
        if sig_cars:
            print(f"  CAR moyen (uniquement significatifs) : {sum(sig_cars) / len(sig_cars):+.2f}%")

    # Verdict
    print("\n  Interpretation :")
    if n_significant == 0:
        print("  Les decisions ne correspondent pas a des mouvements anormaux mesurables.")
        print("  Soit les articles sont trop peu impactants, soit l'echantillon est trop petit.")
    elif n_signal_correct / max(n_evaluated, 1) > 0.6:
        print("  Les signaux sont economiquement justifies — la news contenait")
        print("  une information actionnable au-dela du bruit de marche.")
    else:
        print("  Les signaux ne capturent pas systematiquement l'effet economique de la news.")
        print("  Envisager un seuil d'impact plus strict dans le YOLO classifier.")

    print("=" * 70)

    return {
        "sub": "abnormal_returns",
        "n_predictions": n_results,
        "n_significant": n_significant,
        "n_signal_correct": n_signal_correct,
        "details": results,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    run_abnormal_returns_analysis(limit=a.limit)
