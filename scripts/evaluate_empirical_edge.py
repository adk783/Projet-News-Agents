import sqlite3

# Import event study logic
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent))
from eval.evaluate_event_study import compute_event_car, load_factors

DB_PATH = Path("data/news_database.db")


def classify_regime(row):
    if pd.isna(row["VIX"]):
        return "UNKNOWN"
    if row["VIX"] > 30.0:
        return "HIGH_VOL"
    elif row["SPY_20d_ret"] > 0 and row["SPY_20d_ret"] > 0.5 * row["SPY_20d_vol"] and row["VIX"] < 25.0:
        return "BULL"
    elif row["SPY_20d_ret"] < 0 and abs(row["SPY_20d_ret"]) > 0.5 * row["SPY_20d_vol"] and row["VIX"] > 20.0:
        return "BEAR"
    else:
        return "SIDEWAYS"


def main():
    if not DB_PATH.exists():
        print("Base de données introuvable.")
        return

    # 1. Load articles
    with sqlite3.connect(DB_PATH) as conn:
        q = "SELECT ticker, date_utc, signal_final FROM articles WHERE signal_final IN ('Achat', 'Vente') AND date_utc IS NOT NULL AND ticker IS NOT NULL"
        df_articles = pd.read_sql_query(q, conn)

    if df_articles.empty:
        print("Aucun article avec signal Achat/Vente trouvé.")
        return

    df_articles["date_utc"] = pd.to_datetime(df_articles["date_utc"], utc=True).dt.tz_localize(None)

    min_date = df_articles["date_utc"].min() - timedelta(days=60)
    max_date = df_articles["date_utc"].max() + timedelta(days=20)

    print(f"Telechargement SPY et ^VIX de {min_date.date()} à {max_date.date()}...")
    spy = yf.download(
        "SPY", start=min_date.strftime("%Y-%m-%d"), end=max_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True
    )
    vix = yf.download(
        "^VIX", start=min_date.strftime("%Y-%m-%d"), end=max_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True
    )

    close_spy = spy["Close"].squeeze() if "Close" in spy.columns else spy.iloc[:, 0].squeeze()
    close_vix = vix["Close"].squeeze() if "Close" in vix.columns else vix.iloc[:, 0].squeeze()

    df_macro = pd.DataFrame({"SPY": close_spy, "VIX": close_vix}).dropna()
    df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None)
    df_macro["SPY_20d_ret"] = df_macro["SPY"].pct_change(periods=20) * 100
    df_macro["SPY_20d_vol"] = df_macro["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df_macro = df_macro.dropna().sort_index()

    # 2. Merge asof
    df_articles = df_articles.sort_values("date_utc")
    df_merged = pd.merge_asof(
        df_articles, df_macro.reset_index(), left_on="date_utc", right_on="Date", direction="nearest"
    )

    df_merged["market_regime"] = df_merged.apply(classify_regime, axis=1)

    print("\n=== MAPPING DES DATES ET REGIMES ===")
    for _, row in df_merged.iterrows():
        print(
            f"Article: {row['date_utc'].strftime('%Y-%m-%d')} | Lookup Trading: {row['Date'].strftime('%Y-%m-%d')} | Ticker: {row['ticker']:4s} | Signal: {row['signal_final']:5s} | Regime: {row['market_regime']}"
        )

    # 3. Calculate Correctness via Event Study (CAR)
    print("\nChargement des facteurs Fama-French (proxy ETFs)...")
    ff_start = (min_date - timedelta(days=500)).strftime("%Y-%m-%d")
    ff_end = max_date.strftime("%Y-%m-%d")
    factors, _ = load_factors(ff_start, ff_end, prefer_ken_french=False)

    print("Calcul des Cumulative Abnormal Returns (CAR)...")
    results = []
    for _, row in df_merged.iterrows():
        res = compute_event_car(
            row["ticker"], row["date_utc"].strftime("%Y-%m-%d"), row["signal_final"], factors, direction_aware=True
        )
        if res is not None:
            # Correct if CAR > 0
            is_correct = 1 if res.car > 0 else 0
            results.append({"regime": row["market_regime"], "correct": is_correct})
        else:
            print(f"Impossible de calculer CAR pour {row['ticker']} à {row['date_utc'].date()}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("Aucun CAR calculé.")
        return

    print("\n=== Analyse de l'edge LLM par régime de marché ===")

    def print_stats(regime_name, df_subset):
        n = len(df_subset)
        if n == 0:
            print(f"Régime {regime_name:8s}: {n:2d} articles | Précision LLM : N/A")
            return

        acc = df_subset["correct"].mean() * 100

        # Bootstrap CI
        np.random.seed(42)
        boot = np.random.choice(df_subset["correct"].values, size=(10000, n), replace=True)
        boot_means = boot.mean(axis=1) * 100
        ic_low, ic_high = np.percentile(boot_means, [2.5, 97.5])

        print(
            f"Régime {regime_name:8s}: {n:2d} articles | Précision LLM : {acc:5.1f}% | [IC 95% : {ic_low:5.1f}% - {ic_high:5.1f}%]"
        )

    # Regrouper BULL et BEAR pour la tendance
    trend_df = df_res[df_res["regime"].isin(["BULL", "BEAR"])]
    sideways_df = df_res[df_res["regime"] == "SIDEWAYS"]

    print_stats("BULL", df_res[df_res["regime"] == "BULL"])
    print_stats("BEAR", df_res[df_res["regime"] == "BEAR"])
    print_stats("SIDEWAYS", sideways_df)

    print("\nHypothèse actuelle du backtest : 55.0% Tendance / 50.0% Sideways")

    trend_acc = trend_df["correct"].mean() * 100 if len(trend_df) > 0 else 0
    side_acc = sideways_df["correct"].mean() * 100 if len(sideways_df) > 0 else 0
    print(f"Mesure réelle                  : {trend_acc:.1f}% Tendance / {side_acc:.1f}% Sideways")

    total_n = len(df_res)
    if total_n < 30:
        print(
            f"\n[WARNING] Échantillon Proof-of-Concept trop faible (N={total_n} < 30) pour validation statistique. L'Intervalle de Confiance est trop large pour conclure avec certitude."
        )


if __name__ == "__main__":
    main()
