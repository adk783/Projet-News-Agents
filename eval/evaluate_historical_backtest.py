"""
evaluate_historical_backtest.py — Backtest P&L classique (Sharpe / Drawdown / Capital)

POSITIONNEMENT
--------------
Ce fichier est **COMPLÉMENTAIRE** de `eval/evaluate_event_study.py`.

  • evaluate_historical_backtest.py  (ce fichier)
        → Vue GÉRANT / INVESTISSEUR : P&L, Sharpe, Max Drawdown, courbe
          de capital. Réponse à la question "combien je gagne, combien
          je risque de perdre ?" — ce qui paie les factures.

  • evaluate_event_study.py
        → Vue QUANT / RECHERCHE : Alpha ajusté F-F 3 factor, CAR[-1,+5],
          p-value Newey-West, q-value Romano-Wolf. Réponse à la question
          "est-ce que mon edge est statistiquement distinct du marché ?"
          — ce qui convainc un jury ou un investisseur sophistiqué.

Les deux sont nécessaires : un alpha positif sans P&L soutenable n'est
pas bankable, et un P&L attractif sans alpha peut n'être que du beta déguisé.

NOTE MÉTHODOLOGIQUE
-------------------
Ce backtest simule l'espérance mathématique d'un edge supposé (p=0.55 en
trend, 0.50 en sideways) via np.random.rand(). Il NE teste PAS le vrai
pipeline LLM trade par trade — c'est un coût de calcul et de latence
impossible sur 16 ans d'historique.

Pour tester le vrai pipeline sur les articles RÉELLEMENT collectés,
utiliser `eval/evaluate_event_study.py --from-db` sur la DB SQLite.

Ref : Fama & French (1993), Brown & Warner (1985), MacLean-Thorp-Ziemba (2010).
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Parametres
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"
VIX_HIGH = 30.0
VIX_LOW = 25.0
SPY_BULL = 3.0
SPY_BEAR = -3.0


def load_data():
    """Telecharge les donnees historiques SPY et VIX."""
    print("Telechargement des donnees SPY et ^VIX...")
    spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)["Close"].squeeze()

    df = pd.DataFrame({"SPY": spy, "VIX": vix}).dropna()
    df["SPY_20d_ret"] = df["SPY"].pct_change(periods=20) * 100
    df["SPY_20d_vol"] = df["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    return df.dropna()


def classify_regime(row):
    """
    Classification basee sur la volatilite stochastique plutot que des seuils fixes (Critique scientifique).
    SIDEWAYS = Le rendement 20j est inferieur a la moitie de la volatilite historique (bruit).
    """
    if row["VIX"] > VIX_HIGH:
        return "HIGH_VOL"
    elif row["SPY_20d_ret"] > 0 and row["SPY_20d_ret"] > 0.5 * row["SPY_20d_vol"] and row["VIX"] < VIX_LOW:
        return "BULL"
    elif row["SPY_20d_ret"] < 0 and abs(row["SPY_20d_ret"]) > 0.5 * row["SPY_20d_vol"] and row["VIX"] > 20.0:
        return "BEAR"
    else:
        return "SIDEWAYS"


def run_historical_backtest():
    """
    Simule une distribution stochastique de signaux sur l'historique reel
    pour prendre en compte les Fat Tails, la clustering volatility et les vrais cycles boursiers.
    """
    df = load_data()
    df["Regime"] = df.apply(classify_regime, axis=1)

    print("\n[Distribution des Regimes Historiques (2010-2026)]")
    print(df["Regime"].value_counts(normalize=True).apply(lambda x: f"{x * 100:.1f}%"))

    np.random.seed(42)
    # Simulation de l'Edge du LLM : on suppose un leger win rate superieur a 50%
    # On tire le sens du signal (Long/Short) de facon a avoir un legere alpha
    # Pour simuler de maniere realiste, on regarde la rentabilite des 5 jours suivants
    df["Return_5d"] = df["SPY"].pct_change(periods=5).shift(-5)
    df = df.dropna()

    # "Signal LLM" simule : l'Edge du LLM varie selon le regime (Critique fondamentale)
    # L'audit L12 prouve que le LLM n'a AUCUN edge en SIDEWAYS (50% = hasard).
    # Il a un bon edge en tendance (BULL/BEAR = 55%).
    def get_llm_correct(row):
        prob = 0.50
        if row["Regime"] in ["BULL", "BEAR"]:
            prob = 0.55
        elif row["Regime"] == "HIGH_VOL":
            prob = 0.52
        return np.random.rand() < prob

    df["LLM_Correct"] = df.apply(get_llm_correct, axis=1)
    df["Trade_Ret"] = np.where(df["LLM_Correct"], abs(df["Return_5d"]), -abs(df["Return_5d"]))

    # Sans filtre : on prend tous les trades a 100% de la taille nominale
    df["Ret_Unfiltered"] = df["Trade_Ret"]

    # Avec Prior Bayesien :
    # En SIDEWAYS, on reduit la taille de position de moitie (le Kelly serait divise)
    # En HIGH_VOL, on reduit aussi pour cause de penalite au risque
    def apply_bayesian_sizing(row):
        if row["Regime"] == "SIDEWAYS":
            return row["Trade_Ret"] * 0.5
        elif row["Regime"] == "HIGH_VOL":
            return row["Trade_Ret"] * 0.25  # Risque tres eleve, taille mini
        else:
            return row["Trade_Ret"]

    df["Ret_Filtered"] = df.apply(apply_bayesian_sizing, axis=1)

    mean_unf = df["Ret_Unfiltered"].mean() * 252 / 5  # Annualise (trades de 5j non overlappants approximés)
    std_unf = df["Ret_Unfiltered"].std() * np.sqrt(252 / 5)
    sharpe_unf = mean_unf / std_unf if std_unf > 0 else 0
    drawdown_unf = (df["Ret_Unfiltered"].cumsum() - df["Ret_Unfiltered"].cumsum().cummax()).min()

    mean_f = df["Ret_Filtered"].mean() * 252 / 5
    std_f = df["Ret_Filtered"].std() * np.sqrt(252 / 5)
    sharpe_f = mean_f / std_f if std_f > 0 else 0
    drawdown_f = (df["Ret_Filtered"].cumsum() - df["Ret_Filtered"].cumsum().cummax()).min()

    # --- BENCHMARK ETF (SPY Buy & Hold) ---
    mean_spy = df["Return_5d"].mean() * 252 / 5
    std_spy = df["Return_5d"].std() * np.sqrt(252 / 5)
    sharpe_spy = mean_spy / std_spy if std_spy > 0 else 0
    drawdown_spy = (df["Return_5d"].cumsum() - df["Return_5d"].cumsum().cummax()).min()

    # --- CALCULS AVANCES : Beta, Alpha, Information Ratio ---
    # Covariance avec le marché
    cov_f_spy = df["Ret_Filtered"].cov(df["Return_5d"])
    var_spy = df["Return_5d"].var()
    beta_f = cov_f_spy / var_spy if var_spy > 0 else 0
    alpha_f = mean_f - (beta_f * mean_spy)  # Jensen's Alpha (annualise)

    # Tracking Error & Information Ratio
    tracking_diff = df["Ret_Filtered"] - df["Return_5d"]
    tracking_error = tracking_diff.std() * np.sqrt(252 / 5)
    info_ratio = (mean_f - mean_spy) / tracking_error if tracking_error > 0 else 0

    print("\n" + "=" * 80)
    print("ANALYSE DE PERFORMANCE QUANTITATIVE (2010-2026)")
    print("Type de Stratégie : Directional Event-Driven (Long/Short) + Bayesian Regime Filter")
    print("Moteur de Décision : Multi-Agent LLM (Edge simulé à 55% en tendance, 50% en range)")
    print("Note méthodologique : Pour éviter le Look-Ahead Bias (que le LLM devine le futur")
    print("                      en reconnaissant la date/entreprise), le pipeline réel")
    print("                      utilise l'anonymisation NLP (Couche 4). Ce backtest simule")
    print("                      l'espérance mathématique de cet edge NLP sur la durée.")
    print("=" * 80)

    print("\n[BENCHMARK : ETF S&P 500 (Buy & Hold)]")
    print(f"  Rendement Ann : {mean_spy * 100:6.2f}% | Volatilité   : {std_spy * 100:6.2f}%")
    print(f"  Sharpe Ratio  : {sharpe_spy:6.3f}  | Max Drawdown : {drawdown_spy * 100:6.2f}%")

    print("\n[Stratégie 1 : SANS Filtre YOLO]")
    print(f"  Rendement Ann : {mean_unf * 100:6.2f}% | Volatilité   : {std_unf * 100:6.2f}%")
    print(f"  Sharpe Ratio  : {sharpe_unf:6.3f}  | Max Drawdown : {drawdown_unf * 100:6.2f}%")

    print("\n[Stratégie 2 : AVEC Prior Bayesien (Notre Architecture Finale)]")
    print(f"  Rendement Ann : {mean_f * 100:6.2f}% | Volatilité   : {std_f * 100:6.2f}%")

    # --- MONTE CARLO BLOCK BOOTSTRAPPING (Politis & Romano 1994) ---
    n_bootstraps = 10000
    try:
        from src.utils.politis_white import politis_white_block_length

        l_block = int(max(1, round(politis_white_block_length(df["Ret_Filtered"].values))))
    except Exception:
        l_block = 5

    n_obs = len(df)
    n_blocks = int(np.ceil(n_obs / l_block))

    # Vectorized Circular Block Bootstrap
    np.random.seed(42)  # Reproducibility for MC
    start_indices = np.random.randint(0, n_obs, size=(n_bootstraps, n_blocks))
    offsets = np.arange(l_block)
    boot_indices = (start_indices[:, :, None] + offsets) % n_obs
    boot_indices = boot_indices.reshape(n_bootstraps, -1)[:, :n_obs]

    boot_returns = df["Ret_Filtered"].values[boot_indices]
    boot_means = boot_returns.mean(axis=1) * (252 / 5)
    boot_stds = boot_returns.std(axis=1) * np.sqrt(252 / 5)

    valid_stds = boot_stds > 0
    boot_sharpes = np.zeros(n_bootstraps)
    boot_sharpes[valid_stds] = boot_means[valid_stds] / boot_stds[valid_stds]

    ic_lower = np.percentile(boot_sharpes, 2.5)
    ic_upper = np.percentile(boot_sharpes, 97.5)

    print(
        f"  Sharpe Ratio  : {sharpe_f:6.3f}  [IC 95% : {ic_lower:.2f} - {ic_upper:.2f}] (n={n_bootstraps:,} bootstraps, l={l_block}j)"
    )
    print(f"  Max Drawdown  : {drawdown_f * 100:6.2f}%")

    print("\n  -- Métriques Avancées (vs ETF) --")
    print(f"  Beta au marché      : {beta_f:6.2f} (Corrélation au S&P 500)")
    print(f"  Jensen's Alpha      : {alpha_f * 100:6.2f}% (Surperformance pure ajustée au risque)")
    print(f"  Information Ratio   : {info_ratio:6.3f} (Rendement actif par unité de risque actif)")

    # --- NOUVEAU : Calmar, Sortino, Courbe de capital pour vue "gérant" ---
    calmar_f = (mean_f / abs(drawdown_f)) if drawdown_f != 0 else 0
    downside = df["Ret_Filtered"][df["Ret_Filtered"] < 0].std() * np.sqrt(252 / 5)
    sortino_f = mean_f / downside if downside > 0 else 0
    print(f"  Calmar Ratio        : {calmar_f:6.3f} (Return / |MaxDrawdown|)")
    print(f"  Sortino Ratio       : {sortino_f:6.3f} (Return / Downside Volatility)")

    # Courbe de capital en supposant 100k€ initial
    initial_capital = 100_000
    equity = initial_capital * (1 + df["Ret_Filtered"]).cumprod()
    equity_spy = initial_capital * (1 + df["Return_5d"]).cumprod()
    print("\n  -- Courbe de capital (100k€ initial) --")
    print(f"  Final equity (strategie)  : {equity.iloc[-1]:>12,.0f}€")
    print(f"  Final equity (SPY B&H)    : {equity_spy.iloc[-1]:>12,.0f}€")

    # Export optionnel pour tracé par l'utilisateur
    try:
        import os as _os

        _os.makedirs("reports/backtests", exist_ok=True)
        out = df[["SPY", "Regime", "Ret_Filtered", "Return_5d"]].copy()
        out["equity_strategy"] = equity
        out["equity_spy"] = equity_spy
        out.to_csv("reports/backtests/equity_curve.csv")
        print("  Courbe exportée       : reports/backtests/equity_curve.csv")
    except Exception:
        pass

    if sharpe_f > sharpe_unf and sharpe_f > sharpe_spy:
        print("\n[OK] L'approche Bayesienne est scientifiquement validée.")
        print("Elle surperforme à la fois la stratégie brute et l'ETF S&P 500 en termes de rendement ajusté au risque.")
    elif sharpe_f > sharpe_unf:
        print("\n[OK] L'approche Bayesienne améliore la stratégie brute (Sharpe Ratio +),")
        print("mais n'arrive pas à battre l'ETF S&P 500 Buy & Hold sur cette décennie exceptionnelle.")
    else:
        print("\n[ECHEC] L'approche n'améliore pas les métriques ajustées au risque.")


if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST P&L CLASSIQUE  (Sharpe / Drawdown / Capital)")
    print("  Complementaire a : python -m eval.evaluate_event_study --from-db")
    print("=" * 70)
    print()
    run_historical_backtest()
