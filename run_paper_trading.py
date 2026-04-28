"""
run_paper_trading.py — Demon de paper trading multi-agents (DRY_RUN)

CYCLE DE 30 MINUTES :
  1. Selection dynamique des tickers via TickerDiscoveryEngine
     (volume anomaly + earnings calendar via yfinance, sur un univers de
     leaders S&P 500). Fallback liste curatee si la discovery retourne 0.
  2. Ingestion des news pour les tickers selectionnes (news_pipeline)
  3. Si le marche est ouvert : analyse multi-agents (agent_pipeline)
  4. A 02:30 (UTC local) une fois par jour : refit du calibrateur
     (scripts/fit_calibration_nightly.py)
  5. Le dimanche a 23h : audit hebdomadaire complet
     (scripts/audit_hebdomadaire.py)

VARIABLES D'ENVIRONNEMENT :
  - DRY_RUN=1                    : force le mode simulation (pose ici)
  - LLM_DAILY_BUDGET_USD=5.0     : budget LLM journalier (pose ici)
  - LOCUS_TICKERS="AAPL MSFT"    : override manuel des tickers (optionnel)
  - LOCUS_TOP_N=10               : nombre de tickers selectionnes par cycle
  - LOCUS_INTERVAL_MIN=30        : intervalle entre 2 cycles (minutes)

Lancement :
    python run_paper_trading.py
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
os.environ.setdefault("DRY_RUN", "1")
os.environ.setdefault("LLM_DAILY_BUDGET_USD", "5.0")

INTERVAL_MINUTES = int(os.environ.get("LOCUS_INTERVAL_MIN", "30"))
TOP_N            = int(os.environ.get("LOCUS_TOP_N", "10"))
TICKERS_OVERRIDE = os.environ.get("LOCUS_TICKERS", "").strip()
CALIBRATION_FLAG = Path("data/.last_calibration_date")

# Univers de candidats : leaders S&P 500 par capitalisation.
# La discovery scoore ceux-ci, ne les utilise pas tous.
CANDIDATE_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "WMT", "LLY", "V", "ORCL", "MA", "XOM", "PG", "JNJ", "HD", "COST",
    "ABBV", "BAC", "NFLX", "KO", "CVX", "MRK", "AMD", "PEP", "TMO", "CSCO",
    "ADBE", "WFC", "CRM", "MCD", "LIN", "ACN", "DIS", "ABT", "GE", "VZ",
    "INTC", "QCOM", "TXN", "PFE", "AMGN", "PM", "IBM", "NOW", "CAT", "BA",
)

# Fallback minimal si la discovery retourne 0 candidats.
FALLBACK_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "NVDA", "TSLA")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("paper_trading")


# ----------------------------------------------------------------------------
# Selection dynamique des tickers
# ----------------------------------------------------------------------------

def _yfinance_volume_fetcher(tickers: tuple[str, ...]) -> dict:
    """Retourne {ticker: {volume, volume_ma_20, volume_std_20}} via yfinance."""
    import numpy as np
    import yfinance as yf
    out: dict = {}
    try:
        data = yf.download(
            list(tickers), period="30d", progress=False, auto_adjust=True, threads=True,
        )
        vol_df = data["Volume"] if "Volume" in data.columns else None
        if vol_df is None or vol_df.empty:
            return {}
        for tk in tickers:
            try:
                series = vol_df[tk].dropna() if tk in vol_df.columns else vol_df.dropna()
                if len(series) < 21:
                    continue
                out[tk] = {
                    "volume":         float(series.iloc[-1]),
                    "volume_ma_20":   float(series.iloc[-21:-1].mean()),
                    "volume_std_20":  float(series.iloc[-21:-1].std() or 1.0),
                }
            except Exception:
                continue
    except Exception as e:
        logger.warning("[discovery] yfinance volume fetch echec : %s", e)
    return out


def _yfinance_earnings_fetcher(tickers: tuple[str, ...]) -> dict:
    """Retourne {ticker: days_to_earnings} via yfinance.calendar (best-effort)."""
    import yfinance as yf
    today = datetime.now(timezone.utc).date()
    out: dict = {}
    for tk in tickers:
        try:
            cal = yf.Ticker(tk).calendar
            if cal is None:
                continue
            # cal peut etre dict {'Earnings Date': [date]} ou DataFrame
            earnings_date = None
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                earnings_date = ed[0] if isinstance(ed, list) and ed else ed
            elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
                earnings_date = cal.loc["Earnings Date"].iloc[0]
            if earnings_date is None:
                continue
            if hasattr(earnings_date, "date"):
                earnings_date = earnings_date.date()
            days = (earnings_date - today).days
            if -1 <= days <= 7:
                out[tk] = days
        except Exception:
            continue
    return out


def select_tickers() -> list[str]:
    """
    Choix dynamique des tickers a analyser dans le cycle courant.

    Priorite :
      1. Override manuel via env var LOCUS_TICKERS
      2. TickerDiscoveryEngine (volume anomaly + earnings calendar) sur l'univers
      3. Fallback : FALLBACK_TICKERS (5 leaders)
    """
    if TICKERS_OVERRIDE:
        tickers = [t.strip().upper() for t in TICKERS_OVERRIDE.split() if t.strip()]
        logger.info("[discovery] override LOCUS_TICKERS : %s", tickers)
        return tickers

    try:
        from src.discovery import (
            TickerDiscoveryEngine,
            UniverseFilter,
        )
        from src.discovery.ticker_discovery import (
            EarningsCalendarSource,
            VolumeAnomalySource,
        )

        sources = [
            VolumeAnomalySource(quote_fetcher=lambda: _yfinance_volume_fetcher(CANDIDATE_UNIVERSE)),
            EarningsCalendarSource(calendar_fetcher=lambda: _yfinance_earnings_fetcher(CANDIDATE_UNIVERSE)),
        ]
        engine = TickerDiscoveryEngine(
            sources=sources,
            universe_filter=UniverseFilter(min_market_cap=2e9),
        )
        report = engine.discover(top_n=TOP_N)
        tickers = [s.ticker for s in report.scores] if report.scores else []
        if tickers:
            logger.info(
                "[discovery] %d tickers selectionnes via %s",
                len(tickers), ", ".join(report.sources_used) or "—",
            )
            return tickers
        logger.info("[discovery] aucun ticker scoore — fallback FALLBACK_TICKERS")
    except Exception as e:
        logger.warning("[discovery] echec : %s — fallback FALLBACK_TICKERS", e)

    return list(FALLBACK_TICKERS)


# ----------------------------------------------------------------------------
# Calibration nocturne (une fois par jour)
# ----------------------------------------------------------------------------

def _maybe_run_nightly_calibration(now_dt: datetime) -> None:
    """
    Lance scripts/fit_calibration_nightly.py une fois par jour entre 02:30 et 03:00.
    Utilise un fichier sentinel pour eviter les re-runs apres redemarrage.
    """
    if now_dt.hour != 2 or now_dt.minute < 30:
        return

    today = now_dt.date().isoformat()
    if CALIBRATION_FLAG.exists():
        try:
            last = CALIBRATION_FLAG.read_text(encoding="utf-8").strip()
            if last == today:
                return  # deja fait aujourd'hui
        except OSError:
            pass

    logger.info("[nightly] refit du calibrateur (Platt + Isotonic)...")
    try:
        subprocess.run(
            ["python", "-m", "scripts.fit_calibration_nightly"],
            check=True,
        )
        CALIBRATION_FLAG.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_FLAG.write_text(today, encoding="utf-8")
        logger.info("[nightly] calibration terminee")
    except subprocess.CalledProcessError as e:
        logger.warning("[nightly] calibration echec : %s", e)


# ----------------------------------------------------------------------------
# Audit hebdomadaire (dimanche 23h)
# ----------------------------------------------------------------------------

def _maybe_run_weekly_audit(now_dt: datetime) -> None:
    """Lance audit_hebdomadaire.py le dimanche entre 23h00 et 23h00+INTERVAL."""
    if now_dt.weekday() != 6:  # 6 = dimanche
        return
    if now_dt.hour != 23 or now_dt.minute >= INTERVAL_MINUTES:
        return
    logger.info("[weekly] audit hebdomadaire (counterfactual + event study + regimes)...")
    try:
        subprocess.run(["python", "scripts/audit_hebdomadaire.py"], check=True)
        logger.info("[weekly] audit termine")
    except subprocess.CalledProcessError as e:
        logger.warning("[weekly] audit echec : %s", e)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def print_header() -> None:
    print("=" * 60)
    print(" DEMON PAPER TRADING MULTI-AGENTS")
    print(f" Statut    : ACTIF (DRY_RUN={os.environ['DRY_RUN']})")
    print(f" Univers   : {len(CANDIDATE_UNIVERSE)} candidats S&P 500")
    print(f" Top-N     : {TOP_N} tickers selectionnes par cycle")
    print(f" Intervalle: {INTERVAL_MINUTES} minutes")
    print(f" Budget IA : {os.environ['LLM_DAILY_BUDGET_USD']} USD / jour")
    print(f" Override  : {'oui (' + TICKERS_OVERRIDE + ')' if TICKERS_OVERRIDE else 'non'}")
    print("=" * 60)


def _is_market_open(now_dt: datetime) -> bool:
    """Heures de bourse Paris : Lun-Ven 15h30 a 22h00."""
    if now_dt.weekday() >= 5:
        return False
    if now_dt.hour < 15 or now_dt.hour >= 22:
        return False
    if now_dt.hour == 15 and now_dt.minute < 30:
        return False
    return True


def main() -> None:
    print_header()

    while True:
        now_dt = datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n>>> [{now_str}] NOUVEAU CYCLE")

        tickers = select_tickers()
        tickers_str = " ".join(tickers)

        if _is_market_open(now_dt):
            print(f"[{now_str}] Marche OUVERT — execution complete")
            print(f"[{now_str}] 1/2 Recuperation news pour {len(tickers)} tickers")
            try:
                subprocess.run(
                    f"python -m src.pipelines.news_pipeline --tickers {tickers_str}",
                    shell=True, check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning("Erreur news_pipeline — on continue")

            # Le timer de 30 min demarre ICI (apres les news, pas apres les agents).
            # L'analyse multi-agents peut prendre 15-20 min sur un gros batch,
            # mais les news doivent rester fraiches → on recolte toutes les 30 min.
            cycle_start = time.monotonic()

            print(f"[{now_str}] 2/2 Analyse multi-agents")
            try:
                subprocess.run(
                    "python -m src.pipelines.agent_pipeline",
                    shell=True, check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning("Erreur agent_pipeline — on continue")
        else:
            print(f"[{now_str}] Marche FERME — accumulation news seule")
            try:
                subprocess.run(
                    f"python -m src.pipelines.news_pipeline --tickers {tickers_str}",
                    shell=True, check=True,
                )
            except subprocess.CalledProcessError:
                pass
            cycle_start = time.monotonic()

        # Routines hors-cycle
        _maybe_run_nightly_calibration(now_dt)
        _maybe_run_weekly_audit(now_dt)

        # Sleep = 30 min moins le temps deja ecoule depuis la fin des news.
        elapsed = time.monotonic() - cycle_start
        remaining = max(0, INTERVAL_MINUTES * 60 - elapsed)
        if remaining > 0:
            print(f"\nCycle termine. Prochain dans {remaining / 60:.0f} min. (Ctrl+C pour arreter)")
        else:
            print(f"\nCycle termine. Prochain cycle immediatement (agent a pris {elapsed / 60:.0f} min).")
        try:
            time.sleep(remaining)
        except KeyboardInterrupt:
            print("\nArret manuel demande.")
            break


if __name__ == "__main__":
    main()
