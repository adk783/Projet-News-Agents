"""
audit_hebdomadaire.py — Audit hors-ligne du pipeline (à lancer le dimanche)

OBJECTIF
--------
Regrouper tous les audits coûteux (appels LLM, bootstrap, Romano-Wolf) qui
NE DOIVENT PAS tourner dans le pipeline live, mais qui sont essentiels au
suivi de la qualité de la stratégie.

Les audits live (calibration, Bayesian, YOLO, position-sizing) tournent
à chaque article en quelques millisecondes. Ce script consolide ce qui
est trop lourd pour le live :
  1. Counterfactual Invariance (sur les 50 pires trades de la semaine)
  2. Event Study Fama-French (CAR[-1,+5] + Newey-West + Romano-Wolf)
  3. Calibration fit (Platt + Isotonic) sur les trades fermés
  4. Régime analysis (distribution des régimes + accuracy par régime)
  5. Rapport consolidé HTML/Markdown

PLANIFICATION PROPOSÉE
----------------------
Windows Task Scheduler :
    Dimanche 03:00 → python -m scripts.audit_hebdomadaire
Linux cron :
    0 3 * * 0 cd /path/to/project && python -m scripts.audit_hebdomadaire

BUDGET API (estimé)
-------------------
- Counterfactual Invariance : 50 articles × 11 variants × 4 LLMs ≈ 2200 appels
  → avec Groq/Cerebras free-tier : ≈ 30 min, ~$3 à $5
- Event Study : yfinance only (gratuit), ~5 min pour 1 an d'historique
- Calibration : aucun appel LLM, ~10 s
→ **Total ≈ 30-40 min, ~$5/semaine**
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Imports internes

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("AuditHebdo")

DATABASE_PATH = "data/news_database.db"
REPORT_DIR = Path("reports/weekly_audit")


# ---------------------------------------------------------------------------
# 1. Sélection des "pires trades" de la semaine pour Counterfactual
# ---------------------------------------------------------------------------


def select_worst_trades(db_path: str, n: int = 50) -> list[dict]:
    """
    Retourne les N trades les plus problématiques de la semaine écoulée :
      - Signal à faible confidence
      - Divergence FinBERT vs Consensus
      - Forte volatilité post-trade

    Args:
        db_path : chemin SQLite
        n       : nombre de trades (default 50)
    """
    if not Path(db_path).exists():
        logger.warning("DB absente : %s", db_path)
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    try:
        cur.execute(
            """
            SELECT url, ticker, title, content, signal_final, argument_dominant,
                   score_filtrage, date
            FROM articles
            WHERE signal_final IN ('Achat', 'Vente')
              AND date >= ?
            ORDER BY score_filtrage ASC
            LIMIT ?
            """,
            (week_ago, n),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        logger.error("Query failed : %s", e)
        rows = []
    finally:
        conn.close()
    logger.info("Selected %d worst trades from last 7 days", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 2. Runner des différents audits
# ---------------------------------------------------------------------------


def run_counterfactual_audit(
    worst_trades: list[dict],
    use_pipeline: bool,
) -> dict:
    """
    Lance le Counterfactual Invariance sur les pires trades.
    Si use_pipeline=False, mode MSS-only (gratuit, pas d'appel LLM).
    """
    from src.utils.counterfactual import compute_ci_score

    if use_pipeline:
        try:
            from src.pipelines.agent_pipeline import run_pipeline_single  # type: ignore

            pipeline_fn = lambda t, tk: {"signal": run_pipeline_single(t, tk)}  # noqa
        except Exception as e:
            logger.warning("Fallback MSS-only : %s", e)
            pipeline_fn = None
    else:
        pipeline_fn = None

    ci_scores, anomalies = [], []
    for tr in worst_trades:
        report = compute_ci_score(
            text=tr.get("content", "")[:4000],
            original_ticker=(tr.get("ticker") or "AAPL").upper(),
            pipeline_fn=pipeline_fn,
        )
        ci_scores.append(report.ci_score)
        anomalies.extend(report.anomalies[:2])
    return {
        "n": len(ci_scores),
        "ci_mean": round(sum(ci_scores) / len(ci_scores), 4) if ci_scores else None,
        "ci_min": round(min(ci_scores), 4) if ci_scores else None,
        "ci_below_0.75": sum(1 for s in ci_scores if s < 0.75),
        "worst_anomalies_sample": anomalies[:10],
    }


def run_event_study_audit(limit: int = 200) -> dict:
    """Event Study + Romano-Wolf sur l'historique récent."""
    try:
        from eval.evaluate_event_study import run_event_study
    except ImportError as e:
        logger.warning("evaluate_event_study indisponible : %s", e)
        return {"error": str(e)}
    try:
        return run_event_study(limit=limit, from_db=True) or {}
    except Exception as e:
        logger.error("Event study failed : %s", e)
        return {"error": str(e)}


def run_pnl_backtest_audit() -> dict:
    """Backtest P&L classique (Sharpe, Drawdown, courbe de capital)."""
    try:
        from eval.evaluate_historical_backtest import run_historical_backtest

        run_historical_backtest()
        return {"status": "ok — see stdout for Sharpe/Drawdown/P&L curve"}
    except Exception as e:
        logger.error("P&L backtest failed : %s", e)
        return {"error": str(e)}


def run_calibration_audit() -> dict:
    """Fit Platt + Isotonic sur les trades clôturés."""
    try:
        from eval.evaluate_calibration import run as run_cal  # type: ignore

        return run_cal() or {"status": "ok"}
    except Exception as e:
        logger.warning("Calibration eval unavailable : %s", e)
        return {"error": str(e)}


def run_regime_audit() -> dict:
    """Accuracy par régime SIDEWAYS/BULL/BEAR/HIGH_VOL."""
    try:
        from eval.evaluate_market_regimes import main as run_reg  # type: ignore

        return run_reg() or {"status": "ok"}
    except Exception as e:
        logger.warning("Regime eval unavailable : %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 3. Orchestrateur + rapport
# ---------------------------------------------------------------------------


def main(
    worst_n: int = 50,
    cf_use_pipeline: bool = False,
    skip_cf: bool = False,
    skip_es: bool = False,
    skip_pnl: bool = False,
    skip_cal: bool = False,
    skip_reg: bool = False,
) -> dict:
    logger.info("=" * 70)
    logger.info("AUDIT HEBDOMADAIRE — %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("=" * 70)

    report: dict = {"generated_at": datetime.now().isoformat()}

    if not skip_cf:
        logger.info("[1/5] Counterfactual Invariance (worst %d trades)...", worst_n)
        worst = select_worst_trades(DATABASE_PATH, n=worst_n)
        try:
            report["counterfactual"] = run_counterfactual_audit(worst, cf_use_pipeline)
        except Exception:
            report["counterfactual"] = {"error": traceback.format_exc(limit=3)}

    if not skip_es:
        logger.info("[2/5] Event Study Fama-French...")
        try:
            report["event_study"] = run_event_study_audit(limit=200)
        except Exception:
            report["event_study"] = {"error": traceback.format_exc(limit=3)}

    if not skip_pnl:
        logger.info("[3/5] Backtest P&L classique (Sharpe + Drawdown)...")
        try:
            report["pnl_backtest"] = run_pnl_backtest_audit()
        except Exception:
            report["pnl_backtest"] = {"error": traceback.format_exc(limit=3)}

    if not skip_cal:
        logger.info("[4/5] Calibration (ECE + Brier)...")
        try:
            report["calibration"] = run_calibration_audit()
        except Exception:
            report["calibration"] = {"error": traceback.format_exc(limit=3)}

    if not skip_reg:
        logger.info("[5/5] Régime analysis...")
        try:
            report["regime"] = run_regime_audit()
        except Exception:
            report["regime"] = {"error": traceback.format_exc(limit=3)}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"audit_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Rapport sauvegarde : %s", out)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit hebdomadaire offline")
    parser.add_argument("--worst-n", type=int, default=50)
    parser.add_argument(
        "--cf-pipeline", action="store_true", help="Active les appels LLM dans le Counterfactual (coûteux)."
    )
    parser.add_argument("--skip-cf", action="store_true")
    parser.add_argument("--skip-es", action="store_true")
    parser.add_argument("--skip-pnl", action="store_true")
    parser.add_argument("--skip-cal", action="store_true")
    parser.add_argument("--skip-reg", action="store_true")
    args = parser.parse_args()
    main(
        worst_n=args.worst_n,
        cf_use_pipeline=args.cf_pipeline,
        skip_cf=args.skip_cf,
        skip_es=args.skip_es,
        skip_pnl=args.skip_pnl,
        skip_cal=args.skip_cal,
        skip_reg=args.skip_reg,
    )
