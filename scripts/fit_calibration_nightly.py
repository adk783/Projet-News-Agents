"""
fit_calibration_nightly.py — Refit nocturne de la calibration (Platt/Isotonic)

OBJECTIF
--------
Prendre l'historique (p_raw, outcome) des trades clôturés et apprendre une
fonction g: p_raw -> p_calibrated qui corrige la sur/sous-confiance des
LLMs. À tourner chaque nuit — le pipeline live charge le résultat au
démarrage.

PIPELINE
--------
  1. Lire les articles avec signal_final in (Achat, Vente) ET un outcome
     observable (variation de prix >= date_utc + horizon_days).
  2. Pour chaque trade : extraire (impact_strength, outcome∈{0,1}) où
     outcome=1 si (signal=Achat ∧ return>0) ou (signal=Vente ∧ return<0).
  3. Fit Platt + Isotonic, choisir le meilleur par Brier (helper existant).
  4. Sauvegarde : models/calibrator.pkl (+ metadata JSON à côté).

LE PIPELINE LIVE charge ensuite :
    cal = pickle.load(open("models/calibrator.pkl", "rb"))
    p_calibrated = cal.transform([impact_strength])[0]

PLANIFICATION
-------------
Windows Task Scheduler :
    Tous les jours 02:30 → python -m scripts.fit_calibration_nightly
Linux cron :
    30 2 * * * cd /proj && python -m scripts.fit_calibration_nightly

RÉFÉRENCES
----------
  Platt (1999), Zadrozny & Elkan (2002), Guo et al. (2017)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.config import DATABASE_PATH, ECE_TARGET_POST_CALIBRATION
from src.utils.calibration import (
    brier_score,
    expected_calibration_error,
    fit_best_calibrator,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CalibrationNightly")

MODEL_OUTPUT = Path("models/calibrator.pkl")
METADATA_OUTPUT = Path("models/calibrator.meta.json")
DEFAULT_HORIZON_D = 5  # fenêtre pour mesurer l'outcome
MIN_TRADES_REQUIRED = 30  # refuse de fit avec moins que ça (sur-ajustement)


# ---------------------------------------------------------------------------
# Collecte dataset (p_raw, y_true)
# ---------------------------------------------------------------------------


def _outcome_for_trade(ticker: str, date_iso: str, signal: str, horizon_days: int) -> Optional[int]:
    """
    Retourne 1 si le trade a "gagné" sur `horizon_days` jours, 0 sinon,
    None si on ne peut pas déterminer (pas de données yfinance).

    Règle :
      - Achat : return > 0  → y=1
      - Vente : return < 0  → y=1
      - sinon y=0
    """
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance indispo — impossible de calculer les outcomes.")
        return None

    try:
        start = datetime.fromisoformat(date_iso.replace("Z", ""))
    except Exception:
        return None
    end = start + timedelta(days=max(horizon_days + 5, 10))  # buffer week-end

    try:
        hist = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.debug("yf.download fail %s : %s", ticker, exc)
        return None

    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None

    price_in = float(closes.iloc[0])
    # Prix à T+horizon (ou dernier dispo si week-end / férié)
    idx_target = min(horizon_days, len(closes) - 1)
    price_out = float(closes.iloc[idx_target])
    ret = (price_out - price_in) / price_in

    if signal == "Achat":
        return 1 if ret > 0 else 0
    if signal == "Vente":
        return 1 if ret < 0 else 0
    return None  # signal Hold → non scorable


def collect_dataset(
    db_path: str = DATABASE_PATH,
    horizon_days: int = DEFAULT_HORIZON_D,
    lookback_days: int = 90,
) -> tuple[list[float], list[int], list[dict]]:
    """
    Parcourt la DB, appelle yfinance pour calculer l'outcome.
    Retourne (probs, labels, trades_meta).
    """
    if not Path(db_path).exists():
        logger.error("DB absente : %s", db_path)
        return [], [], []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # On ne prend que les trades assez vieux pour avoir un outcome observable
    threshold = (datetime.now(timezone.utc) - timedelta(days=horizon_days + 1)).isoformat()
    lookback = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    try:
        cur.execute(
            """
            SELECT url, ticker, signal_final, impact_strength, date_utc
            FROM articles
            WHERE signal_final IN ('Achat', 'Vente')
              AND impact_strength IS NOT NULL
              AND date_utc < ?
              AND date_utc >= ?
            ORDER BY date_utc DESC
            LIMIT 2000
            """,
            (threshold, lookback),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("Query échouée : %s", exc)
        rows = []
    finally:
        conn.close()

    probs: list[float] = []
    labels: list[int] = []
    meta: list[dict] = []
    for r in rows:
        outcome = _outcome_for_trade(
            ticker=(r["ticker"] or "").upper(),
            date_iso=r["date_utc"] or "",
            signal=r["signal_final"],
            horizon_days=horizon_days,
        )
        if outcome is None:
            continue
        p = float(r["impact_strength"])
        if not (0.0 <= p <= 1.0):
            continue
        probs.append(p)
        labels.append(int(outcome))
        meta.append({"url": r["url"], "ticker": r["ticker"], "signal": r["signal_final"]})

    logger.info("Collecte : %d trades scorables (horizon %dj)", len(probs), horizon_days)
    return probs, labels, meta


# ---------------------------------------------------------------------------
# Fit + persistance
# ---------------------------------------------------------------------------


def fit_and_save(
    probs: list[float],
    labels: list[int],
) -> dict:
    """
    Fit Platt + Isotonic, choisit le meilleur, persiste en pickle.
    Retourne un dict de métadonnées pour le rapport.
    """
    if len(probs) < MIN_TRADES_REQUIRED:
        msg = f"Dataset trop petit ({len(probs)} < {MIN_TRADES_REQUIRED}) — skip"
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "n": len(probs)}

    cal, info = fit_best_calibrator(probs, labels)
    p_cal = cal.transform(probs)

    ece_before = expected_calibration_error(probs, labels)
    ece_after = expected_calibration_error(p_cal, labels)
    brier_before = brier_score(probs, labels)
    brier_after = brier_score(p_cal, labels)

    # Persistance
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(cal, f)

    meta = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(probs),
        "positive_rate": round(sum(labels) / len(labels), 4),
        "chosen": info.get("chosen"),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "ece_target": ECE_TARGET_POST_CALIBRATION,
        "ece_target_met": bool(ece_after <= ECE_TARGET_POST_CALIBRATION),
        "info": info,
    }
    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Calibrator sauvé : %s (%s)", MODEL_OUTPUT, info.get("chosen"))
    logger.info("  ECE   : %.4f -> %.4f  (cible %.3f)", ece_before, ece_after, ECE_TARGET_POST_CALIBRATION)
    logger.info("  Brier : %.4f -> %.4f", brier_before, brier_after)
    return meta


# ---------------------------------------------------------------------------
# Helper : chargement côté pipeline live
# ---------------------------------------------------------------------------


def load_calibrator():
    """
    Utilisé par le pipeline live au démarrage pour charger le calibrator.
    Retourne None si aucun modèle fit n'est disponible (fallback identité).
    """
    if not MODEL_OUTPUT.exists():
        return None
    try:
        with open(MODEL_OUTPUT, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Échec chargement calibrator : %s", exc)
        return None


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------


def main(
    horizon_days: int = DEFAULT_HORIZON_D,
    lookback_days: int = 90,
) -> dict:
    logger.info("=" * 70)
    logger.info("NIGHTLY CALIBRATION REFIT — %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("=" * 70)
    probs, labels, _meta = collect_dataset(
        DATABASE_PATH,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
    )
    return fit_and_save(probs, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_D)
    parser.add_argument("--lookback-days", type=int, default=90)
    args = parser.parse_args()
    result = main(horizon_days=args.horizon_days, lookback_days=args.lookback_days)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
