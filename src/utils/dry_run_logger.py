"""
dry_run_logger.py — Journal des ordres "à blanc" (paper trading)

Quand DRY_RUN=1, le pipeline ne persiste pas les décisions dans le
portefeuille réel. À la place, chaque ordre que l'on aurait exécuté est
loggué ici (JSONL append-only), permettant de reconstituer la P&L
théorique et d'évaluer la stratégie sans risque financier.

FORMAT JSONL
------------
Une ligne JSON par ordre, colonnes :
    timestamp, ticker, signal, prix, quantite, montant_eur, sizing_method,
    win_prob, risk_level, market_regime, vix, yield_curve_spread, notes

USAGE
-----
    from src.utils.dry_run_logger import log_dry_run_order
    log_dry_run_order(
        ticker="AAPL", signal="Achat", prix=180.0, quantite=12,
        montant_eur=2160.0, sizing_method="half_kelly",
        win_prob=0.58, risk_level="FAIBLE",
        market_regime="BULL", vix=18.5, yield_curve_spread=0.35,
        notes="DRY_RUN hypothétique",
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from src.config import DRY_RUN_LOG_PATH

logger = logging.getLogger("DryRunLogger")

_lock = Lock()


def log_dry_run_order(
    ticker: str,
    signal: str,
    prix: float,
    quantite: float,
    montant_eur: float,
    sizing_method: str,
    win_prob: float,
    risk_level: str,
    market_regime: Optional[str] = None,
    vix: Optional[float] = None,
    yield_curve_spread: Optional[float] = None,
    notes: str = "",
    extras: Optional[dict] = None,
) -> Path:
    """
    Append une ligne JSONL dans DRY_RUN_LOG_PATH. Thread-safe.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "signal": signal,
        "prix": round(float(prix), 4) if prix else None,
        "quantite": round(float(quantite), 4) if quantite else 0.0,
        "montant_eur": round(float(montant_eur), 2) if montant_eur else 0.0,
        "sizing_method": sizing_method,
        "win_prob": round(float(win_prob), 4) if win_prob is not None else None,
        "risk_level": risk_level,
        "market_regime": market_regime,
        "vix": vix,
        "yield_curve_spread": yield_curve_spread,
        "notes": notes,
    }
    if extras:
        record.update(extras)

    path = Path(DRY_RUN_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    logger.info(
        "[DRY_RUN] %s %s x%s @%s = %s€  (p=%s, risk=%s)",
        signal,
        ticker,
        record["quantite"],
        record["prix"],
        record["montant_eur"],
        record["win_prob"],
        risk_level,
    )
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = log_dry_run_order(
        ticker="AAPL",
        signal="Achat",
        prix=180.0,
        quantite=12,
        montant_eur=2160.0,
        sizing_method="half_kelly",
        win_prob=0.58,
        risk_level="FAIBLE",
        market_regime="BULL",
        vix=18.5,
        yield_curve_spread=0.35,
        notes="smoke-test",
    )
    print("Wrote to:", p)
