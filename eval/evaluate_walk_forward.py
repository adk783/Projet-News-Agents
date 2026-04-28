"""
evaluate_walk_forward.py — Backtest strict Walk-Forward
========================================================

OBJECTIF
--------
Un Sharpe global de 2.0 calculé sur 5 ans d'historique peut masquer un
performance fortement concentrée sur 1-2 mois chanceux, suivie de 58 mois
de drift vers 0. C'est le biais le plus commun en backtest retail : on
optimise sur un espace de paramètres en voyant *tout* l'historique.

Le **Walk-Forward** (Bailey, de Prado 2014, "The Probability of Backtest
Overfitting") corrige ça en imposant :
  1. On sépare l'historique en fenêtres non-chevauchantes
  2. Sur chaque fenêtre *trade* on mesure la performance **sans jamais**
     avoir vu ces décisions au moment du tuning
  3. On reporte la distribution des Sharpe par fenêtre → on juge la
     **stabilité** du signal, pas juste son meilleur mois

MÉTHODE
-------
Pour chaque fenêtre roulante (par défaut 1 mois calendaire) :
  - Charge les décisions `signal_final ∈ {Achat, Vente}` de la fenêtre
    depuis SQLite (data/news_database.db)
  - Calcule leur CAR[-1,+5] via `compute_event_car` (Fama-French 3-factor,
    réutilisé de `evaluate_event_study.py`)
  - Calcule : n_trades, hit_rate, mean_CAR, std_CAR, Sharpe_annualisé
    (√252 · mean/std pour daily, √12 pour mensuel)

Agrégation cross-window :
  - mean_Sharpe, std_Sharpe, min_Sharpe, max_Sharpe
  - `stability_score` = fraction de fenêtres avec Sharpe > 0
  - `consistency_ratio` = mean/std des Sharpe (plus c'est haut, plus la
    performance est régulière)

RÉFÉRENCES
----------
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  Journal of Portfolio Management, 40(5), 94-107.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014).
  "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
  Overfitting on Out-of-Sample Performance." Notices of the AMS, 61(5).
- López de Prado, M. (2018). "Advances in Financial Machine Learning."
  Chap. 7, Cross-Validation in Finance (walk-forward vs k-fold).
- Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies,"
  2nd ed., Wiley — chap. 5, Walk-Forward Analysis.

USAGE
-----
    python eval/evaluate_walk_forward.py
    python eval/evaluate_walk_forward.py --window-days 30 --min-trades 3
    python eval/evaluate_walk_forward.py --synthetic         # sans DB réelle
    python eval/evaluate_walk_forward.py --output walk_forward.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# Réutilise la plomberie event-study (FF3, fetch prices, etc.)
from eval.evaluate_event_study import (  # noqa: E402
    HAS_YF,
    _generate_synthetic_events,
    _load_decisions_from_db,
    _load_factors_ken_french,
    _load_factors_yfinance_proxy,
    compute_event_car,
)

logger = logging.getLogger("WalkForward")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WindowMetrics:
    """Métriques d'une fenêtre individuelle."""

    start_date: str
    end_date: str
    n_trades: int
    hit_rate: float
    mean_car: float
    std_car: float
    sharpe_ann: float
    win_mean: float = 0.0
    loss_mean: float = 0.0
    pnl_ratio: float = 0.0  # |win_mean| / |loss_mean|


@dataclass
class WalkForwardReport:
    """Rapport agrégé cross-window."""

    n_windows_total: int
    n_windows_active: int  # fenêtres avec ≥ min_trades
    mean_sharpe: float
    std_sharpe: float
    median_sharpe: float
    min_sharpe: float
    max_sharpe: float
    stability_score: float  # % fenêtres Sharpe > 0
    consistency_ratio: float  # mean / std
    mean_hit_rate: float
    mean_trades_per_window: float
    windows: list[WindowMetrics] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Slice par fenêtre
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None


def _group_events_by_window(
    events: list[dict],
    window_days: int = 30,
    stride_days: Optional[int] = None,
) -> list[tuple[datetime, datetime, list[dict]]]:
    """
    Découpe les événements en fenêtres roulantes [start, end[.

    `stride_days` : décalage entre deux fenêtres. None = non-chevauchantes
    (stride = window_days).

    Retourne une liste de (start, end, events_dans_la_fenêtre) triée par date.
    """
    if stride_days is None:
        stride_days = window_days

    dated = [(d, ev) for ev in events if (d := _parse_date(ev.get("date_utc", ""))) is not None]
    if not dated:
        return []
    dated.sort(key=lambda x: x[0])

    first_date = dated[0][0]
    last_date = dated[-1][0]
    windows = []
    cur_start = first_date
    while cur_start <= last_date:
        cur_end = cur_start + timedelta(days=window_days)
        in_window = [ev for d, ev in dated if cur_start <= d < cur_end]
        windows.append((cur_start, cur_end, in_window))
        cur_start = cur_start + timedelta(days=stride_days)
    return windows


# ---------------------------------------------------------------------------
# Calcul des métriques d'une fenêtre
# ---------------------------------------------------------------------------


def _sharpe_annualized(cars: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Sharpe annualisé des CARs. On considère chaque CAR comme un rendement
    de trade. `periods_per_year=252` = approximation "1 trade = 1 jour" ;
    pour des fenêtres mensuelles ça peut être 12, à adapter.

    Robustesse numerique : on traite std < 1e-12 comme zero pour eviter
    les divisions par eps qui produisent des Sharpe absurdes (ex: 1e16) sur
    des series quasi-constantes.
    """
    if len(cars) < 2:
        return 0.0
    m = cars.mean()
    s = cars.std(ddof=1)
    # Seuil numerique : une serie "constante" en pratique a std < 1e-12 a
    # cause des arrondis flottants meme si conceptuellement std == 0.
    if s < 1e-12:
        return 0.0
    return float(m / s * math.sqrt(periods_per_year))


def _window_metrics(
    start: datetime,
    end: datetime,
    events: list[dict],
    factors,
    direction_aware: bool,
) -> Optional[WindowMetrics]:
    """Calcule les métriques d'une fenêtre. None si pas assez d'events."""
    cars = []
    for ev in events:
        sig = ev.get("signal_final", "Neutre")
        if sig not in ("Achat", "Vente"):
            continue
        res = compute_event_car(
            ticker=ev["ticker"],
            event_date=ev["date_utc"],
            signal=sig,
            factors=factors,
            direction_aware=direction_aware,
        )
        if res is not None:
            cars.append(res.car)

    if not cars:
        return WindowMetrics(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            n_trades=0,
            hit_rate=0.0,
            mean_car=0.0,
            std_car=0.0,
            sharpe_ann=0.0,
        )

    arr = np.array(cars)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    win_mean = float(wins.mean()) if wins.size else 0.0
    loss_mean = float(losses.mean()) if losses.size else 0.0
    pnl_ratio = (abs(win_mean) / abs(loss_mean)) if loss_mean != 0 else float("inf")

    return WindowMetrics(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        n_trades=int(arr.size),
        hit_rate=round(float((arr > 0).mean()), 4),
        mean_car=round(float(arr.mean()), 6),
        std_car=round(float(arr.std(ddof=1) if arr.size > 1 else 0.0), 6),
        sharpe_ann=round(_sharpe_annualized(arr), 3),
        win_mean=round(win_mean, 6),
        loss_mean=round(loss_mean, 6),
        pnl_ratio=round(float(pnl_ratio) if pnl_ratio != float("inf") else 0.0, 3),
    )


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


def run_walk_forward(
    events: list[dict],
    window_days: int = 30,
    stride_days: Optional[int] = None,
    min_trades: int = 3,
    direction_aware: bool = True,
    factor_source: str = "yfinance",
) -> WalkForwardReport:
    """
    Fait tourner le walk-forward complet.

    Parameters
    ----------
    events : liste de dicts {ticker, date_utc, signal_final}
    window_days : taille de la fenêtre de trading (jours cal.)
    stride_days : décalage entre fenêtres (None = non-chevauchantes)
    min_trades : fenêtres avec moins = exclues de l'agrégation
    direction_aware : CAR signé par direction (Vente inversée)
    factor_source : "yfinance" (défaut) ou "ken_french"
    """
    if not HAS_YF:
        raise RuntimeError("yfinance est requis pour le walk-forward.")

    # Slice d'abord pour connaître la plage de dates
    windows = _group_events_by_window(events, window_days, stride_days)
    if not windows:
        return WalkForwardReport(
            n_windows_total=0,
            n_windows_active=0,
            mean_sharpe=0.0,
            std_sharpe=0.0,
            median_sharpe=0.0,
            min_sharpe=0.0,
            max_sharpe=0.0,
            stability_score=0.0,
            consistency_ratio=0.0,
            mean_hit_rate=0.0,
            mean_trades_per_window=0.0,
        )

    # On charge les facteurs une seule fois sur la plage globale (+marge FF3)
    global_start = windows[0][0] - timedelta(days=300)
    global_end = windows[-1][1] + timedelta(days=30)
    gs_str = global_start.strftime("%Y-%m-%d")
    ge_str = global_end.strftime("%Y-%m-%d")

    factors = None
    if factor_source == "ken_french":
        factors = _load_factors_ken_french(gs_str, ge_str)
    if factors is None:
        logger.info("Chargement facteurs FF3 (yfinance proxy) [%s, %s]", gs_str, ge_str)
        factors = _load_factors_yfinance_proxy(gs_str, ge_str)

    window_metrics: list[WindowMetrics] = []
    for i, (ws, we, evs) in enumerate(windows, 1):
        wm = _window_metrics(ws, we, evs, factors, direction_aware)
        if wm is not None:
            window_metrics.append(wm)
        if i % 10 == 0:
            logger.info(
                "  Fenêtre %d/%d traitée (active=%d)",
                i,
                len(windows),
                sum(1 for w in window_metrics if w.n_trades >= min_trades),
            )

    # Agrégation : garder seulement fenêtres actives
    active = [w for w in window_metrics if w.n_trades >= min_trades]
    if not active:
        logger.warning(
            "Aucune fenêtre avec ≥ %d trades. Voyez --min-trades ou allongez l'historique.",
            min_trades,
        )
        return WalkForwardReport(
            n_windows_total=len(window_metrics),
            n_windows_active=0,
            mean_sharpe=0.0,
            std_sharpe=0.0,
            median_sharpe=0.0,
            min_sharpe=0.0,
            max_sharpe=0.0,
            stability_score=0.0,
            consistency_ratio=0.0,
            mean_hit_rate=0.0,
            mean_trades_per_window=0.0,
            windows=window_metrics,
            meta={"min_trades": min_trades, "window_days": window_days},
        )

    sharpes = np.array([w.sharpe_ann for w in active])
    hits = np.array([w.hit_rate for w in active])
    ntrades = np.array([w.n_trades for w in active])

    mean_s = float(sharpes.mean())
    std_s = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0
    median_s = float(np.median(sharpes))
    stability = float((sharpes > 0).mean())
    consistency = (mean_s / std_s) if std_s > 0 else 0.0

    return WalkForwardReport(
        n_windows_total=len(window_metrics),
        n_windows_active=len(active),
        mean_sharpe=round(mean_s, 3),
        std_sharpe=round(std_s, 3),
        median_sharpe=round(median_s, 3),
        min_sharpe=round(float(sharpes.min()), 3),
        max_sharpe=round(float(sharpes.max()), 3),
        stability_score=round(stability, 3),
        consistency_ratio=round(consistency, 3),
        mean_hit_rate=round(float(hits.mean()), 4),
        mean_trades_per_window=round(float(ntrades.mean()), 2),
        windows=window_metrics,
        meta={
            "window_days": window_days,
            "stride_days": stride_days or window_days,
            "min_trades": min_trades,
            "direction_aware": direction_aware,
            "factor_source": factor_source,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    )


# ---------------------------------------------------------------------------
# Walk-Forward STRICT avec optimisation OOS de seuil de confiance
# ---------------------------------------------------------------------------
#
# Difference avec run_walk_forward() ci-dessus :
# -- run_walk_forward()        : rolling backtest, chaque fenetre independante,
#                                pas de tuning. Mesure la stabilite du signal.
# -- run_walk_forward_oos()    : pour chaque fenetre, split 70/30 train/test,
#                                optimise le seuil de confiance min sur train,
#                                applique sur test. Mesure la PERFORMANCE OOS,
#                                resistante a l'overfitting (Bailey-Lopez 2014).
#
# Le walk-forward strict OOS detecte le "p-hacking" : si un seuil optimal
# trouve sur train ne fonctionne pas sur test, c'est que le signal est faible.
# ---------------------------------------------------------------------------


@dataclass
class OOSWindowMetrics:
    """Metriques d'une fenetre OOS : sharpe train / sharpe test / seuil retenu."""

    start_date: str
    end_date: str
    split_date: str  # date de la coupure train | test
    n_train: int
    n_test: int
    best_threshold: float  # seuil de confiance min retenu sur train
    sharpe_train: float
    sharpe_test: float
    sharpe_decay: float  # sharpe_train - sharpe_test (ideal proche de 0)


@dataclass
class OOSReport:
    """Agrege le walk-forward OOS strict."""

    n_windows: int
    mean_sharpe_train: float
    mean_sharpe_test: float
    mean_decay: float  # > 0 = overfitting (perf train > test)
    overfitting_score: float  # decay / |sharpe_train|, normalise
    windows: list[OOSWindowMetrics] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _sharpe_from_events_with_threshold(
    events: list[dict],
    factors: object,
    threshold: float,
    direction_aware: bool = True,
) -> float:
    """Calcule le Sharpe d'un sous-ensemble d'evenements filtres par seuil.

    Filtre : on garde seulement les evenements dont la confiance >= threshold.
    Retourne 0.0 si moins de 2 trades passent le seuil.
    """
    cars = []
    for ev in events:
        sig = ev.get("signal_final", "Neutre")
        conf = float(ev.get("consensus_rate", 0.0))
        if sig not in ("Achat", "Vente") or conf < threshold:
            continue
        res = compute_event_car(
            ticker=ev["ticker"],
            event_date=ev["date_utc"],
            signal=sig,
            factors=factors,
            direction_aware=direction_aware,
        )
        if res is not None:
            cars.append(res.car)

    if len(cars) < 2:
        return 0.0
    return _sharpe_annualized(np.array(cars))


def _find_optimal_threshold(
    train_events: list[dict],
    factors: object,
    threshold_grid: list[float] | None = None,
    direction_aware: bool = True,
) -> tuple[float, float]:
    """Grid search sur le seuil de confiance pour maximiser Sharpe(train).

    Returns
    -------
    (best_threshold, sharpe_train)
    """
    if threshold_grid is None:
        threshold_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    best_t = threshold_grid[0]
    best_sharpe = -float("inf")
    for t in threshold_grid:
        s = _sharpe_from_events_with_threshold(train_events, factors, t, direction_aware)
        if s > best_sharpe:
            best_sharpe = s
            best_t = t
    return best_t, best_sharpe if best_sharpe > -float("inf") else 0.0


def run_walk_forward_oos(
    events: list[dict],
    window_days: int = 90,
    train_ratio: float = 0.70,
    threshold_grid: list[float] | None = None,
    direction_aware: bool = True,
    factor_source: str = "yfinance",
    factors: object = None,  # injection optionnelle pour tests
) -> OOSReport:
    """Walk-Forward STRICT avec optimisation OOS du seuil de confiance.

    Pour chaque fenetre [T, T+window_days] :
    1. Split en train (premiers `train_ratio*window_days`) et test (reste).
    2. Grid search du seuil de confiance optimal sur train (max Sharpe).
    3. Applique ce seuil sur test, mesure Sharpe(test).
    4. Reporte le decay = Sharpe(train) - Sharpe(test).

    Un decay systematiquement positif et grand = signal overfitte au train.
    Reference : Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.

    Parameters
    ----------
    events
        Liste de dicts {ticker, date_utc, signal_final, consensus_rate}.
    window_days
        Taille de la fenetre totale (train + test). Defaut 90 (3 mois).
    train_ratio
        Fraction de la fenetre allouee au train. Defaut 0.70 (~63 jours train, ~27 test).
    threshold_grid
        Valeurs de seuil a tester. Defaut [0.50, 0.55, ..., 0.80].
    direction_aware, factor_source
        Idem `run_walk_forward`.
    factors
        Optionnel : injecter des facteurs pre-charges (utile pour tests).

    Returns
    -------
    OOSReport
    """
    windows = _group_events_by_window(events, window_days, stride_days=window_days)
    if not windows:
        return OOSReport(
            n_windows=0,
            mean_sharpe_train=0.0,
            mean_sharpe_test=0.0,
            mean_decay=0.0,
            overfitting_score=0.0,
        )

    # Charger les facteurs FF3 sur la plage globale (1 fois) sauf si injection.
    if factors is None:
        if not HAS_YF:
            raise RuntimeError("yfinance requis pour walk-forward OOS sans factors injectes.")
        global_start = (windows[0][0] - timedelta(days=300)).strftime("%Y-%m-%d")
        global_end = (windows[-1][1] + timedelta(days=30)).strftime("%Y-%m-%d")
        factors = None
        if factor_source == "ken_french":
            factors = _load_factors_ken_french(global_start, global_end)
        if factors is None:
            factors = _load_factors_yfinance_proxy(global_start, global_end)

    oos_metrics: list[OOSWindowMetrics] = []
    for ws, we, evs in windows:
        if len(evs) < 4:  # besoin de >= 2 train + >= 2 test
            continue
        # Tri par date pour split temporel correct.
        evs_sorted = sorted(evs, key=lambda e: e.get("date_utc", ""))
        n_train = int(round(len(evs_sorted) * train_ratio))
        if n_train < 2 or len(evs_sorted) - n_train < 2:
            continue

        train, test = evs_sorted[:n_train], evs_sorted[n_train:]
        split_date = (test[0].get("date_utc", "") or "")[:10]

        best_t, sharpe_train = _find_optimal_threshold(
            train,
            factors,
            threshold_grid,
            direction_aware,
        )
        sharpe_test = _sharpe_from_events_with_threshold(
            test,
            factors,
            best_t,
            direction_aware,
        )

        oos_metrics.append(
            OOSWindowMetrics(
                start_date=ws.strftime("%Y-%m-%d"),
                end_date=we.strftime("%Y-%m-%d"),
                split_date=split_date,
                n_train=len(train),
                n_test=len(test),
                best_threshold=round(best_t, 3),
                sharpe_train=round(sharpe_train, 3),
                sharpe_test=round(sharpe_test, 3),
                sharpe_decay=round(sharpe_train - sharpe_test, 3),
            )
        )

    if not oos_metrics:
        return OOSReport(
            n_windows=0,
            mean_sharpe_train=0.0,
            mean_sharpe_test=0.0,
            mean_decay=0.0,
            overfitting_score=0.0,
            meta={"reason": "aucune fenetre suffisamment peuplee"},
        )

    train_sharpes = np.array([w.sharpe_train for w in oos_metrics])
    test_sharpes = np.array([w.sharpe_test for w in oos_metrics])
    decays = train_sharpes - test_sharpes
    mean_train = float(train_sharpes.mean())
    overfit_score = float(decays.mean() / abs(mean_train)) if abs(mean_train) > 1e-6 else 0.0

    return OOSReport(
        n_windows=len(oos_metrics),
        mean_sharpe_train=round(mean_train, 3),
        mean_sharpe_test=round(float(test_sharpes.mean()), 3),
        mean_decay=round(float(decays.mean()), 3),
        overfitting_score=round(overfit_score, 3),
        windows=oos_metrics,
        meta={
            "window_days": window_days,
            "train_ratio": train_ratio,
            "threshold_grid": threshold_grid or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            "factor_source": factor_source,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Walk-Forward strict Backtest")
    ap.add_argument("--from-db", action="store_true", help="Charge les décisions depuis data/news_database.db")
    ap.add_argument(
        "--synthetic", action="store_true", help="Utilise 500 événements aléatoires (sanity-check plomberie)"
    )
    ap.add_argument("--synthetic-n", type=int, default=500, help="Nombre d'événements synthétiques (défaut 500)")
    ap.add_argument("--window-days", type=int, default=30, help="Taille fenêtre trading en jours cal. (défaut 30)")
    ap.add_argument("--stride-days", type=int, default=None, help="Décalage entre fenêtres (défaut = window-days)")
    ap.add_argument("--min-trades", type=int, default=3, help="Fenêtres < min-trades exclues de l'agrégat (défaut 3)")
    ap.add_argument("--factor-source", choices=["yfinance", "ken_french"], default="yfinance")
    ap.add_argument(
        "--output", type=str, default=None, help="Chemin JSON de sortie (défaut: eval_results/walk_forward.json)"
    )
    args = ap.parse_args()

    if args.synthetic:
        events = _generate_synthetic_events(args.synthetic_n)
        logger.info("Mode synthétique : %d événements générés", len(events))
    elif args.from_db:
        events = _load_decisions_from_db()
        logger.info("Chargé %d décisions depuis SQLite", len(events))
        if not events:
            logger.error("DB vide — utilisez --synthetic pour tester la plomberie")
            return 1
    else:
        events = _load_decisions_from_db()
        if not events:
            logger.info("DB vide ou absente, bascule sur --synthetic")
            events = _generate_synthetic_events(args.synthetic_n)

    report = run_walk_forward(
        events=events,
        window_days=args.window_days,
        stride_days=args.stride_days,
        min_trades=args.min_trades,
        factor_source=args.factor_source,
    )

    out_path = Path(args.output) if args.output else EVAL_RESULTS_DIR / "walk_forward.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info("Rapport écrit : %s", out_path)

    # Résumé console
    print("\n" + "=" * 60)
    print("WALK-FORWARD BACKTEST — RESUME")
    print("=" * 60)
    print(f"Fenêtres totales  : {report.n_windows_total}")
    print(f"Fenêtres actives  : {report.n_windows_active} (>= {args.min_trades} trades)")
    print(f"Sharpe moyen      : {report.mean_sharpe:.3f}  (std {report.std_sharpe:.3f})")
    print(f"Sharpe médian     : {report.median_sharpe:.3f}")
    print(f"Sharpe [min, max] : [{report.min_sharpe:.3f}, {report.max_sharpe:.3f}]")
    print(f"Stability score   : {report.stability_score:.1%}  (% fenêtres Sharpe>0)")
    print(f"Consistency ratio : {report.consistency_ratio:.3f}")
    print(f"Hit-rate moyen    : {report.mean_hit_rate:.2%}")
    print(f"Trades / fenêtre  : {report.mean_trades_per_window:.1f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
