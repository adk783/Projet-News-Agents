"""Tests pour le walk-forward (rolling backtest + OOS strict).

Couvre principalement la fonction `run_walk_forward_oos` et ses helpers
(_find_optimal_threshold, _sharpe_from_events_with_threshold). On evite
toute dependance yfinance reelle en injectant un objet `factors` minimal
et en mockant `compute_event_car`.

Le walk-forward OOS implemente la critique de Bailey-Lopez de Prado 2014 :
si Sharpe(train) >> Sharpe(test) systematiquement, le signal est overfitte.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from eval.evaluate_walk_forward import (
    OOSReport,
    OOSWindowMetrics,
    WalkForwardReport,
    WindowMetrics,
    _find_optimal_threshold,
    _group_events_by_window,
    _sharpe_annualized,
    _sharpe_from_events_with_threshold,
    run_walk_forward_oos,
)


# =============================================================================
# Helpers : generation d'evenements synthetiques
# =============================================================================
def _make_events(n: int, start_date: str = "2025-01-01", confidences: list[float] | None = None):
    """Cree n evenements synthetiques etales sur n*3 jours."""
    from datetime import datetime, timedelta

    base = datetime.strptime(start_date, "%Y-%m-%d")
    if confidences is None:
        confidences = [0.7] * n
    return [
        {
            "ticker": "AAPL",
            "date_utc": (base + timedelta(days=i * 3)).strftime("%Y-%m-%d"),
            "signal_final": "Achat" if i % 2 == 0 else "Vente",
            "consensus_rate": confidences[i],
        }
        for i in range(n)
    ]


# =============================================================================
# Tests _sharpe_annualized
# =============================================================================
class TestSharpeAnnualized:
    """Sharpe doit etre nul pour serie courte ou variance nulle, sinon coherent."""

    def test_returns_zero_for_single_value(self):
        import numpy as np

        assert _sharpe_annualized(np.array([0.05])) == 0.0

    def test_returns_zero_for_constant_series(self):
        """Variance nulle -> sharpe = 0 (pas de division par zero)."""
        import numpy as np

        assert _sharpe_annualized(np.array([0.05, 0.05, 0.05])) == 0.0

    def test_positive_mean_yields_positive_sharpe(self):
        import numpy as np

        cars = np.array([0.01, 0.02, 0.03, 0.025, 0.015])
        s = _sharpe_annualized(cars)
        assert s > 0


# =============================================================================
# Tests _group_events_by_window
# =============================================================================
class TestGroupByWindow:
    """Le grouping doit produire des fenetres temporellement coherentes."""

    def test_empty_events_returns_empty(self):
        assert _group_events_by_window([]) == []

    def test_events_without_date_filtered(self):
        events = [{"signal_final": "Achat"}, {"date_utc": "invalid"}]
        assert _group_events_by_window(events) == []

    def test_window_size_30_groups_correctly(self):
        events = _make_events(20)  # 20 evts sur 60 jours
        windows = _group_events_by_window(events, window_days=30)
        # 60 jours / 30 jours stride = 2 fenetres au moins
        assert len(windows) >= 2
        # Les 10 premiers events sont dans la 1ere fenetre
        assert len(windows[0][2]) >= 8

    def test_stride_smaller_than_window_overlaps(self):
        """stride < window -> chevauchement."""
        events = _make_events(10)
        wins_no_overlap = _group_events_by_window(events, window_days=15, stride_days=15)
        wins_overlap = _group_events_by_window(events, window_days=15, stride_days=5)
        assert len(wins_overlap) > len(wins_no_overlap)


# =============================================================================
# Tests _find_optimal_threshold
# =============================================================================
class TestFindOptimalThreshold:
    """Grid search doit retourner le seuil maximisant Sharpe sur train."""

    def test_returns_best_from_grid(self):
        """Cas synthetique : on patch _sharpe_from_events_with_threshold pour
        controler la valeur retournee a chaque seuil."""
        events = _make_events(5)
        factors = MagicMock()

        # Plus le seuil est haut, mieux c'est (best = 0.80)
        sharpe_by_threshold = {0.50: 0.5, 0.60: 0.8, 0.70: 1.2, 0.80: 1.5}

        def fake_sharpe(_evs, _facs, t, _direction_aware=True):
            return sharpe_by_threshold[t]

        with patch("eval.evaluate_walk_forward._sharpe_from_events_with_threshold", side_effect=fake_sharpe):
            best_t, best_s = _find_optimal_threshold(
                events,
                factors,
                threshold_grid=[0.50, 0.60, 0.70, 0.80],
            )
            assert best_t == 0.80
            assert best_s == 1.5

    def test_default_grid_used_when_none(self):
        events = _make_events(5)
        factors = MagicMock()

        with patch("eval.evaluate_walk_forward._sharpe_from_events_with_threshold", return_value=0.5):
            best_t, _ = _find_optimal_threshold(events, factors, threshold_grid=None)
            # Default grid contient 0.50 -> retournera ce premier
            assert best_t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


# =============================================================================
# Tests _sharpe_from_events_with_threshold
# =============================================================================
class TestSharpeWithThreshold:
    """Le filtrage par seuil de confiance doit fonctionner."""

    def test_threshold_filters_low_confidence_events(self):
        """Si seuil > toutes les confiances, on doit avoir 0 trade -> sharpe=0."""
        events = _make_events(5, confidences=[0.4, 0.3, 0.5, 0.45, 0.35])
        factors = MagicMock()
        # Aucun event ne passe seuil 0.80
        s = _sharpe_from_events_with_threshold(events, factors, threshold=0.80)
        assert s == 0.0

    def test_neutre_signal_filtered(self):
        """Signaux 'Neutre' sont ignores meme si confiance haute."""
        events = [
            {"ticker": "AAPL", "date_utc": "2025-01-01", "signal_final": "Neutre", "consensus_rate": 0.95},
            {"ticker": "MSFT", "date_utc": "2025-01-02", "signal_final": "Neutre", "consensus_rate": 0.90},
        ]
        factors = MagicMock()
        s = _sharpe_from_events_with_threshold(events, factors, threshold=0.5)
        assert s == 0.0

    def test_sharpe_computed_when_compute_event_car_returns_results(self):
        """Avec mock retournant des CAR positifs, sharpe doit etre > 0."""
        events = _make_events(5, confidences=[0.7] * 5)
        factors = MagicMock()

        # Mock compute_event_car pour retourner des CAR positifs constants.
        fake_result = MagicMock()
        fake_result.car = 0.02  # 2%

        with patch("eval.evaluate_walk_forward.compute_event_car", return_value=fake_result):
            s = _sharpe_from_events_with_threshold(events, factors, threshold=0.5)
            # 5 CARs identiques -> std=0 -> sharpe=0 (par construction)
            assert s == 0.0

    def test_sharpe_positive_with_varied_returns(self):
        events = _make_events(6, confidences=[0.7] * 6)
        factors = MagicMock()

        # Returns varies pour avoir une vraie distribution
        fake_cars = [0.01, 0.02, 0.015, 0.025, 0.005, 0.018]
        fake_results = [MagicMock(car=c) for c in fake_cars]

        with patch("eval.evaluate_walk_forward.compute_event_car", side_effect=fake_results):
            s = _sharpe_from_events_with_threshold(events, factors, threshold=0.5)
            assert s > 0


# =============================================================================
# Tests run_walk_forward_oos (integration)
# =============================================================================
class TestRunWalkForwardOOS:
    """Integration : le walk-forward OOS produit un OOSReport coherent."""

    def test_no_events_returns_empty_report(self):
        rep = run_walk_forward_oos([], window_days=30, factors=MagicMock())
        assert isinstance(rep, OOSReport)
        assert rep.n_windows == 0
        assert rep.mean_sharpe_train == 0.0

    def test_too_few_events_per_window_returns_empty(self):
        """Avec 2 events seulement, aucune fenetre ne peut faire train+test (>= 4)."""
        events = _make_events(2)
        rep = run_walk_forward_oos(events, window_days=90, factors=MagicMock())
        assert rep.n_windows == 0

    def test_full_oos_run_produces_metrics_per_window(self):
        """Avec 30 events et 1 fenetre, on doit avoir 1 window OOSWindowMetrics."""
        events = _make_events(30, confidences=[0.7] * 30)
        factors = MagicMock()

        # Mock compute_event_car : returns positifs et varies
        fake_cars = [0.01 + 0.005 * (i % 3) for i in range(60)]
        results_iter = (MagicMock(car=c) for c in fake_cars)
        results_list = list(results_iter)

        with patch("eval.evaluate_walk_forward.compute_event_car", side_effect=results_list * 5):  # large pool
            rep = run_walk_forward_oos(
                events,
                window_days=200,
                train_ratio=0.7,
                threshold_grid=[0.5, 0.6, 0.7],
                factors=factors,
            )

        assert rep.n_windows >= 1
        for w in rep.windows:
            assert isinstance(w, OOSWindowMetrics)
            assert w.n_train >= 2
            assert w.n_test >= 2
            assert w.best_threshold in [0.5, 0.6, 0.7]
            assert w.split_date  # date non vide

    def test_decay_computed_correctly(self):
        """sharpe_decay = sharpe_train - sharpe_test (verifie sur 1 window)."""
        events = _make_events(10, confidences=[0.7] * 10)
        factors = MagicMock()

        with (
            patch("eval.evaluate_walk_forward._find_optimal_threshold", return_value=(0.6, 1.5)),
            patch("eval.evaluate_walk_forward._sharpe_from_events_with_threshold", return_value=0.5),
        ):
            rep = run_walk_forward_oos(
                events,
                window_days=200,
                factors=factors,
            )
            assert rep.n_windows >= 1
            for w in rep.windows:
                assert w.sharpe_train == 1.5
                assert w.sharpe_test == 0.5
                assert w.sharpe_decay == round(1.5 - 0.5, 3)

    def test_overfitting_score_positive_when_decay_positive(self):
        """Si train >> test (decay > 0), overfitting_score > 0."""
        events = _make_events(20, confidences=[0.7] * 20)
        factors = MagicMock()

        with (
            patch("eval.evaluate_walk_forward._find_optimal_threshold", return_value=(0.6, 2.0)),
            patch("eval.evaluate_walk_forward._sharpe_from_events_with_threshold", return_value=0.0),
        ):
            rep = run_walk_forward_oos(
                events,
                window_days=300,
                factors=factors,
            )
            assert rep.mean_decay > 0
            assert rep.overfitting_score > 0


# =============================================================================
# Tests dataclasses (sanity)
# =============================================================================
class TestDataclasses:
    """Les dataclasses ont les champs attendus."""

    def test_oos_window_metrics_has_required_fields(self):
        m = OOSWindowMetrics(
            start_date="2025-01-01",
            end_date="2025-04-01",
            split_date="2025-03-01",
            n_train=10,
            n_test=5,
            best_threshold=0.6,
            sharpe_train=1.5,
            sharpe_test=0.8,
            sharpe_decay=0.7,
        )
        assert m.n_train == 10
        assert m.sharpe_decay == 0.7

    def test_oos_report_default_empty_windows_list(self):
        r = OOSReport(n_windows=0, mean_sharpe_train=0.0, mean_sharpe_test=0.0, mean_decay=0.0, overfitting_score=0.0)
        assert r.windows == []

    def test_window_metrics_has_required_fields(self):
        """Sanity check sur la dataclass de la version rolling (pas OOS)."""
        wm = WindowMetrics(
            start_date="2025-01-01",
            end_date="2025-01-30",
            n_trades=5,
            hit_rate=0.6,
            mean_car=0.01,
            std_car=0.02,
            sharpe_ann=1.5,
        )
        assert wm.n_trades == 5

    def test_walk_forward_report_default_empty_windows(self):
        r = WalkForwardReport(
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
        assert r.windows == []
