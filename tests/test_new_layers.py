"""Tests unitaires pytest-natifs pour les couches 10-12 de la matrice d'evaluation.

Migration : ce fichier utilisait auparavant un systeme PASS/FAIL maison +
sys.exit(), ce qui le rendait invisible pour `pytest tests/`. Refactor en
pytest natif (assert + fixtures) pour qu'il participe a la CI.

Couches couvertes :
- 10 : Execution costs (cheap mode + Almgren-Chriss)
- 11 : Latency / alpha decay (fit exponentiel)
- 12 : Market regimes (BULL/BEAR/HIGH_VOL/SIDEWAYS/UNKNOWN)
"""

from __future__ import annotations

import pytest

from eval.evaluate_execution_costs import (
    BROKER_PROFILES,
    compute_annualized_turnover,
    compute_breakeven_accuracy,
    compute_execution_cost,
)
from eval.evaluate_latency import (
    DELAY_BUCKETS,
    HOUR_BUCKETS,
    fit_alpha_decay,
)
from eval.evaluate_market_regimes import (
    REGIME_LABELS,  # noqa: F401
    REGIME_THRESHOLDS,  # noqa: F401  (re-export verifie indirectement)
    _classify_regime,
    _compute_regime_metrics,
    _compute_robustness_metrics,
)


# =============================================================================
# Couche 10 : Execution Costs
# =============================================================================
class TestExecutionCosts:
    """Couche 10 : couts d'execution (commissions, spread, impact)."""

    def test_retail_cost_strictly_positive(self):
        """Un trade RETAIL doit toujours generer un cout > 0."""
        cost = compute_execution_cost(
            price=150.0,
            position_eur=5000.0,
            broker=BROKER_PROFILES["RETAIL"],
            signal="Achat",
        )
        assert cost["total_roundtrip_pct"] > 0

    def test_retail_more_expensive_than_pro(self):
        """Le profil RETAIL doit etre plus cher que PRO (commissions superieures)."""
        cost_retail = compute_execution_cost(
            price=150.0,
            position_eur=5000.0,
            broker=BROKER_PROFILES["RETAIL"],
            signal="Achat",
        )
        cost_pro = compute_execution_cost(
            price=150.0,
            position_eur=5000.0,
            broker=BROKER_PROFILES["PRO"],
            signal="Achat",
        )
        assert cost_retail["total_roundtrip_pct"] > cost_pro["total_roundtrip_pct"]

    def test_neutre_signal_zero_cost(self):
        """Signal Neutre = pas de trade = pas de cout."""
        cost = compute_execution_cost(
            price=150.0,
            position_eur=5000.0,
            broker=BROKER_PROFILES["RETAIL"],
            signal="Neutre",
        )
        assert cost["total_roundtrip_pct"] == 0.0

    def test_breakeven_accuracy_in_unit_interval(self):
        """Break-even accuracy doit etre dans (0, 1) pour des inputs realistes."""
        cost = compute_execution_cost(
            price=150.0,
            position_eur=5000.0,
            broker=BROKER_PROFILES["RETAIL"],
            signal="Achat",
        )
        be = compute_breakeven_accuracy(
            avg_win_pct=2.5,
            avg_loss_pct=1.5,
            total_cost_pct=cost["total_roundtrip_pct"],
        )
        assert 0.0 < be["breakeven_accuracy"] < 1.0
        assert be["payoff_ratio_net"] is not None
        assert be["payoff_ratio_net"] > 0

    def test_annualized_turnover_positive(self):
        """Turnover annualise > 0 pour 50 trades sur 20j de holding."""
        to = compute_annualized_turnover(n_trades=50, avg_holding_days=20)
        assert to["turnover_annual"] is not None
        assert to["trades_per_year"] > 0


# =============================================================================
# Couche 11 : Latency / Alpha Decay
# =============================================================================
class TestAlphaDecay:
    """Couche 11 : decay du signal (fit exponentiel sur la latence)."""

    def test_exponential_fit_succeeds_with_4_points(self):
        """Fit exponentiel doit reussir avec >= 4 points."""
        result = fit_alpha_decay([1.0, 3.0, 10.0, 30.0], [3.2, 2.1, 1.1, 0.4])
        assert result.get("fitted") is True
        assert result.get("half_life_hours", 0) > 0

    def test_fit_fails_with_insufficient_points(self):
        """Fit doit echouer proprement (fitted=False) si N < 4."""
        result = fit_alpha_decay([1.0], [2.0])
        assert result.get("fitted") is False

    def test_canonical_buckets_count(self):
        """6 buckets horaires + 6 buckets de delai (constantes du module)."""
        assert len(HOUR_BUCKETS) == 6
        assert len(DELAY_BUCKETS) == 6


# =============================================================================
# Couche 12 : Market Regimes
# =============================================================================
class TestMarketRegimes:
    """Couche 12 : classification du regime de marche (VIX + SPY return)."""

    @pytest.mark.parametrize(
        "vix,spy_ret,expected",
        [
            (35.0, 5.0, "HIGH_VOL"),  # VIX > 30 -> HIGH_VOL prioritaire
            (15.0, 6.0, "BULL"),  # SPY > 3% et VIX < 25
            (28.0, -5.0, "BEAR"),  # SPY < -3% et VIX > 20
            (18.0, 1.0, "SIDEWAYS"),  # ni bull ni bear ni high-vol
            (None, None, "UNKNOWN"),  # data manquante
            (35.0, 8.0, "HIGH_VOL"),  # priorite HIGH_VOL meme si SPY positif
        ],
    )
    def test_regime_classification(self, vix, spy_ret, expected):
        """Test parametrique de la matrice de classification."""
        assert _classify_regime(vix, spy_ret) == expected

    @pytest.fixture
    def mock_trades(self):
        """Jeu de trades synthetiques mixant signaux actionnables et neutres."""
        return [
            {
                "signal": "Achat",
                "return_pct": 2.5,
                "signal_correct": True,
                "actionable": True,
                "spy_return": 1.0,
                "risk_level": "FAIBLE",
            },
            {
                "signal": "Vente",
                "return_pct": -1.0,
                "signal_correct": False,
                "actionable": True,
                "spy_return": 0.5,
                "risk_level": "MOYEN",
            },
            {
                "signal": "Achat",
                "return_pct": 1.8,
                "signal_correct": True,
                "actionable": True,
                "spy_return": 0.8,
                "risk_level": "FAIBLE",
            },
            {
                "signal": "Neutre",
                "return_pct": 0.0,
                "signal_correct": None,
                "actionable": False,
                "spy_return": 0.2,
                "risk_level": "FAIBLE",
            },
        ]

    def test_regime_metrics_compute_sharpe_winrate_dd(self, mock_trades):
        """Metriques cles : n_active, sharpe, win_rate, max_drawdown."""
        m = _compute_regime_metrics(mock_trades)
        assert m.get("n_active", 0) > 0
        assert m.get("sharpe") is not None
        assert 0.0 <= m.get("win_rate", -1) <= 1.0
        assert m.get("max_drawdown", -1) >= 0

    def test_robustness_identifies_best_and_worst_regimes(self):
        """Robustesse doit ranker les regimes par Sharpe (best/worst)."""
        fake_results = {
            "BULL": {"sharpe": 1.5, "n_active": 10},
            "BEAR": {"sharpe": 0.3, "n_active": 5},
            "HIGH_VOL": {"sharpe": 0.8, "n_active": 3},
        }
        rob = _compute_robustness_metrics(fake_results)
        assert rob.get("best_regime") == "BULL"
        assert rob.get("worst_regime") == "BEAR"
        assert rob.get("sharpe_range") is not None
        assert rob.get("bull_bear_ratio") is not None
        assert rob.get("verdict") is not None
