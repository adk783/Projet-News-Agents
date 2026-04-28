"""
test_phase3_modules.py — Tests unitaires des modules Phase 3 (pure-python)

Ces tests ne nécessitent NI clé API LLM, NI accès réseau (tout yfinance/LLM
est mocké ou évité). Ils tournent en CI en <5 s.

Portée :
  - src/utils/rate_limiter.py        : backoff + détection 429
  - src/utils/llm_cost_tracker.py    : compteur + budget
  - src/utils/dry_run_logger.py      : JSONL append-only
  - src/strategy/portfolio_constraints.py : cap sectoriel + cap correlation
  - src/strategy/position_sizer.py   : guards HOLD_SYSTEMIC / HOLD_SECTOR_CAP / HOLD_CORR_CAP
"""

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_retries_on_429():
    from src.utils.rate_limiter import with_backoff

    counter = {"n": 0}

    class FakeRateLimit(Exception):
        def __init__(self):
            super().__init__("429 Too Many Requests")

    @with_backoff(max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False)
    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise FakeRateLimit()
        return "OK"

    assert flaky() == "OK"
    assert counter["n"] == 3


def test_rate_limiter_does_not_retry_value_error():
    from src.utils.rate_limiter import with_backoff

    counter = {"n": 0}

    @with_backoff(max_retries=3, base_delay=0.01)
    def bad():
        counter["n"] += 1
        raise ValueError("programmer error")

    with pytest.raises(ValueError):
        bad()
    assert counter["n"] == 1  # pas de retry sur programmer error


def test_rate_limiter_gives_up_after_max_retries():
    from src.utils.rate_limiter import with_backoff

    counter = {"n": 0}

    class FakeRateLimit(Exception):
        def __init__(self):
            super().__init__("429")

    @with_backoff(max_retries=2, base_delay=0.01, max_delay=0.05, jitter=False)
    def always_fail():
        counter["n"] += 1
        raise FakeRateLimit()

    with pytest.raises(FakeRateLimit):
        always_fail()
    assert counter["n"] == 3  # 1 + 2 retries


# ---------------------------------------------------------------------------
# LLM cost tracker
# ---------------------------------------------------------------------------


def test_llm_cost_tracker_accumulates():
    from src.utils import llm_cost_tracker as tracker

    # Reset singleton state pour test isolé
    tracker._state["total_prompt_tokens"] = 0
    tracker._state["total_completion_tokens"] = 0
    tracker._state["total_usd"] = 0.0
    tracker._state["calls_by_model"] = {}
    tracker._state["date"] = None

    cost1 = tracker.track_llm_call("llama-3.1-8b-instant", 1000, 200)
    cost2 = tracker.track_llm_call("llama-3.1-8b-instant", 500, 100)
    assert cost1 > 0
    assert cost2 > 0
    snap = tracker.current_snapshot()
    # Le snapshot est arrondi à 4 décimales, on compare à la même précision
    assert snap["total_usd"] == pytest.approx(cost1 + cost2, abs=5e-5)
    assert snap["calls_by_model"]["llama-3.1-8b-instant"]["calls"] == 2


def test_llm_cost_tracker_raises_on_budget_exceeded(monkeypatch):
    from src.utils import llm_cost_tracker as tracker

    # Reset et met un budget ridiculement bas
    tracker._state["total_prompt_tokens"] = 0
    tracker._state["total_completion_tokens"] = 0
    tracker._state["total_usd"] = 0.0
    tracker._state["calls_by_model"] = {}
    tracker._state["date"] = None
    monkeypatch.setattr(tracker, "LLM_DAILY_BUDGET_USD", 0.0001)

    with pytest.raises(tracker.BudgetExceededError):
        # Un seul appel de 1M tokens à 0.59$/M dépasse largement 0.0001$
        tracker.track_llm_call("llama-3.3-70b-versatile", 1_000_000, 0)


# ---------------------------------------------------------------------------
# DRY_RUN logger
# ---------------------------------------------------------------------------


def test_dry_run_logger_writes_jsonl(tmp_path, monkeypatch):
    from src.utils import dry_run_logger

    log_path = tmp_path / "dry.jsonl"
    monkeypatch.setattr(dry_run_logger, "DRY_RUN_LOG_PATH", str(log_path))

    p = dry_run_logger.log_dry_run_order(
        ticker="AAPL",
        signal="Achat",
        prix=180.0,
        quantite=10,
        montant_eur=1800.0,
        sizing_method="half_kelly",
        win_prob=0.58,
        risk_level="FAIBLE",
    )
    assert p.exists()
    line = p.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["ticker"] == "AAPL"
    assert record["signal"] == "Achat"
    assert record["montant_eur"] == 1800.0


# ---------------------------------------------------------------------------
# Portfolio constraints : cap sectoriel
# ---------------------------------------------------------------------------


def test_sector_concentration_allows_within_cap():
    from src.strategy.portfolio_constraints import check_sector_concentration

    class MockPortfolio:
        def valeur_totale(self):
            return 100_000.0

        def exposition_sectorielle(self):
            return {"Technology": 0.25}

    r = check_sector_concentration(MockPortfolio(), "Technology", 4_000)
    assert r.allowed is True
    assert r.projected_pct == pytest.approx(0.29, abs=0.01)


def test_sector_concentration_refuses_over_cap():
    from src.strategy.portfolio_constraints import check_sector_concentration

    class MockPortfolio:
        def valeur_totale(self):
            return 100_000.0

        def exposition_sectorielle(self):
            return {"Technology": 0.25}

    r = check_sector_concentration(MockPortfolio(), "Technology", 10_000)
    assert r.allowed is False
    assert r.projected_pct > 0.30


# ---------------------------------------------------------------------------
# Portfolio constraints : cap de corrélation cross-sectionnelle
# ---------------------------------------------------------------------------


def test_pairwise_correlation_refuses_high_rho(monkeypatch):
    """
    Deux tickers avec des log-returns parfaitement corrélés (ρ=1.0) doivent
    être refusés quand cap=0.80.
    """
    from src.strategy import portfolio_constraints as pc

    # Mock _fetch_log_returns pour retourner deux séries corrélées à 1.0
    series = [
        0.01,
        -0.02,
        0.015,
        -0.005,
        0.02,
        -0.01,
        0.008,
        -0.015,
        0.012,
        -0.007,
        0.018,
        -0.011,
        0.009,
        -0.004,
        0.016,
    ]

    def fake_fetch(ticker, lookback_days):
        return series[:]  # copie : ρ(r, r) = 1.0 exactement

    monkeypatch.setattr(pc, "_fetch_log_returns", fake_fetch)

    class MockPortfolio:
        positions = {"AAPL": object()}  # une position existante

    r = pc.check_pairwise_correlation(MockPortfolio(), "MSFT", cap_rho=0.80)
    assert r.allowed is False
    assert r.worst_partner == "AAPL"
    assert r.worst_rho == pytest.approx(1.0, abs=1e-6)


def test_pairwise_correlation_allows_low_rho(monkeypatch):
    """
    Deux tickers avec des log-returns décorrélés (ρ proche de 0) doivent
    passer le cap. On utilise deux séries binaires orthogonales :
    somme du produit = 0 par construction → ρ ≈ 0.
    """
    from src.strategy import portfolio_constraints as pc

    # Pattern 1 : alternance (1, -1, 1, -1, ...)
    series_a = [0.01 if i % 2 == 0 else -0.01 for i in range(16)]
    # Pattern 2 : blocs de 2 (1, 1, -1, -1, ...) → orthogonal à l'alternance
    series_b = [0.01 if (i // 2) % 2 == 0 else -0.01 for i in range(16)]

    def fake_fetch(ticker, lookback_days):
        return series_a[:] if ticker == "AAPL" else series_b[:]

    monkeypatch.setattr(pc, "_fetch_log_returns", fake_fetch)

    class MockPortfolio:
        positions = {"AAPL": object()}

    r = pc.check_pairwise_correlation(MockPortfolio(), "TLT", cap_rho=0.80)
    # ρ doit être nettement sous 0.80 (proche de 0 par orthogonalité)
    assert r.allowed is True
    assert abs(r.worst_rho) < 0.80


def test_pairwise_correlation_empty_portfolio_allows():
    from src.strategy.portfolio_constraints import check_pairwise_correlation

    class MockPortfolio:
        positions = {}

    r = check_pairwise_correlation(MockPortfolio(), "AAPL", cap_rho=0.80)
    assert r.allowed is True  # aucun partenaire à comparer


# ---------------------------------------------------------------------------
# Position sizer : HOLD guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal", ["HOLD_SYSTEMIC", "HOLD_SECTOR_CAP", "HOLD_CORR_CAP"])
def test_position_sizer_respects_hold_overrides(signal):
    from src.strategy.investor_profile import load_investor_profile
    from src.strategy.portfolio_state import PortfolioState
    from src.strategy.position_sizer import calculate_position_size

    portfolio = PortfolioState(capital_initial=100_000, cash_disponible=100_000, positions={})
    profile = load_investor_profile()

    r = calculate_position_size(
        ticker="TEST",
        prix_actuel=100.0,
        signal_final=signal,
        impact_strength=0.6,
        trust_trade_score=0.6,
        profile=profile,
        portfolio=portfolio,
    )
    assert r.nb_actions == 0
    assert r.action_type == "TENIR"


# ---------------------------------------------------------------------------
# Politis-White : sélection auto de la longueur de bloc
# ---------------------------------------------------------------------------


def test_politis_white_white_noise_returns_small_block():
    """Série i.i.d. → bloc proche du minimum (pas d'autocorrélation à capturer)."""
    import numpy as np

    from src.utils.politis_white import politis_white_block_length

    rng = np.random.default_rng(42)
    white = rng.standard_normal(500)
    b = politis_white_block_length(white)
    assert 1 <= b <= 5, f"Attendu bloc <=5 pour bruit blanc, obtenu {b}"


def test_politis_white_ar1_persistent_returns_larger_block():
    """AR(1) phi=0.8 → bloc significativement plus grand que bruit blanc."""
    import numpy as np

    from src.utils.politis_white import politis_white_block_length

    rng = np.random.default_rng(42)
    ar1 = np.zeros(500)
    for t in range(1, 500):
        ar1[t] = 0.8 * ar1[t - 1] + rng.standard_normal()
    b = politis_white_block_length(ar1)
    assert b >= 6, f"AR(1) phi=0.8 devrait donner b>=6, obtenu {b}"


def test_politis_white_short_series_fallback():
    """n < 20 → valeur par défaut (5) pour éviter les artefacts."""
    from src.utils.politis_white import politis_white_block_length

    b = politis_white_block_length([0.1, -0.2, 0.3, -0.1, 0.05])
    assert b == 5


def test_politis_white_clamp_respected():
    """b_max respecté (pas de blocs absurdes sur séries courtes)."""
    import numpy as np

    from src.utils.politis_white import politis_white_block_length

    # AR(1) très persistant, forcé à b_max=10
    rng = np.random.default_rng(0)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.95 * ar[t - 1] + rng.standard_normal()
    b = politis_white_block_length(ar, b_max=10)
    assert 1 <= b <= 10


# ---------------------------------------------------------------------------
# Walk-Forward : groupement par fenêtre (logique pure, sans yfinance)
# ---------------------------------------------------------------------------


def test_walk_forward_window_grouping():
    """_group_events_by_window découpe correctement par date."""
    from eval.evaluate_walk_forward import _group_events_by_window

    events = [
        {"ticker": "AAPL", "date_utc": "2024-01-05", "signal_final": "Achat"},
        {"ticker": "MSFT", "date_utc": "2024-01-15", "signal_final": "Vente"},
        {"ticker": "GOOGL", "date_utc": "2024-02-10", "signal_final": "Achat"},
        {"ticker": "NVDA", "date_utc": "2024-03-05", "signal_final": "Achat"},
    ]
    wins = _group_events_by_window(events, window_days=30)
    # Première fenêtre commence au premier événement
    assert len(wins) >= 3
    # Les événements de janvier sont regroupés
    jan_window = wins[0]
    assert len(jan_window[2]) == 2
    # Pas de perte d'événements
    total_events = sum(len(w[2]) for w in wins)
    assert total_events == len(events)


def test_walk_forward_sharpe_helper():
    """_sharpe_annualized : sanity checks (0 si const, positif si cars>0)."""
    import numpy as np

    from eval.evaluate_walk_forward import _sharpe_annualized

    # Série constante → std=0 → Sharpe=0
    assert _sharpe_annualized(np.array([0.01, 0.01, 0.01])) == 0.0
    # Série positive → Sharpe > 0
    assert _sharpe_annualized(np.array([0.01, 0.02, 0.015, 0.008])) > 0
    # Série négative → Sharpe < 0
    assert _sharpe_annualized(np.array([-0.01, -0.02, -0.005])) < 0


# ---------------------------------------------------------------------------
# Smoke : tous les modules Phase 3 s'importent
# ---------------------------------------------------------------------------


def test_phase3_imports():
    import scripts.fit_calibration_nightly  # noqa: F401
    import src.knowledge.liquidity  # noqa: F401
    import src.strategy.portfolio_constraints  # noqa: F401
    import src.utils.dry_run_logger  # noqa: F401
    import src.utils.llm_cost_tracker  # noqa: F401
    import src.utils.politis_white  # noqa: F401
    import src.utils.rate_limiter  # noqa: F401
