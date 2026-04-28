"""
test_ticker_discovery.py — Tests pour le module discovery.ticker_discovery.

Couvre :
  - DiscoveryScore.summary
  - DiscoveryReport.tickers / summary
  - UniverseFilter (etf, market_cap, sector, white/blacklist)
  - CallableSource (clipping, error swallowing)
  - TrendingNewsSource (cashtag + parenthesis extractor, normalisation)
  - VolumeAnomalySource (z-score)
  - SocialSpikeSource (log scaling)
  - EarningsCalendarSource (peak at d=0, decay)
  - SecRecentSource (form weights, recency)
  - BigMoversSource (saturation)
  - TickerDiscoveryEngine.discover (aggregation, decay, hash)
  - recent_history_from_harvest_dir (JSONL replay)
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Score / Report
# ---------------------------------------------------------------------------


def test_discovery_score_summary_ascii_safe():
    from src.discovery.ticker_discovery import DiscoveryScore

    s = DiscoveryScore(
        ticker="AAPL",
        total_score=0.6,
        raw_score=0.8,
        contributions={"trending_news": 0.9, "volume_anomaly": 0.7, "social_spike": 0.5},
        novelty_multiplier=0.75,
    )
    summary = s.summary()
    summary.encode("cp1252")
    assert "AAPL" in summary


def test_discovery_report_tickers_and_summary():
    from src.discovery.ticker_discovery import DiscoveryReport, DiscoveryScore

    rep = DiscoveryReport(
        run_at=datetime.now(timezone.utc),
        top_n=2,
        universe_size=5,
        candidates_seen=5,
        scores=[
            DiscoveryScore(ticker="A", total_score=0.9),
            DiscoveryScore(ticker="B", total_score=0.7),
            DiscoveryScore(ticker="C", total_score=0.4),
        ],
        sources_used=["s1"],
        duration_sec=1.0,
    )
    assert rep.tickers() == ["A", "B"]
    assert "candidates=5" in rep.summary()


# ---------------------------------------------------------------------------
# UniverseFilter
# ---------------------------------------------------------------------------


def test_universe_filter_excludes_etfs_by_default():
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(min_market_cap=0)
    ok, reason = uf.is_eligible("SPY")
    assert ok is False
    assert reason == "etf"


def test_universe_filter_blacklist_wins_over_whitelist():
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(
        min_market_cap=0,
        explicit_whitelist=["AAPL", "MSFT"],
        explicit_blacklist=["AAPL"],
    )
    ok, reason = uf.is_eligible("AAPL")
    assert ok is False
    assert reason == "blacklisted"


def test_universe_filter_whitelist_only_admits_listed():
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(min_market_cap=0, explicit_whitelist=["AAPL"])
    ok_a, _ = uf.is_eligible("AAPL")
    ok_b, reason_b = uf.is_eligible("MSFT")
    assert ok_a is True
    assert ok_b is False
    assert reason_b == "not_in_whitelist"


def test_universe_filter_market_cap_below_excluded():
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(
        min_market_cap=10e9,
        market_cap_lookup_fn=lambda t: 1e9,  # too small
    )
    ok, reason = uf.is_eligible("AAPL")
    assert ok is False
    assert reason == "below_market_cap"


def test_universe_filter_sector_exclusion():
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(
        min_market_cap=0,
        excluded_sectors=["Utilities"],
        sector_lookup_fn=lambda t: "Utilities" if t == "DUK" else "Tech",
    )
    ok_t, _ = uf.is_eligible("AAPL")
    ok_d, reason_d = uf.is_eligible("DUK")
    assert ok_t is True
    assert ok_d is False
    assert reason_d == "excluded_sector"


def test_universe_filter_invalid_format():
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(min_market_cap=0)
    ok_empty, _ = uf.is_eligible("")
    ok_long, _ = uf.is_eligible("ABCDEFG")
    assert ok_empty is False
    assert ok_long is False


def test_universe_filter_market_cap_lookup_returns_none_soft_pass():
    """Si market_cap inconnu, on n'exclut pas (soft-pass)."""
    from src.discovery.ticker_discovery import UniverseFilter

    uf = UniverseFilter(
        min_market_cap=10e9,
        market_cap_lookup_fn=lambda t: None,
    )
    ok, _ = uf.is_eligible("XYZ")
    assert ok is True


# ---------------------------------------------------------------------------
# CallableSource
# ---------------------------------------------------------------------------


def test_callable_source_clips_and_uppercases():
    from src.discovery.ticker_discovery import CallableSource

    src = CallableSource("test", lambda: {"aapl": 1.5, "msft": -0.2, "nvda": 0.7})
    out = src.fetch_scores()
    assert out["AAPL"] == 1.0  # clip a 1.0
    assert out["MSFT"] == 0.0  # clip a 0.0
    assert out["NVDA"] == 0.7


def test_callable_source_swallows_exceptions():
    from src.discovery.ticker_discovery import CallableSource

    def boom():
        raise RuntimeError("network down")

    src = CallableSource("test", boom)
    assert src.fetch_scores() == {}


# ---------------------------------------------------------------------------
# TrendingNewsSource
# ---------------------------------------------------------------------------


def test_trending_news_extracts_cashtags_and_parens():
    from src.discovery.ticker_discovery import TrendingNewsSource

    articles = [
        {"title": "Apple ($AAPL) beats earnings", "summary": "$AAPL surges"},
        {"title": "Nvidia (NVDA) ATH", "summary": "$NVDA AI demand"},
        {"title": "Random news", "summary": "no tickers here"},
    ]
    src = TrendingNewsSource(news_fetcher=lambda h: articles)
    scores = src.fetch_scores()
    # AAPL apparait 2x, NVDA 2x => normalises a 1.0 et 1.0
    assert "AAPL" in scores
    assert "NVDA" in scores
    assert scores["AAPL"] > 0


def test_trending_news_normalises_to_max():
    from src.discovery.ticker_discovery import TrendingNewsSource

    articles = [
        {"title": "$AAPL", "summary": ""},
        {"title": "$AAPL", "summary": ""},
        {"title": "$AAPL", "summary": ""},
        {"title": "$MSFT", "summary": ""},
    ]
    src = TrendingNewsSource(news_fetcher=lambda h: articles)
    scores = src.fetch_scores()
    assert scores["AAPL"] == pytest.approx(1.0)
    assert scores["MSFT"] == pytest.approx(1.0 / 3.0)


def test_trending_news_no_fetcher_returns_empty():
    from src.discovery.ticker_discovery import TrendingNewsSource

    src = TrendingNewsSource(news_fetcher=None)
    assert src.fetch_scores() == {}


def test_trending_news_fetcher_raises_returns_empty():
    from src.discovery.ticker_discovery import TrendingNewsSource

    src = TrendingNewsSource(news_fetcher=lambda h: (_ for _ in ()).throw(RuntimeError("api 500")))
    assert src.fetch_scores() == {}


# ---------------------------------------------------------------------------
# VolumeAnomalySource
# ---------------------------------------------------------------------------


def test_volume_anomaly_z_score_above_threshold():
    from src.discovery.ticker_discovery import VolumeAnomalySource

    quotes = {
        "AAPL": {"volume": 100, "volume_ma_20": 50, "volume_std_20": 10},  # z=5 -> sat
        "MSFT": {"volume": 55, "volume_ma_20": 50, "volume_std_20": 10},  # z=0.5 -> below min
    }
    src = VolumeAnomalySource(
        quote_fetcher=lambda t: quotes,
        candidate_universe=["AAPL", "MSFT"],
    )
    scores = src.fetch_scores()
    assert "AAPL" in scores
    assert scores["AAPL"] == pytest.approx(1.0)  # z=5 / sat=4 -> 1.25 -> clip 1.0
    assert "MSFT" not in scores


def test_volume_anomaly_handles_zero_std():
    from src.discovery.ticker_discovery import VolumeAnomalySource

    quotes = {"AAPL": {"volume": 100, "volume_ma_20": 50, "volume_std_20": 0}}
    src = VolumeAnomalySource(
        quote_fetcher=lambda t: quotes,
        candidate_universe=["AAPL"],
    )
    assert src.fetch_scores() == {}


# ---------------------------------------------------------------------------
# SocialSpikeSource
# ---------------------------------------------------------------------------


def test_social_spike_log_scaling():
    from src.discovery.ticker_discovery import SocialSpikeSource

    src = SocialSpikeSource(trending_fetcher=lambda: {"GME": 1000, "AAPL": 100, "NVDA": 10})
    scores = src.fetch_scores()
    # GME (max) = 1.0, AAPL et NVDA proportionnels au log
    assert scores["GME"] == pytest.approx(1.0)
    assert 0 < scores["AAPL"] < 1.0
    assert scores["NVDA"] < scores["AAPL"]


def test_social_spike_empty_input():
    from src.discovery.ticker_discovery import SocialSpikeSource

    src = SocialSpikeSource(trending_fetcher=lambda: {})
    assert src.fetch_scores() == {}


# ---------------------------------------------------------------------------
# EarningsCalendarSource
# ---------------------------------------------------------------------------


def test_earnings_calendar_peak_at_zero_days():
    from src.discovery.ticker_discovery import EarningsCalendarSource

    src = EarningsCalendarSource(calendar_fetcher=lambda: {"AAPL": 0, "MSFT": 1, "GOOGL": 5, "META": -1, "TSLA": 10})
    scores = src.fetch_scores()
    assert scores["AAPL"] == 1.0
    assert scores["MSFT"] == pytest.approx(0.85)
    assert scores["META"] == pytest.approx(0.85)
    assert "TSLA" not in scores  # hors fenetre


def test_earnings_calendar_decay_through_window():
    from src.discovery.ticker_discovery import EarningsCalendarSource

    src = EarningsCalendarSource(
        calendar_fetcher=lambda: {"X": 3},  # entre 1 et days_window
        days_window=5,
    )
    scores = src.fetch_scores()
    assert 0.0 < scores["X"] < 1.0


# ---------------------------------------------------------------------------
# SecRecentSource
# ---------------------------------------------------------------------------


def test_sec_recent_form_weighting():
    from src.discovery.ticker_discovery import SecRecentSource

    now = datetime.now(timezone.utc)
    src = SecRecentSource(
        filings_fetcher=lambda: {
            "AAPL": [{"form_type": "8-K", "filing_date": (now - timedelta(hours=1)).isoformat()}],
            "MSFT": [{"form_type": "10-K", "filing_date": (now - timedelta(hours=1)).isoformat()}],
        }
    )
    scores = src.fetch_scores()
    # 8-K (poids 1.0) doit > 10-K (poids 0.3) a recence egale
    assert scores["AAPL"] > scores["MSFT"]


def test_sec_recent_filters_old_filings():
    from src.discovery.ticker_discovery import SecRecentSource

    now = datetime.now(timezone.utc)
    src = SecRecentSource(
        filings_fetcher=lambda: {
            "OLD": [{"form_type": "8-K", "filing_date": (now - timedelta(days=10)).isoformat()}],
        },
        lookback_hours=72,
    )
    assert src.fetch_scores() == {}


# ---------------------------------------------------------------------------
# BigMoversSource
# ---------------------------------------------------------------------------


def test_big_movers_min_threshold_filters_noise():
    from src.discovery.ticker_discovery import BigMoversSource

    src = BigMoversSource(
        movers_fetcher=lambda: {"AAPL": 0.005, "NVDA": 0.05, "GME": -0.07},
        min_abs_return=0.03,
    )
    scores = src.fetch_scores()
    assert "AAPL" not in scores
    assert "NVDA" in scores
    assert "GME" in scores  # absolute value used


def test_big_movers_saturation():
    from src.discovery.ticker_discovery import BigMoversSource

    src = BigMoversSource(
        movers_fetcher=lambda: {"X": 0.20, "Y": 0.05},
        min_abs_return=0.03,
        return_saturation=0.10,
    )
    scores = src.fetch_scores()
    assert scores["X"] == pytest.approx(1.0)  # capped
    assert scores["Y"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TickerDiscoveryEngine
# ---------------------------------------------------------------------------


def test_engine_aggregates_and_normalises_weights():
    from src.discovery.ticker_discovery import (
        CallableSource,
        TickerDiscoveryEngine,
        UniverseFilter,
    )

    sources = [
        CallableSource("trending_news", lambda: {"AAPL": 1.0, "MSFT": 0.5}),
        CallableSource("volume_anomaly", lambda: {"AAPL": 0.8}),
    ]
    eng = TickerDiscoveryEngine(
        sources=sources,
        universe_filter=UniverseFilter(min_market_cap=0),
        signal_weights={"trending_news": 0.6, "volume_anomaly": 0.4},
    )
    rep = eng.discover(top_n=5)
    assert len(rep.scores) == 2
    # AAPL: 1.0*0.6 + 0.8*0.4 = 0.92
    aapl = next(s for s in rep.scores if s.ticker == "AAPL")
    assert aapl.raw_score == pytest.approx(0.92, abs=1e-3)
    msft = next(s for s in rep.scores if s.ticker == "MSFT")
    assert msft.raw_score == pytest.approx(0.5 * 0.6, abs=1e-3)


def test_engine_applies_decay_for_recent_history():
    from src.discovery.ticker_discovery import (
        CallableSource,
        TickerDiscoveryEngine,
        UniverseFilter,
    )

    src = CallableSource("trending_news", lambda: {"AAPL": 1.0, "MSFT": 1.0})
    eng = TickerDiscoveryEngine(
        sources=[src],
        universe_filter=UniverseFilter(min_market_cap=0),
        recent_history_fn=lambda: {"AAPL": 1.0},  # vu hier
        decay_per_day=0.5,
    )
    rep = eng.discover(top_n=2)
    aapl = next(s for s in rep.scores if s.ticker == "AAPL")
    msft = next(s for s in rep.scores if s.ticker == "MSFT")
    assert aapl.novelty_multiplier < 1.0
    assert msft.novelty_multiplier == pytest.approx(1.0)
    # Donc MSFT doit passer devant AAPL apres decay
    assert rep.scores[0].ticker == "MSFT"


def test_engine_universe_hash_is_deterministic_and_order_invariant():
    from src.discovery.ticker_discovery import TickerDiscoveryEngine

    h1 = TickerDiscoveryEngine._universe_hash(["AAPL", "MSFT", "NVDA"])
    h2 = TickerDiscoveryEngine._universe_hash(["NVDA", "AAPL", "MSFT"])
    h3 = TickerDiscoveryEngine._universe_hash(["AAPL", "TSLA"])
    assert h1 == h2  # invariant a l'ordre
    assert h1 != h3
    assert len(h1) == 16


def test_engine_filters_etfs():
    from src.discovery.ticker_discovery import (
        CallableSource,
        TickerDiscoveryEngine,
        UniverseFilter,
    )

    src = CallableSource("trending_news", lambda: {"AAPL": 1.0, "SPY": 1.0, "QQQ": 1.0})
    eng = TickerDiscoveryEngine(
        sources=[src],
        universe_filter=UniverseFilter(min_market_cap=0, exclude_etfs=True),
    )
    rep = eng.discover(top_n=10)
    selected = [s.ticker for s in rep.scores]
    assert "AAPL" in selected
    assert "SPY" not in selected
    assert "QQQ" not in selected


def test_engine_handles_source_exception():
    from src.discovery.ticker_discovery import (
        CallableSource,
        DiscoverySource,
        TickerDiscoveryEngine,
        UniverseFilter,
    )

    class BoomSource(DiscoverySource):
        name = "boom"

        def fetch_scores(self):
            raise RuntimeError("oops")

    src_ok = CallableSource("trending_news", lambda: {"AAPL": 1.0})
    eng = TickerDiscoveryEngine(
        sources=[BoomSource(), src_ok],
        universe_filter=UniverseFilter(min_market_cap=0),
    )
    rep = eng.discover(top_n=5)
    # Le source crash ne tue pas la run
    assert rep.scores
    assert rep.scores[0].ticker == "AAPL"
    assert any(name == "boom" for name, _err in rep.sources_failed)


def test_engine_top_n_limits_results():
    from src.discovery.ticker_discovery import (
        CallableSource,
        TickerDiscoveryEngine,
        UniverseFilter,
    )

    src = CallableSource(
        "trending_news",
        lambda: {f"T{i}": 1.0 - i * 0.1 for i in range(10)},
    )
    eng = TickerDiscoveryEngine(
        sources=[src],
        universe_filter=UniverseFilter(min_market_cap=0),
    )
    rep = eng.discover(top_n=3)
    assert len(rep.scores) == 3
    # Les top doivent etre les plus haut score
    assert rep.scores[0].ticker == "T0"


def test_engine_empty_sources_returns_empty_report():
    from src.discovery.ticker_discovery import TickerDiscoveryEngine, UniverseFilter

    eng = TickerDiscoveryEngine(
        sources=[],
        universe_filter=UniverseFilter(min_market_cap=0),
    )
    rep = eng.discover(top_n=5)
    assert rep.scores == []
    assert rep.candidates_seen == 0


# ---------------------------------------------------------------------------
# recent_history_from_harvest_dir
# ---------------------------------------------------------------------------


def test_recent_history_from_harvest_dir_reads_jsonl():
    from src.discovery.ticker_discovery import recent_history_from_harvest_dir

    with tempfile.TemporaryDirectory() as tmp:
        # Ecrit 2 records dans 2 fichiers
        now = datetime.now(timezone.utc)
        f1 = Path(tmp) / "harvest_2026-04-24.jsonl"
        f2 = Path(tmp) / "harvest_2026-04-25.jsonl"
        f1.write_text(
            json.dumps(
                {
                    "ticker": "AAPL",
                    "timestamp": (now - timedelta(days=1)).isoformat(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        f2.write_text(
            json.dumps({"ticker": "MSFT", "timestamp": now.isoformat()})
            + "\n"
            + json.dumps({"ticker": "AAPL", "timestamp": now.isoformat()})
            + "\n",
            encoding="utf-8",
        )
        history = recent_history_from_harvest_dir(tmp, lookback_days=7)
        # AAPL le plus recent doit gagner (= ~0 jours)
        assert history.get("AAPL", 99) < 0.5
        assert "MSFT" in history


def test_recent_history_handles_missing_dir_silently():
    from src.discovery.ticker_discovery import recent_history_from_harvest_dir

    out = recent_history_from_harvest_dir("/nonexistent/path/qwerty")
    assert out == {}


def test_recent_history_skips_corrupt_lines():
    from src.discovery.ticker_discovery import recent_history_from_harvest_dir

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "harvest_2026-04-25.jsonl"
        path.write_text(
            "not a json line\n"
            + json.dumps({"ticker": "AAPL", "timestamp": datetime.now(timezone.utc).isoformat()})
            + "\n",
            encoding="utf-8",
        )
        out = recent_history_from_harvest_dir(tmp)
        assert "AAPL" in out
