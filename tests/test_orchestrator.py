"""
test_orchestrator.py — Tests pour discovery.orchestrator.DailyHarvestOrchestrator.

Couvre :
  - Cycle complet : discovery -> pipeline -> harvest
  - Cold-start triggered uniquement si is_cold(t) -> True
  - Persistence : 1 record par ticker, schema correct
  - Resilience : pipeline crash sur 1 ticker n'arrete pas la boucle
  - Resilience : timeout pipeline traite proprement
  - Resilience : discovery crash -> rapport vide propre
  - Idempotence : skip_already_harvested_today
  - Parallel mode (max_workers > 1) : meme resultat que sequentiel
  - On_ticker_done callback
  - Cold start budget timeout
  - Persistence cold_start signals
"""

import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers : mocks
# ---------------------------------------------------------------------------


class _MockColdStart:
    """Mock duck-typed ColdStartManager."""

    def __init__(self, cold_tickers=None, raise_on=None, slow=False):
        self.cold_tickers = set(cold_tickers or [])
        self.raise_on = set(raise_on or [])
        self.slow = bool(slow)
        self.bootstrap_calls = []

    def is_cold(self, ticker):
        return ticker.upper() in self.cold_tickers

    def bootstrap(self, ticker, company_name=None, sector=None):
        if self.slow:
            time.sleep(2.0)
        self.bootstrap_calls.append(ticker)
        if ticker.upper() in self.raise_on:
            raise RuntimeError(f"bootstrap failed for {ticker}")
        self.cold_tickers.discard(ticker.upper())

        class _Rep:
            was_cold = True
            is_cold_after = False
            docs_before = 0
            docs_after = 5
            peers_used = ["MSFT", "GOOGL"]

        return _Rep()


def _make_engine(scores_per_source):
    """Construit un engine avec des sources callable mockees."""
    from src.discovery.ticker_discovery import (
        CallableSource,
        TickerDiscoveryEngine,
        UniverseFilter,
    )

    sources = [CallableSource(name, lambda data=data: data) for name, data in scores_per_source.items()]
    return TickerDiscoveryEngine(
        sources=sources,
        universe_filter=UniverseFilter(min_market_cap=0, exclude_etfs=False),
    )


def _good_pipeline(ticker, context):
    """Pipeline de test toujours OK."""
    rank = context["rank"]
    return {
        "decision": "BUY" if rank <= 1 else "HOLD",
        "confidence": 0.7,
        "position_size": 0.1,
        "rationale": f"test {ticker} rank {rank}",
        "signals": {"score": context["discovery_score"].total_score},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Cycle de base
# ---------------------------------------------------------------------------


def test_orchestrator_full_cycle_basic():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine(
        {
            "trending_news": {"AAPL": 1.0, "MSFT": 0.8, "NVDA": 0.6},
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=3,
            pipeline_version="test",
        )
        rep = orch.run_daily_harvest()
        assert len(rep.ticker_results) == 3
        assert all(r.success for r in rep.ticker_results)
        assert rep.discovery_report is not None

        # Persisted
        records = h.load_records()
        assert len(records) == 3
        assert all(r["meta"]["pipeline_version"] == "test" for r in records)


def test_orchestrator_attaches_universe_hash():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"AAPL": 1.0}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=1,
        )
        rep = orch.run_daily_harvest()
        records = h.load_records()
        assert records[0]["discovery"]["universe_hash"] == rep.discovery_report.universe_hash


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_orchestrator_cold_start_triggered_only_for_cold_tickers():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "OXY": 0.8}})
    cold = _MockColdStart(cold_tickers={"OXY"})

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            cold_start_manager=cold,
            top_n=2,
        )
        rep = orch.run_daily_harvest()
        # Seul OXY doit etre bootstrap
        assert cold.bootstrap_calls == ["OXY"]
        assert rep.cold_starts_triggered == 1
        assert rep.cold_starts_succeeded == 1

        # Cold-start metadata persistee dans signals
        records = h.load_records()
        oxy_rec = next(r for r in records if r["ticker"] == "OXY")
        assert oxy_rec["signals"]["cold_start_was_cold"] is True
        assert oxy_rec["signals"]["cold_start_docs_added"] == 5
        assert oxy_rec["signals"]["cold_start_peers_used"] == ["MSFT", "GOOGL"]

        aapl_rec = next(r for r in records if r["ticker"] == "AAPL")
        assert "cold_start_was_cold" not in aapl_rec["signals"]


def test_orchestrator_cold_start_exception_does_not_kill_pipeline():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"BAD": 1.0}})
    cold = _MockColdStart(cold_tickers={"BAD"}, raise_on={"BAD"})

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            cold_start_manager=cold,
            top_n=1,
        )
        rep = orch.run_daily_harvest()
        # Le pipeline doit quand meme tourner
        assert len(rep.ticker_results) == 1
        r = rep.ticker_results[0]
        assert r.cold_start_triggered is True
        assert r.cold_start_succeeded is False
        assert r.success is True  # le pipeline a quand meme tourne
        assert any("cold_start" in e for e in r.errors)


def test_orchestrator_cold_start_budget_timeout():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"SLOW": 1.0}})
    cold = _MockColdStart(cold_tickers={"SLOW"}, slow=True)

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            cold_start_manager=cold,
            top_n=1,
            cold_start_budget_sec=0.5,  # bien plus court que les 2s du mock
        )
        rep = orch.run_daily_harvest()
        # Cold start doit avoir timeout, mais le pipeline tourne quand meme
        r = rep.ticker_results[0]
        assert r.cold_start_triggered is True
        assert r.cold_start_succeeded is False
        assert r.success is True


# ---------------------------------------------------------------------------
# Resilience pipeline
# ---------------------------------------------------------------------------


def test_orchestrator_pipeline_crash_does_not_stop_loop():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    def bad_pipeline(ticker, context):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return _good_pipeline(ticker, context)

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "BAD": 0.9, "MSFT": 0.8}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=bad_pipeline,
            top_n=3,
        )
        rep = orch.run_daily_harvest()
        assert len(rep.ticker_results) == 3
        # 2 OK, 1 erreur
        assert sum(1 for r in rep.ticker_results if r.success) == 2
        bad = next(r for r in rep.ticker_results if r.ticker == "BAD")
        assert bad.success is False
        assert bad.decision == "ERROR"
        assert any("pipeline" in e for e in bad.errors)
        # Tous persistes
        records = h.load_records()
        assert len(records) == 3


def test_orchestrator_pipeline_returns_garbage_handled():
    """Si le pipeline retourne autre chose qu'un dict, le ticker est marque ERROR."""
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    def garbage_pipeline(ticker, context):
        return "not a dict"

    eng = _make_engine({"trending_news": {"AAPL": 1.0}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=garbage_pipeline,
            top_n=1,
        )
        rep = orch.run_daily_harvest()
        assert rep.ticker_results[0].success is False
        assert rep.ticker_results[0].decision == "ERROR"


def test_orchestrator_pipeline_timeout():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    def slow_pipeline(ticker, context):
        time.sleep(2.0)
        return _good_pipeline(ticker, context)

    eng = _make_engine({"trending_news": {"AAPL": 1.0}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=slow_pipeline,
            per_ticker_timeout_sec=0.5,
            top_n=1,
        )
        rep = orch.run_daily_harvest()
        assert rep.ticker_results[0].success is False
        assert any("timeout" in e.lower() for e in rep.ticker_results[0].errors)


def test_orchestrator_partial_pipeline_output():
    """Pipeline qui retourne un dict incomplet est accepte avec defaults."""
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    def partial_pipeline(ticker, context):
        return {"decision": "HOLD"}  # juste decision, pas de confidence/signals

    eng = _make_engine({"trending_news": {"AAPL": 1.0}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=partial_pipeline,
            top_n=1,
        )
        rep = orch.run_daily_harvest()
        r = rep.ticker_results[0]
        assert r.success is True
        assert r.decision == "HOLD"
        assert r.confidence is None


# ---------------------------------------------------------------------------
# Discovery crash
# ---------------------------------------------------------------------------


def test_orchestrator_discovery_crash_returns_empty_report():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    class _BoomEngine:
        def discover(self, top_n):
            raise RuntimeError("network kaboom")

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=_BoomEngine(),
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=5,
        )
        rep = orch.run_daily_harvest()
        assert rep.discovery_report is None
        assert rep.ticker_results == []
        assert h.load_records() == []


def test_orchestrator_no_tickers_in_universe():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({})  # aucune source -> 0 ticker
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=5,
        )
        rep = orch.run_daily_harvest()
        assert rep.ticker_results == []


# ---------------------------------------------------------------------------
# Idempotence skip_already_harvested
# ---------------------------------------------------------------------------


def test_orchestrator_skip_already_harvested_today():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "MSFT": 0.5}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        # Run 1
        orch1 = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=2,
        )
        rep1 = orch1.run_daily_harvest()
        assert len(rep1.ticker_results) == 2

        # Run 2 avec skip
        orch2 = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=2,
            skip_already_harvested_today=True,
        )
        rep2 = orch2.run_daily_harvest()
        assert len(rep2.ticker_results) == 0
        # Pas de nouveaux records
        assert len(h.load_records()) == 2


def test_orchestrator_skip_partial_already_harvested():
    """Si seul AAPL a deja ete harvested, MSFT doit toujours etre traite."""
    from src.discovery.data_harvester import DataHarvester, HarvestRecord
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "MSFT": 0.5}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        # Pre-populate avec AAPL deja harvested
        h.write(HarvestRecord(ticker="AAPL", decision="BUY"))

        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=2,
            skip_already_harvested_today=True,
        )
        rep = orch.run_daily_harvest()
        assert len(rep.ticker_results) == 1
        assert rep.ticker_results[0].ticker == "MSFT"


# ---------------------------------------------------------------------------
# Parallel mode
# ---------------------------------------------------------------------------


def test_orchestrator_parallel_mode_same_results():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine(
        {
            "trending_news": {f"T{i}": 1.0 - i * 0.05 for i in range(10)},
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            max_workers=4,
            top_n=5,
        )
        rep = orch.run_daily_harvest()
        assert len(rep.ticker_results) == 5
        # Toujours par ordre de rank
        ranks = [r.rank for r in rep.ticker_results]
        assert ranks == sorted(ranks)
        # Tous persistes
        assert len(h.load_records()) == 5


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def test_orchestrator_on_ticker_done_callback():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    seen = []

    def on_done(result):
        seen.append((result.ticker, result.success))

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "MSFT": 0.8}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=2,
            on_ticker_done=on_done,
        )
        orch.run_daily_harvest()
        assert len(seen) == 2
        assert all(ok for _t, ok in seen)


def test_orchestrator_on_ticker_done_callback_exception_is_isolated():
    """Exception dans le callback n'arrete pas le run."""
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    def on_done(result):
        raise RuntimeError("callback boom")

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "MSFT": 0.8}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=2,
            on_ticker_done=on_done,
        )
        rep = orch.run_daily_harvest()
        # Le callback echoue mais le run continue et persiste
        assert len(rep.ticker_results) == 2
        assert len(h.load_records()) == 2


# ---------------------------------------------------------------------------
# OrchestratorReport
# ---------------------------------------------------------------------------


def test_orchestrator_report_summary_ascii_safe():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"AAPL": 1.0}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=1,
        )
        rep = orch.run_daily_harvest()
        summary = rep.summary()
        # Doit etre encodable cp1252 (terminal Windows)
        summary.encode("cp1252")
        assert "AAPL" not in summary  # le summary parle pas du ticker direct
        assert "decisions=" in summary

        for line in rep.detail_lines():
            line.encode("cp1252")


# ---------------------------------------------------------------------------
# Pipeline context contract
# ---------------------------------------------------------------------------


def test_orchestrator_pipeline_receives_full_context():
    """Verifie que le pipeline recoit bien discovery_score, rank, etc."""
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    captured = []

    def capture_pipeline(ticker, context):
        captured.append(
            {
                "ticker": ticker,
                "rank": context["rank"],
                "score_total": context["discovery_score"].total_score,
                "report_present": context["discovery_report"] is not None,
                "cold_start_present": context["cold_start_report"] is not None,
                "date_label": context["date_label"],
            }
        )
        return _good_pipeline(ticker, context)

    eng = _make_engine({"trending_news": {"AAPL": 1.0, "MSFT": 0.5}})
    cold = _MockColdStart(cold_tickers={"MSFT"})

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=capture_pipeline,
            cold_start_manager=cold,
            top_n=2,
        )
        orch.run_daily_harvest(date_label="2026-04-25")

        assert len(captured) == 2
        aapl = next(c for c in captured if c["ticker"] == "AAPL")
        msft = next(c for c in captured if c["ticker"] == "MSFT")
        assert aapl["rank"] == 1
        assert msft["rank"] == 2
        assert aapl["score_total"] > msft["score_total"]
        assert aapl["cold_start_present"] is False  # AAPL pas cold
        assert msft["cold_start_present"] is True  # MSFT cold -> bootstrap fait
        assert aapl["date_label"] == "2026-04-25"


# ---------------------------------------------------------------------------
# Meta passing
# ---------------------------------------------------------------------------


def test_orchestrator_meta_is_persisted():
    from src.discovery.data_harvester import DataHarvester
    from src.discovery.orchestrator import DailyHarvestOrchestrator

    eng = _make_engine({"trending_news": {"AAPL": 1.0}})
    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=h,
            analysis_pipeline_fn=_good_pipeline,
            top_n=1,
            pipeline_version="v9.9.9",
            prompts_hash="hash-xyz",
            git_sha="abcdef0",
            model_versions={"absa": "llama-3.3", "embedder": "bge-m3"},
        )
        orch.run_daily_harvest()
        rec = h.load_records()[0]
        assert rec["meta"]["pipeline_version"] == "v9.9.9"
        assert rec["meta"]["prompts_hash"] == "hash-xyz"
        assert rec["meta"]["git_sha"] == "abcdef0"
        assert rec["meta"]["model_versions"]["embedder"] == "bge-m3"
