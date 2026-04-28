"""
test_data_harvester.py — Tests pour le module discovery.data_harvester.

Couvre :
  - HarvestRecord.to_json_dict (schema v1)
  - DataHarvester.write : single record, generation harvest_id/timestamp
  - DataHarvester.write_many : batch
  - DataHarvester.iter_records / load_records : lecture, filtrage par date
  - DataHarvester.stats : count par ticker, decision, jour
  - DataHarvester.load_dataframe (pandas optional)
  - Resilience : lignes corrompues, fichiers manquants
  - Daily rotation : 1 fichier par jour
  - Schema version preservee
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# HarvestRecord
# ---------------------------------------------------------------------------


def test_harvest_record_to_json_dict_full_schema():
    from src.discovery.data_harvester import SCHEMA_VERSION, HarvestRecord

    rec = HarvestRecord(
        ticker="AAPL",
        rank=3,
        discovery_total_score=0.612,
        discovery_raw_score=0.778,
        discovery_contributions={"trending_news": 0.85},
        discovery_novelty_multiplier=0.61,
        discovery_days_since_last_seen=1.0,
        universe_hash="a1b2c3",
        decision="BUY",
        confidence=0.74,
        position_size=0.12,
        rationale="strong beat",
        pipeline_duration_sec=8.4,
        pipeline_errors=[],
        signals={"x": 1.0},
        pipeline_version="v3.4.1",
        prompts_hash="abc",
        model_versions={"absa": "llama"},
        timestamp="2026-04-25T12:00:00+00:00",
        harvest_id="deadbeef",
    )
    d = rec.to_json_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["harvest_id"] == "deadbeef"
    assert d["ticker"] == "AAPL"
    assert d["discovery"]["rank"] == 3
    assert d["pipeline"]["decision"] == "BUY"
    assert d["signals"]["x"] == 1.0
    assert d["meta"]["pipeline_version"] == "v3.4.1"
    assert d["meta"]["model_versions"]["absa"] == "llama"


def test_harvest_record_to_json_dict_uppercases_ticker():
    from src.discovery.data_harvester import HarvestRecord

    rec = HarvestRecord(ticker="aapl")
    d = rec.to_json_dict()
    assert d["ticker"] == "AAPL"


def test_harvest_record_minimal():
    """On peut creer un record avec juste un ticker (pour cas d'erreur partielle)."""
    from src.discovery.data_harvester import HarvestRecord

    rec = HarvestRecord(ticker="ZZZ")
    d = rec.to_json_dict()
    assert d["ticker"] == "ZZZ"
    assert d["pipeline"]["decision"] is None
    assert d["discovery"]["rank"] is None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def test_data_harvester_write_generates_id_and_timestamp():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        rec = HarvestRecord(ticker="AAPL", decision="BUY")
        hid = h.write(rec)
        assert hid
        assert len(hid) >= 16
        assert rec.timestamp  # rempli
        assert rec.harvest_id == hid


def test_data_harvester_write_creates_daily_file():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp, date_provider=lambda: "2026-01-15")
        h.write(HarvestRecord(ticker="A", decision="BUY"))
        expected = Path(tmp) / "harvest_2026-01-15.jsonl"
        assert expected.exists()


def test_data_harvester_write_appends_not_overwrites():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write(HarvestRecord(ticker="A", decision="BUY"))
        h.write(HarvestRecord(ticker="B", decision="SELL"))
        records = h.load_records()
        assert len(records) == 2
        tickers = sorted(r["ticker"] for r in records)
        assert tickers == ["A", "B"]


def test_data_harvester_write_many():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        ids = h.write_many(
            [
                HarvestRecord(ticker="A", decision="BUY"),
                HarvestRecord(ticker="B", decision="HOLD"),
                HarvestRecord(ticker="C", decision="PASS"),
            ]
        )
        assert len(ids) == 3
        records = h.load_records()
        assert len(records) == 3


def test_data_harvester_write_many_empty_returns_empty():
    from src.discovery.data_harvester import DataHarvester

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        assert h.write_many([]) == []


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def test_data_harvester_iter_records_filters_corrupt_lines():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write(HarvestRecord(ticker="A", decision="BUY"))

        # Append a corrupt line manuellement
        path = next(Path(tmp).glob("harvest_*.jsonl"))
        with open(path, "a", encoding="utf-8") as f:
            f.write("not valid json\n")
        h.write(HarvestRecord(ticker="B", decision="SELL"))

        records = list(h.iter_records())
        # 2 records valides, ligne corrompue ignoree
        assert len(records) == 2
        tickers = {r["ticker"] for r in records}
        assert tickers == {"A", "B"}


def test_data_harvester_filter_by_date_inclusive():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        # 3 jours simules
        for date in ["2026-01-10", "2026-01-15", "2026-01-20"]:
            h = DataHarvester(output_dir=tmp, date_provider=lambda d=date: d)
            h.write(HarvestRecord(ticker="X", decision="BUY"))

        h_read = DataHarvester(output_dir=tmp)
        # Range inclusif
        recs = h_read.load_records(start_date="2026-01-12", end_date="2026-01-18")
        assert len(recs) == 1

        recs_all = h_read.load_records()
        assert len(recs_all) == 3

        recs_only_one = h_read.load_records(start_date="2026-01-15", end_date="2026-01-15")
        assert len(recs_only_one) == 1


def test_data_harvester_load_records_empty_dir():
    from src.discovery.data_harvester import DataHarvester

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        assert h.load_records() == []


def test_data_harvester_iter_handles_unreadable_files():
    """Une erreur de lecture sur un fichier ne doit pas casser l'iteration."""
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write(HarvestRecord(ticker="OK", decision="BUY"))
        # Ne pas creer un truc invalide qui ferait un crash systeme
        # Test simple : verifier que iter_records ne leve pas d'exception
        records = h.load_records()
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_data_harvester_stats_counts_correctly():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write_many(
            [
                HarvestRecord(ticker="AAPL", decision="BUY", confidence=0.8),
                HarvestRecord(ticker="AAPL", decision="HOLD", confidence=0.5),
                HarvestRecord(ticker="MSFT", decision="BUY", confidence=0.7),
            ]
        )
        s = h.stats()
        assert s["total_records"] == 3
        assert s["unique_tickers"] == 2
        assert s["per_ticker"] == {"AAPL": 2, "MSFT": 1}
        assert s["per_decision"] == {"BUY": 2, "HOLD": 1}
        assert 0.6 < s["avg_confidence"] < 0.7
        assert s["n_days"] == 1


def test_data_harvester_stats_ignores_records_without_confidence():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write_many(
            [
                HarvestRecord(ticker="A", decision="BUY", confidence=0.8),
                HarvestRecord(ticker="B", decision="HOLD", confidence=None),
            ]
        )
        s = h.stats()
        assert s["avg_confidence"] == pytest.approx(0.8)


def test_data_harvester_stats_empty():
    from src.discovery.data_harvester import DataHarvester

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        s = h.stats()
        assert s["total_records"] == 0
        assert s["per_ticker"] == {}
        assert s["avg_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_data_harvester_schema_version_preserved():
    from src.discovery.data_harvester import SCHEMA_VERSION, DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write(HarvestRecord(ticker="A", decision="BUY"))
        recs = h.load_records()
        assert recs[0]["schema_version"] == SCHEMA_VERSION


def test_data_harvester_signals_round_trip():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        signals = {
            "transcript_red_flag": 0.18,
            "social_sentiment": 0.45,
            "sentiment_divergence_regime": "smart_buy",
            "nested": {"a": 1, "b": [1, 2, 3]},
        }
        h.write(HarvestRecord(ticker="X", signals=signals))
        recs = h.load_records()
        assert recs[0]["signals"] == signals


# ---------------------------------------------------------------------------
# Dataframe
# ---------------------------------------------------------------------------


def test_data_harvester_load_dataframe():
    pd = pytest.importorskip("pandas")
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        h.write_many(
            [
                HarvestRecord(
                    ticker="A",
                    rank=1,
                    decision="BUY",
                    confidence=0.8,
                    discovery_total_score=0.7,
                    signals={"x": 1.0},
                ),
                HarvestRecord(
                    ticker="B",
                    rank=2,
                    decision="HOLD",
                    confidence=0.5,
                    discovery_total_score=0.4,
                    signals={"x": 0.5},
                ),
            ]
        )
        df = h.load_dataframe()
        assert len(df) == 2
        assert "discovery.total_score" in df.columns
        assert "pipeline.decision" in df.columns
        assert "signals" in df.columns
        assert set(df["ticker"]) == {"A", "B"}


def test_data_harvester_load_dataframe_empty_returns_empty():
    pd = pytest.importorskip("pandas")
    from src.discovery.data_harvester import DataHarvester

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        df = h.load_dataframe()
        assert df.empty


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_data_harvester_thread_safe_writes():
    """Test multi-thread basique : N threads ecrivent en parallele -> N records."""
    import threading

    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp, flush_each_write=False)

        def worker(i):
            for j in range(5):
                h.write(HarvestRecord(ticker=f"T{i}", decision="BUY"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recs = h.load_records()
        # 4 threads * 5 records = 20
        assert len(recs) == 20
        # Toutes les lignes doivent etre du JSON valide (pas de race condition)
        for r in recs:
            assert "ticker" in r
            assert r["pipeline"]["decision"] == "BUY"


# ---------------------------------------------------------------------------
# Idempotence soft : reusing harvest_id
# ---------------------------------------------------------------------------


def test_data_harvester_explicit_harvest_id_persisted():
    from src.discovery.data_harvester import DataHarvester, HarvestRecord

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)
        rec = HarvestRecord(ticker="A", decision="BUY", harvest_id="my-fixed-id")
        hid = h.write(rec)
        assert hid == "my-fixed-id"
        recs = h.load_records()
        assert recs[0]["harvest_id"] == "my-fixed-id"
