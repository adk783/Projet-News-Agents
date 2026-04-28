"""
test_cold_start.py — Tests pour le cold-start orchestrator.

Couvre :
  - is_cold (detection bas-niveau)
  - bootstrap : profile synthetique
  - bootstrap : SEC backfill
  - bootstrap : earnings backfill
  - bootstrap : news backfill
  - bootstrap : peer fallback
  - idempotence (re-run sur ticker chaud)
  - cache TTL et invalidation
  - default_peer_resolver
  - graceful failure si fetcher absent
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Fixtures : mock RAG store, mock fundamentals, mock SEC, mock fetchers
# ---------------------------------------------------------------------------


class _MockDoc:
    def __init__(self, doc_id, ticker, text, doc_type, date_iso, metadata=None):
        self.doc_id = doc_id
        self.ticker = ticker
        self.text = text
        self.doc_type = doc_type
        self.date_iso = date_iso
        self.metadata = metadata or {}


class _MockResult:
    def __init__(self, doc):
        self.doc = doc
        self.cosine_score = 0.5
        self.temporal_score = 0.5
        self.days_old = 1.0


class _MockRAGStore:
    def __init__(self):
        self.docs: Dict[str, List[_MockDoc]] = {}

    def collection_size(self, ticker):
        return len(self.docs.get(ticker.upper(), []))

    def add_document(self, doc):
        t = doc.ticker.upper()
        self.docs.setdefault(t, []).append(
            _MockDoc(
                doc_id=doc.doc_id,
                ticker=t,
                text=doc.text,
                doc_type=doc.doc_type,
                date_iso=doc.date_iso,
                metadata=doc.metadata,
            )
        )
        return True

    def query(self, ticker, query_text, k=5):
        return [_MockResult(d) for d in self.docs.get(ticker.upper(), [])[:k]]


class _MockFiling:
    def __init__(self, form, date, desc, acc):
        self.found = True
        self.form_type = form
        self.filing_date = date
        self.company_name = "ACME Corp"
        self.cik = "0000123456"
        self.accession_number = acc
        self.description = desc


class _MockEdgar:
    def __init__(self, n_filings: int = 3):
        self.n_filings = n_filings

    def find_recent_8k(self, ticker, lookback_days=365):
        return [_MockFiling("10-Q", "2026-02-12", f"Filing {i}", f"00-{i}") for i in range(self.n_filings)]


class _MockFundamentals:
    company_name = "ACME Corp"
    sector = "Technology"
    industry = "Software"
    market_cap = 100_000_000_000
    pe_trailing = 25.0
    pe_forward = 22.0
    revenue_ttm = 50_000_000_000
    revenue_growth_yoy = 0.12
    gross_margin = 0.55
    operating_margin = 0.25
    net_margin = 0.18
    return_on_equity = 0.30
    eps_trailing = 5.5
    eps_forward = 6.2
    debt_to_equity = 0.4
    current_ratio = 2.1
    free_cash_flow = 8_000_000_000
    dividend_yield = 0.015
    analyst_consensus = "Buy"
    analyst_mean_target = 200.0
    analyst_upside = 0.15
    n_analysts = 28


def _mock_fundamentals_fn(ticker: str):
    return _MockFundamentals() if ticker.upper() == "ACME" else None


def _mock_earnings_fn(ticker: str) -> List[Tuple[str, str]]:
    if ticker.upper() != "ACME":
        return []
    return [
        ("2026-02-12T16:00:00+00:00", "Q4 2025 earnings call. " * 30),
        ("2025-11-10T16:00:00+00:00", "Q3 2025 earnings call. " * 30),
    ]


def _mock_news_fn(ticker: str, days: int) -> List[Dict[str, Any]]:
    if ticker.upper() != "ACME":
        return []
    return [
        {
            "title": f"ACME news {i}",
            "summary": f"Detailed analysis of ACME quarter results edition {i}.",
            "url": f"https://x.com/{i}",
            "date_iso": "2026-02-13T09:00:00+00:00",
            "source": "Reuters",
        }
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_is_cold_when_collection_empty():
    from src.knowledge.cold_start import ColdStartManager

    csm = ColdStartManager(rag_store=_MockRAGStore(), min_docs=5)
    assert csm.is_cold("NEWX") is True


def test_is_cold_false_when_collection_has_enough_docs():
    from src.knowledge.cold_start import ColdStartManager

    rag = _MockRAGStore()
    for i in range(10):
        rag.docs.setdefault("HOT", []).append(
            _MockDoc(
                f"d{i}",
                "HOT",
                f"t{i}",
                "article",
                "2026-01-01T00:00:00+00:00",
            )
        )
    csm = ColdStartManager(rag_store=rag, min_docs=5)
    assert csm.is_cold("HOT") is False


def test_count_documents_zero_when_no_store():
    from src.knowledge.cold_start import ColdStartManager

    csm = ColdStartManager(rag_store=None)
    assert csm.count_documents("ANY") == 0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_indexes_synthetic_profile():
    from src.knowledge.cold_start import (
        DOC_TYPE_COLD_PROFILE,
        ColdStartManager,
    )

    rag = _MockRAGStore()
    csm = ColdStartManager(
        rag_store=rag,
        fundamentals_fn=_mock_fundamentals_fn,
        min_docs=5,
    )
    rep = csm.bootstrap("ACME", company_name="ACME Corp", sector="Technology")
    assert rep.profile_indexed is True
    assert any(d.doc_type == DOC_TYPE_COLD_PROFILE for d in rag.docs.get("ACME", []))


def test_bootstrap_backfills_sec_filings():
    from src.knowledge.cold_start import (
        DOC_TYPE_COLD_SEC,
        ColdStartManager,
    )

    rag = _MockRAGStore()
    csm = ColdStartManager(
        rag_store=rag,
        edgar_client=_MockEdgar(n_filings=4),
        min_docs=10,  # force le bootstrap
    )
    rep = csm.bootstrap("ACME", force=True)
    sec_added = sum(s.documents_added for s in rep.sources if s.name == "sec_edgar")
    assert sec_added == 4
    assert any(d.doc_type == DOC_TYPE_COLD_SEC for d in rag.docs.get("ACME", []))


def test_bootstrap_backfills_earnings_calls():
    from src.knowledge.cold_start import (
        DOC_TYPE_COLD_CALL,
        ColdStartManager,
    )

    rag = _MockRAGStore()
    csm = ColdStartManager(
        rag_store=rag,
        earnings_fetcher=_mock_earnings_fn,
        min_docs=10,
    )
    rep = csm.bootstrap("ACME", force=True)
    call_added = sum(s.documents_added for s in rep.sources if s.name == "earnings_calls")
    assert call_added == 2
    assert any(d.doc_type == DOC_TYPE_COLD_CALL for d in rag.docs.get("ACME", []))


def test_bootstrap_backfills_news():
    from src.knowledge.cold_start import (
        DOC_TYPE_COLD_NEWS,
        ColdStartManager,
    )

    rag = _MockRAGStore()
    csm = ColdStartManager(
        rag_store=rag,
        news_fetcher=_mock_news_fn,
        min_docs=10,
    )
    rep = csm.bootstrap("ACME", force=True)
    news_added = sum(s.documents_added for s in rep.sources if s.name == "historical_news")
    assert news_added == 3
    assert any(d.doc_type == DOC_TYPE_COLD_NEWS for d in rag.docs.get("ACME", []))


def test_bootstrap_full_pipeline_resolves_cold():
    from src.knowledge.cold_start import ColdStartManager

    rag = _MockRAGStore()
    csm = ColdStartManager(
        rag_store=rag,
        fundamentals_fn=_mock_fundamentals_fn,
        edgar_client=_MockEdgar(n_filings=3),
        earnings_fetcher=_mock_earnings_fn,
        news_fetcher=_mock_news_fn,
        min_docs=5,
    )
    rep = csm.bootstrap("ACME", company_name="ACME Corp")
    assert rep.was_cold is True
    assert rep.is_cold_after is False
    assert rep.docs_after >= 5
    assert rep.profile_indexed


def test_bootstrap_idempotent_on_warm_ticker():
    from src.knowledge.cold_start import ColdStartManager

    rag = _MockRAGStore()
    # Pre-warm avec 10 docs
    for i in range(10):
        rag.docs.setdefault("HOT", []).append(
            _MockDoc(
                f"d{i}",
                "HOT",
                f"t{i}",
                "article",
                "2026-01-01T00:00:00+00:00",
            )
        )
    csm = ColdStartManager(
        rag_store=rag,
        fundamentals_fn=_mock_fundamentals_fn,
        min_docs=5,
    )
    rep = csm.bootstrap("HOT")
    assert rep.was_cold is False
    assert rep.docs_after == 10  # rien d'ajoute


# ---------------------------------------------------------------------------
# Peer fallback
# ---------------------------------------------------------------------------


def test_peer_fallback_when_no_other_sources_succeed():
    from src.knowledge.cold_start import (
        DOC_TYPE_COLD_PEER,
        ColdStartManager,
    )

    rag = _MockRAGStore()
    # Pre-warm un peer
    for i in range(8):
        rag.docs.setdefault("PEER", []).append(
            _MockDoc(
                f"p{i}",
                "PEER",
                f"peer text {i}",
                "article",
                "2026-01-15T00:00:00+00:00",
            )
        )

    csm = ColdStartManager(
        rag_store=rag,
        peer_resolver=lambda t: ["PEER"],
        min_docs=3,
        peer_fallback=True,
    )
    rep = csm.bootstrap("COLD")
    assert "PEER" in rep.peers_used
    assert any(d.doc_type == DOC_TYPE_COLD_PEER for d in rag.docs.get("COLD", []))


def test_peer_fallback_disabled_leaves_cold():
    from src.knowledge.cold_start import ColdStartManager

    rag = _MockRAGStore()
    for i in range(8):
        rag.docs.setdefault("PEER", []).append(
            _MockDoc(
                f"p{i}",
                "PEER",
                f"peer {i}",
                "article",
                "2026-01-01T00:00:00+00:00",
            )
        )
    csm = ColdStartManager(
        rag_store=rag,
        peer_resolver=lambda t: ["PEER"],
        min_docs=3,
        peer_fallback=False,
    )
    rep = csm.bootstrap("COLD")
    assert rep.peers_used == []
    assert rep.is_cold_after is True


def test_peer_seeding_avoids_transitive_chains():
    """Les docs deja seedes par d'autres peers ne doivent pas etre re-seedes."""
    from src.knowledge.cold_start import (
        DOC_TYPE_COLD_PEER,
        ColdStartManager,
    )

    rag = _MockRAGStore()
    rag.docs["PEER"] = [
        _MockDoc("legit-1", "PEER", "good content", "article", "2026-01-15T00:00:00+00:00"),
        _MockDoc("seed-from-other", "PEER", "[Peer-seed from X] foo", DOC_TYPE_COLD_PEER, "2026-01-15T00:00:00+00:00"),
    ]
    csm = ColdStartManager(
        rag_store=rag,
        peer_resolver=lambda t: ["PEER"],
        min_docs=3,
        peer_fallback=True,
    )
    csm.bootstrap("COLD")
    # Les docs seedes vers COLD ne doivent pas inclure le DOC_TYPE_COLD_PEER source
    seeded_texts = [d.text for d in rag.docs.get("COLD", []) if d.doc_type == DOC_TYPE_COLD_PEER]
    for t in seeded_texts:
        assert "[Peer-seed from X]" not in t


def test_default_peer_resolver_known_tickers():
    from src.knowledge.cold_start import default_peer_resolver

    assert "MSFT" in default_peer_resolver("AAPL")
    assert "CVX" in default_peer_resolver("XOM")
    assert default_peer_resolver("UNKNOWN_XYZ") == []


def test_default_peer_resolver_case_insensitive():
    from src.knowledge.cold_start import default_peer_resolver

    assert default_peer_resolver("aapl") == default_peer_resolver("AAPL")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_is_cold_uses_cache_within_ttl():
    from src.knowledge.cold_start import ColdStartManager

    rag = _MockRAGStore()
    csm = ColdStartManager(rag_store=rag, min_docs=5)

    # 1er appel : cold
    assert csm.is_cold("X") is True

    # On rajoute des docs sans invalider le cache
    for i in range(20):
        rag.docs.setdefault("X", []).append(
            _MockDoc(
                f"d{i}",
                "X",
                f"t{i}",
                "article",
                "2026-01-01",
            )
        )
    # Toujours cold (cache hit)
    assert csm.is_cold("X") is True

    # Apres invalidation
    csm.invalidate_cache("X")
    assert csm.is_cold("X") is False


def test_force_refresh_bypasses_cache():
    from src.knowledge.cold_start import ColdStartManager

    rag = _MockRAGStore()
    csm = ColdStartManager(rag_store=rag, min_docs=5)
    csm.is_cold("X")  # cache cold=True
    for i in range(20):
        rag.docs.setdefault("X", []).append(
            _MockDoc(
                f"d{i}",
                "X",
                f"t{i}",
                "article",
                "2026-01-01",
            )
        )
    assert csm.is_cold("X", force_refresh=True) is False


# ---------------------------------------------------------------------------
# Graceful failure
# ---------------------------------------------------------------------------


def test_bootstrap_works_with_no_optional_components():
    from src.knowledge.cold_start import ColdStartManager

    csm = ColdStartManager(rag_store=_MockRAGStore(), min_docs=5)
    rep = csm.bootstrap("UNKNOWN", company_name="Unknown")
    # rien d'ajoute, mais ne doit pas raise
    assert rep.was_cold is True
    assert rep.is_cold_after is True
    assert rep.profile_indexed is False
    assert rep.docs_after == 0


def test_bootstrap_fundamentals_returning_none_is_safe():
    from src.knowledge.cold_start import ColdStartManager

    csm = ColdStartManager(
        rag_store=_MockRAGStore(),
        fundamentals_fn=lambda t: None,
        min_docs=5,
    )
    rep = csm.bootstrap("UNKNOWN")
    assert rep.profile_indexed is False


def test_bootstrap_handles_fetcher_exceptions():
    from src.knowledge.cold_start import ColdStartManager

    def boom_news(ticker, days):
        raise RuntimeError("API down")

    csm = ColdStartManager(
        rag_store=_MockRAGStore(),
        news_fetcher=boom_news,
        min_docs=5,
    )
    rep = csm.bootstrap("X")
    news_src = next((s for s in rep.sources if s.name == "historical_news"), None)
    assert news_src is not None
    assert news_src.success is False
    assert news_src.error  # message present


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def test_report_to_prompt_block_ascii_safe():
    from src.knowledge.cold_start import ColdStartManager

    csm = ColdStartManager(rag_store=_MockRAGStore(), min_docs=5)
    rep = csm.bootstrap("X")
    block = rep.to_prompt_block()
    block.encode("cp1252")
    assert "<cold_start_report>" in block
    assert "</cold_start_report>" in block
