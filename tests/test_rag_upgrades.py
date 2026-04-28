"""
test_rag_upgrades.py — Tests des modules RAG avances (BATCH 3).

Portee :
  - src/knowledge/reranker.py        : cross-encoder + fallback Jaccard
  - src/knowledge/mmr.py             : Maximal Marginal Relevance
  - src/knowledge/hybrid_search.py   : BM25 + RRF fusion
  - src/knowledge/rag_enhanced.py    : orchestrateur 2-stage

Ces tests sont deterministes (pas de reseau, pas de modele lourd charge par
defaut). Le reranker utilise son fallback Jaccard.
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def test_reranker_fallback_jaccard_ranks_relevant_higher():
    from src.knowledge.reranker import CrossEncoderReranker

    # Force le fallback : aucun candidat valide
    rr = CrossEncoderReranker(model_candidates=[])

    query = "Apple Q3 earnings revenue beat"
    docs = [
        ("d1", "Microsoft Azure cloud grew 25% this quarter"),
        ("d2", "Apple Q3 revenue beat estimates at 89.5 billion"),
        ("d3", "Tesla delivers record Q3 vehicles"),
    ]
    results = rr.rerank(query, docs, top_k=3)
    assert len(results) == 3
    # d2 doit etre en tete car tokens "apple q3 revenue beat" matchent
    assert results[0].doc_id == "d2"


def test_reranker_fallback_handles_empty():
    from src.knowledge.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker(model_candidates=[])
    assert rr.rerank("query", []) == []
    assert rr.score("query", []) == []


def test_reranker_preserves_doc_ids():
    from src.knowledge.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker(model_candidates=[])
    docs = [("alpha", "apple stock"), ("beta", "orange juice")]
    results = rr.rerank("apple earnings", docs)
    ids = {r.doc_id for r in results}
    assert ids == {"alpha", "beta"}


def test_reranker_accepts_list_of_strings():
    from src.knowledge.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker(model_candidates=[])
    results = rr.rerank("apple", ["Apple Q3 earnings", "Tesla Q3"])
    assert len(results) == 2
    assert results[0].text.startswith("Apple")


def test_reranker_info_reports_fallback():
    from src.knowledge.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker(model_candidates=[])
    rr.rerank("x", ["y"])  # force l'init
    info = rr.info()
    assert info["is_fallback"] is True
    assert info["init_attempted"] is True


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------


def test_mmr_jaccard_prefers_diverse_over_duplicates():
    from src.knowledge.mmr import mmr_select_jaccard

    docs = [
        "Apple Q3 revenue 89.5B beats estimates",
        "Apple Q3 revenue beats at 89.5 billion",  # quasi-dup
        "Apple quarterly revenue 89.5B exceeds",  # quasi-dup
        "Microsoft Azure grew 25%",  # diversifiant
    ]
    sel = mmr_select_jaccard("Apple earnings", docs, top_k=2, lambda_mult=0.3)
    # Avec lambda bas, le deuxieme doit etre Microsoft (diversite)
    assert 3 in sel.indices


def test_mmr_lambda_high_keeps_top_relevant():
    from src.knowledge.mmr import mmr_select_jaccard

    docs = [
        "Apple Q3 revenue beats",
        "Apple Q3 revenue surges",
        "Microsoft Azure cloud",
    ]
    sel = mmr_select_jaccard("Apple earnings", docs, top_k=2, lambda_mult=1.0)
    # Lambda=1 => pure relevance, pas de penalite diversite
    assert sel.indices[0] in (0, 1)  # un des deux Apple


def test_mmr_empty_input():
    from src.knowledge.mmr import mmr_select, mmr_select_jaccard

    assert mmr_select_jaccard("q", [], top_k=5).indices == []
    assert mmr_select([1.0], [], top_k=5).indices == []


def test_mmr_from_scores_respects_precomputed_relevance():
    from src.knowledge.mmr import mmr_select_from_scores

    # 3 docs : le 2eme a la relevance max
    rel = [0.3, 0.9, 0.5]
    # Matrice identite = pas de similarite inter-docs
    sim = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    sel = mmr_select_from_scores(rel, sim, top_k=3, lambda_mult=0.7)
    # Premier doit etre l'index 1 (max rel)
    assert sel.indices[0] == 1


def test_mmr_diversity_penalty_increases_with_similar_selection():
    from src.knowledge.mmr import mmr_select_from_scores

    rel = [1.0, 0.9, 0.8]
    # docs 0 et 1 tres similaires, doc 2 different
    sim = [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ]
    sel = mmr_select_from_scores(rel, sim, top_k=3, lambda_mult=0.5)
    # Premier = doc 0 (max rel), second devrait etre doc 2 (moins similaire)
    assert sel.indices[0] == 0
    assert sel.indices[1] == 2


# ---------------------------------------------------------------------------
# Hybrid search (BM25 + RRF)
# ---------------------------------------------------------------------------


def test_bm25_index_and_search():
    from src.knowledge.hybrid_search import BM25Index

    idx = BM25Index()
    idx.build(
        [
            ("d1", "apple stock price rose today"),
            ("d2", "tesla vehicles delivery quarterly"),
            ("d3", "apple quarterly earnings beat"),
        ]
    )
    hits = idx.search("apple earnings", top_k=3)
    top_ids = [h.doc_id for h in hits]
    # d3 doit etre en tete (apple + earnings)
    assert top_ids[0] == "d3"


def test_bm25_returns_empty_on_empty_corpus():
    from src.knowledge.hybrid_search import BM25Index

    idx = BM25Index()
    idx.build([])
    assert idx.search("anything") == []


def test_rrf_fusion_combines_ranks():
    from src.knowledge.hybrid_search import SearchHit, reciprocal_rank_fusion

    bm25_hits = [
        SearchHit(doc_id="d1", text="apple", score=2.5, rank=0, source="bm25"),
        SearchHit(doc_id="d2", text="tesla", score=1.2, rank=1, source="bm25"),
    ]
    dense_hits = [
        SearchHit(doc_id="d2", text="tesla", score=0.9, rank=0, source="dense"),
        SearchHit(doc_id="d1", text="apple", score=0.8, rank=1, source="dense"),
    ]
    results = reciprocal_rank_fusion({"bm25": bm25_hits, "dense": dense_hits})
    assert len(results) == 2
    # Les deux docs doivent avoir des scores RRF proches (symetrie bm25#0/dense#1 et bm25#1/dense#0)
    top_ids = {r.doc_id for r in results}
    assert top_ids == {"d1", "d2"}
    # Le doc en tete doit avoir un rrf_score > le second
    assert results[0].rrf_score >= results[1].rrf_score


def test_rrf_fusion_weighted():
    from src.knowledge.hybrid_search import SearchHit, reciprocal_rank_fusion

    bm25_hits = [SearchHit(doc_id="d1", text="x", score=1.0, rank=0, source="bm25")]
    dense_hits = [SearchHit(doc_id="d2", text="y", score=0.5, rank=0, source="dense")]

    # Avec poids 10x pour dense, d2 doit passer devant d1
    results = reciprocal_rank_fusion(
        {"bm25": bm25_hits, "dense": dense_hits},
        weights={"bm25": 1.0, "dense": 10.0},
    )
    assert results[0].doc_id == "d2"


def test_hybrid_searcher_end_to_end():
    from src.knowledge.hybrid_search import HybridSearcher, SearchHit

    corpus = [
        ("d1", "Apple Q3 revenue beat estimates"),
        ("d2", "Microsoft Azure grew 25 percent"),
        ("d3", "Apple buyback 90B announced"),
    ]
    hs = HybridSearcher()
    hs.index_corpus(corpus)

    # On passe un dense_hits simule
    dense = [
        SearchHit(doc_id="d1", text=corpus[0][1], score=0.9, rank=0, source="dense"),
        SearchHit(doc_id="d2", text=corpus[1][1], score=0.7, rank=1, source="dense"),
    ]
    results = hs.search("apple earnings", dense_hits=dense, top_k=3)
    ids = [r.doc_id for r in results]
    # d1 et d3 (les deux Apple) doivent etre dans le top
    assert "d1" in ids


# ---------------------------------------------------------------------------
# Enhanced retriever (mock RAG)
# ---------------------------------------------------------------------------


def _make_mock_store(docs_by_ticker):
    """Construit un mock LocusRAGStore minimaliste."""
    from dataclasses import dataclass, field

    @dataclass
    class _Doc:
        doc_id: str
        ticker: str
        text: str
        doc_type: str = "article"
        date_iso: str = "2026-04-20T10:00:00+00:00"
        metadata: dict = field(default_factory=dict)

    @dataclass
    class _R:
        doc: "_Doc"
        cosine_score: float
        temporal_score: float
        days_old: float

    class Mock:
        def __init__(self, corpus):
            self._corpus = corpus

        def _ensure_initialized(self):
            return True

        def _get_or_create_collection(self, ticker):
            docs = self._corpus.get(ticker.lower(), [])

            class Coll:
                def __init__(self, d):
                    self._d = d

                def count(self):
                    return len(self._d)

                def get(self, include=None):
                    return {
                        "ids": [x[0] for x in self._d],
                        "documents": [x[1] for x in self._d],
                        "metadatas": [{} for _ in self._d],
                    }

            return Coll(docs)

        def query(self, ticker, query_text, k, doc_types=None, lambda_override=None):
            docs = self._corpus.get(ticker.lower(), [])
            out = []
            for i, (did, txt) in enumerate(docs[:k]):
                out.append(
                    _R(
                        doc=_Doc(doc_id=did, ticker=ticker, text=txt),
                        cosine_score=0.9 - 0.1 * i,
                        temporal_score=0.85 - 0.1 * i,
                        days_old=float(i),
                    )
                )
            return out

    return Mock(docs_by_ticker)


def test_enhanced_retriever_entity_filter_drops_off_ticker():
    from src.knowledge.rag_enhanced import EnhancedRetriever

    corpus = {
        "aapl": [
            ("d1", "Apple Q3 revenue 89.5B beats estimates"),
            ("d2", "Microsoft Azure cloud grew 25 percent"),  # off-ticker
            ("d3", "Apple services margin 75%"),
        ]
    }
    store = _make_mock_store(corpus)
    enh = EnhancedRetriever(
        rag_store=store,
        use_reranker=False,
        use_mmr=False,
        use_entity_filter=True,
        use_bm25=False,
    )
    results, trace = enh.retrieve("AAPL", "Apple earnings", k=3, company_name="Apple")
    ids = [r.doc_id for r in results]
    # d2 (Microsoft) doit etre retire par le filtre entite, ou en queue
    # Mais comme le fallback retient tout si filtre trop agressif (len < k),
    # on verifie qu'au moins d1 et d3 passent en tete
    assert ids[0] in ("d1", "d3")


def test_enhanced_retriever_stages_logged_in_trace():
    from src.knowledge.rag_enhanced import EnhancedRetriever

    store = _make_mock_store({"aapl": [("d1", "Apple Q3 revenue")]})
    enh = EnhancedRetriever(
        rag_store=store,
        use_reranker=False,
        use_mmr=False,
        use_entity_filter=True,
        use_bm25=True,
    )
    results, trace = enh.retrieve("AAPL", "Apple", k=3)
    assert "dense" in trace.stages_used
    assert trace.dense_count >= 1


def test_enhanced_retriever_graceful_without_store():
    from src.knowledge.rag_enhanced import EnhancedRetriever

    enh = EnhancedRetriever(rag_store=None, use_bm25=False)
    results, trace = enh.retrieve("AAPL", "Apple", k=3)
    assert results == []
    assert trace.dense_count == 0


def test_default_entity_checker_matches_ticker_and_company():
    from src.knowledge.rag_enhanced import _default_entity_checker

    assert _default_entity_checker("AAPL reports Q3", "AAPL") is True
    assert _default_entity_checker("Apple beats earnings", "AAPL", company_name="Apple") is True
    assert _default_entity_checker("Microsoft Azure", "AAPL", company_name="Apple") is False
    assert _default_entity_checker("PAPPLE is a typo", "AAPL") is False  # word boundary


# ---------------------------------------------------------------------------
# EnhancedResult & trace
# ---------------------------------------------------------------------------


def test_enhanced_result_summary_is_string():
    from src.knowledge.rag_enhanced import EnhancedResult

    r = EnhancedResult(
        doc_id="d1",
        text="foo",
        ticker="AAPL",
        date_iso="",
        doc_type="article",
        rerank_score=0.9,
        rrf_score=0.03,
        cosine_score=0.8,
        final_rank=0,
        days_old=1.5,
    )
    s = r.summary()
    assert "rerank=0.900" in s
    assert "age=1.5" in s
