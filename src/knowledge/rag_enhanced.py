"""
rag_enhanced.py — Pipeline RAG 2-stage avec hybrid search + reranker + MMR.

ARCHITECTURE
------------
Le retrieval "standard" du projet utilise MiniLM-L6-v2 + ChromaDB cosine. C'est
rapide mais :
  1. Les embeddings bi-encoder manquent les matchs lexicaux rares (ticker, chiffres)
  2. Le top-k peut contenir des quasi-doublons (meme depeche de Reuters reprise)
  3. Pas de filtre par entite (on peut ramener des articles sur MSFT meme si
     le query est sur AAPL, si la similarite cosinus est haute)

Ce module ajoute un pipeline 2-stage :
  1. Stage 1 (recall) : ChromaDB dense + BM25 fusionnes par RRF (top-50)
  2. Stage 2 (precision) :
     a) Filtre entite : drop les docs ne mentionnant pas le ticker/company
     b) Cross-encoder reranker : trie les candidats restants
     c) MMR : diversifie le top-k final

PLUG-IN, NON INVASIF
--------------------
On ne modifie PAS LocusRAGStore. On l'enveloppe :

    store = LocusRAGStore()
    enh = EnhancedRetriever(store, reranker=..., entity_checker=...)
    results = enh.retrieve("AAPL", "Apple Q3 results", k=5)

REFERENCES
----------
- Karpukhin, V. et al. (2020). "Dense Passage Retrieval for Open-Domain QA." EMNLP
- Nogueira, R. et al. (2019). "Passage Re-ranking with BERT."
- Cormack, G. et al. (2009). "Reciprocal Rank Fusion."
- Carbonell, J. & Goldstein, J. (1998). "Maximal Marginal Relevance."
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Imports absolus (cf. ADR-008, mode editable `pip install -e .`).
from src.knowledge.hybrid_search import (
    HybridResult,
    HybridSearcher,
    SearchHit,
    reciprocal_rank_fusion,
)
from src.knowledge.mmr import (
    MMRSelection,
    _tokenize_simple,
    mmr_select_from_scores,
    mmr_select_jaccard,
)
from src.knowledge.reranker import CrossEncoderReranker, RerankResult

# rag_store : fallback None si la dep chromadb n'est pas installee.
try:
    from src.knowledge.rag_store import DEFAULT_K, LocusRAGStore, RAGResult
except ImportError:
    LocusRAGStore = None  # type: ignore
    RAGResult = None  # type: ignore
    DEFAULT_K = 5


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class EnhancedResult:
    """Resultat du pipeline enhanced."""

    doc_id: str
    text: str
    ticker: str
    date_iso: str
    doc_type: str

    # Scores intermediaires pour l'observabilite
    cosine_score: float = 0.0
    temporal_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    # Ranks successifs (pour debug)
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    final_rank: int = -1

    # Flags de filtrage
    entity_mentioned: bool = True
    mmr_diversity_penalty: float = 0.0

    days_old: float = 0.0
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"#{self.final_rank} rerank={self.rerank_score:.3f} "
            f"rrf={self.rrf_score:.3f} cosine={self.cosine_score:.3f} "
            f"age={self.days_old:.1f}j ent={self.entity_mentioned}"
        )


@dataclass
class RetrievalTrace:
    """Trace de debug du pipeline."""

    dense_count: int = 0
    bm25_count: int = 0
    rrf_count: int = 0
    entity_kept: int = 0
    reranked_count: int = 0
    mmr_final: int = 0
    stages_used: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"stages={','.join(self.stages_used)} "
            f"dense={self.dense_count} bm25={self.bm25_count} "
            f"rrf={self.rrf_count} entity_kept={self.entity_kept} "
            f"reranked={self.reranked_count} final={self.mmr_final}"
        )


# ---------------------------------------------------------------------------
# Entity checker par defaut
# ---------------------------------------------------------------------------


def _default_entity_checker(
    text: str,
    ticker: str,
    company_name: Optional[str] = None,
) -> bool:
    """
    Verifie la mention du ticker ou du nom de la societe.

    Strategie :
      - Ticker en UPPERCASE avec boundaries (AAPL, pas PAPPLE)
      - Company name en case-insensitive
      - Retourne True si au moins une des mentions presentes
    """
    if not text:
        return False

    t = text
    if ticker:
        import re

        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(ticker.upper())}(?![A-Za-z])")
        if pattern.search(t):
            return True

    if company_name:
        if company_name.lower() in t.lower():
            return True

    return False


# ---------------------------------------------------------------------------
# Enhanced retriever
# ---------------------------------------------------------------------------


class EnhancedRetriever:
    """
    Pipeline RAG 2-stage :
      1. Hybrid retrieval : dense (ChromaDB via LocusRAGStore) + BM25 -> RRF
      2. Entity filter : drop les docs ne mentionnant pas le ticker/company
      3. Reranker cross-encoder : trie les candidats restants
      4. MMR : diversifie le top-k

    Chaque stage est OPTIONNEL : on peut desactiver le reranker si le modele
    n'est pas dispo, desactiver la diversification MMR, etc.
    """

    def __init__(
        self,
        rag_store: Optional[LocusRAGStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        hybrid: Optional[HybridSearcher] = None,
        entity_checker: Optional[Callable[[str, str, Optional[str]], bool]] = None,
        use_reranker: bool = True,
        use_mmr: bool = True,
        use_entity_filter: bool = True,
        use_bm25: bool = True,
        mmr_lambda: float = 0.7,
        rerank_pool_size: int = 30,
        dense_pool_size: int = 20,
    ):
        self.rag_store = rag_store
        self.reranker = reranker if reranker is not None else (CrossEncoderReranker() if use_reranker else None)
        self.hybrid = hybrid if hybrid is not None else HybridSearcher()
        self.entity_checker = entity_checker or _default_entity_checker
        self.use_reranker = use_reranker
        self.use_mmr = use_mmr
        self.use_entity_filter = use_entity_filter
        self.use_bm25 = use_bm25
        self.mmr_lambda = max(0.0, min(1.0, mmr_lambda))
        self.rerank_pool_size = rerank_pool_size
        self.dense_pool_size = dense_pool_size

        self._corpus_indexed_ticker: Optional[str] = None

    # -- Index corpus pour BM25 --

    def index_ticker_corpus(
        self,
        ticker: str,
        force_rebuild: bool = False,
    ) -> int:
        """
        Peuple l'index BM25 avec tout le corpus d'un ticker.

        On itere la collection ChromaDB via l'API sous-jacente.
        Return : nombre de documents indexes.
        """
        if not self.use_bm25:
            return 0
        if self._corpus_indexed_ticker == ticker and not force_rebuild:
            return 0
        if self.rag_store is None or not hasattr(self.rag_store, "_ensure_initialized"):
            return 0
        if not self.rag_store._ensure_initialized():
            return 0

        try:
            coll = self.rag_store._get_or_create_collection(ticker)
            if coll.count() == 0:
                return 0
            # ChromaDB : recupere tout le corpus
            dump = coll.get(include=["documents", "metadatas"])
            docs_list: List[Tuple[str, str]] = []
            for i in range(len(dump.get("documents", []))):
                doc_text = dump["documents"][i] if dump.get("documents") else ""
                doc_id = dump["ids"][i] if dump.get("ids") else f"bm25_{i}"
                if doc_text:
                    docs_list.append((doc_id, doc_text))

            self.hybrid.index_corpus(docs_list)
            self._corpus_indexed_ticker = ticker
            logger.info("[EnhancedRAG] BM25 indexe %d docs pour %s", len(docs_list), ticker)
            return len(docs_list)

        except Exception as e:
            logger.warning("[EnhancedRAG] BM25 index echec (%s) — dense seul", e)
            return 0

    # -- Pipeline principal --

    def retrieve(
        self,
        ticker: str,
        query_text: str,
        k: int = DEFAULT_K,
        company_name: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
    ) -> Tuple[List[EnhancedResult], RetrievalTrace]:
        """
        Pipeline complet. Retourne (results, trace).
        """
        trace = RetrievalTrace()

        # -- Stage 0 : Dense retrieval via LocusRAGStore --
        rag_results: List = []
        if self.rag_store is not None:
            try:
                rag_results = self.rag_store.query(
                    ticker=ticker,
                    query_text=query_text,
                    k=self.dense_pool_size,
                    doc_types=doc_types,
                )
                trace.dense_count = len(rag_results)
                trace.stages_used.append("dense")
            except Exception as e:
                logger.warning("[EnhancedRAG] dense retrieval echec : %s", e)

        # Convertir en SearchHit pour fusion
        dense_hits: List[SearchHit] = []
        for i, r in enumerate(rag_results):
            dense_hits.append(
                SearchHit(
                    doc_id=r.doc.doc_id,
                    text=r.doc.text,
                    score=r.cosine_score,
                    rank=i,
                    source="dense",
                    metadata={
                        "temporal_score": r.temporal_score,
                        "days_old": r.days_old,
                        "date_iso": r.doc.date_iso,
                        "doc_type": r.doc.doc_type,
                        "ticker": r.doc.ticker,
                    },
                )
            )

        # -- Stage 1 : BM25 + RRF --
        hybrid_results: List[HybridResult] = []
        if self.use_bm25:
            try:
                self.index_ticker_corpus(ticker)
                hybrid_results = self.hybrid.search(
                    query=query_text,
                    dense_hits=dense_hits,
                    top_k=self.rerank_pool_size,
                    bm25_top_k=self.rerank_pool_size,
                )
                trace.bm25_count = sum(1 for _ in hybrid_results if "bm25" in _.ranks)
                trace.rrf_count = len(hybrid_results)
                if "bm25" in {s for r in hybrid_results for s in r.ranks}:
                    trace.stages_used.append("bm25")
                trace.stages_used.append("rrf")
            except Exception as e:
                logger.warning("[EnhancedRAG] hybrid fusion echec : %s — dense only", e)

        if not hybrid_results:
            # Fallback : convertir dense_hits en HybridResult
            for i, h in enumerate(dense_hits[: self.rerank_pool_size]):
                hybrid_results.append(
                    HybridResult(
                        doc_id=h.doc_id,
                        text=h.text,
                        rrf_score=1.0 / (60 + i + 1),
                        ranks={"dense": i},
                        scores={"dense": h.score},
                        final_rank=i,
                    )
                )
                hybrid_results[-1].__dict__["_metadata"] = h.metadata

        # On recupere la metadata originale (temporal_score, days_old) depuis dense_hits
        dense_meta_by_id = {h.doc_id: h.metadata for h in dense_hits}

        # -- Stage 2a : Entity filter --
        entity_kept: List[HybridResult] = []
        if self.use_entity_filter:
            for r in hybrid_results:
                if self.entity_checker(r.text, ticker, company_name):
                    entity_kept.append(r)
            trace.entity_kept = len(entity_kept)
            trace.stages_used.append("entity_filter")
            # Si entity filter supprime trop, on fallback sur top-3 de hybrid_results
            if len(entity_kept) < max(3, k):
                logger.debug(
                    "[EnhancedRAG] entity filter trop agressif (%d < %d), conservation top-%d",
                    len(entity_kept),
                    k,
                    max(3, k),
                )
                seen = {r.doc_id for r in entity_kept}
                for r in hybrid_results:
                    if r.doc_id not in seen:
                        entity_kept.append(r)
                    if len(entity_kept) >= max(3, k):
                        break
        else:
            entity_kept = list(hybrid_results)

        # -- Stage 2b : Rerank cross-encoder --
        reranked_docs: List[Tuple[HybridResult, float]] = []
        if self.use_reranker and self.reranker is not None and entity_kept:
            try:
                pairs = [(r.doc_id, r.text) for r in entity_kept]
                rerank_out = self.reranker.rerank(query_text, pairs, top_k=None)
                score_by_id = {r.doc_id: r.score for r in rerank_out}
                rank_by_id = {r.doc_id: r.new_rank for r in rerank_out}
                reranked_docs = sorted(
                    [(r, score_by_id.get(r.doc_id, 0.0)) for r in entity_kept], key=lambda x: x[1], reverse=True
                )
                trace.reranked_count = len(reranked_docs)
                trace.stages_used.append("rerank")
            except Exception as e:
                logger.warning("[EnhancedRAG] rerank echec : %s", e)
                reranked_docs = [(r, r.rrf_score) for r in entity_kept]
                rank_by_id = {r.doc_id: i for i, r in enumerate(entity_kept)}
        else:
            reranked_docs = [(r, r.rrf_score) for r in entity_kept]
            rank_by_id = {r.doc_id: i for i, r in enumerate(entity_kept)}

        # -- Stage 3 : MMR diversification --
        #
        # On utilise les scores du reranker (si dispos) comme RELEVANCE,
        # et Jaccard textuelle comme PAIRWISE SIMILARITY. Ca permet a MMR
        # de preserver l'ordre du cross-encoder tout en diversifiant.
        selected_indices: List[int]
        mmr_penalties: List[float] = []
        if self.use_mmr and len(reranked_docs) > k:
            # Relevance = rerank scores (deja normalises [0,1] pour le cross-encoder)
            rerank_scores = [pair[1] for pair in reranked_docs]
            # Si tous les scores sont ~0 (fallback Jaccard du reranker), on
            # bascule sur mmr_select_jaccard qui recalcule rel depuis le texte.
            max_rel = max(rerank_scores) if rerank_scores else 0.0
            if max_rel <= 0.01:
                texts = [pair[0].text for pair in reranked_docs]
                mmr_sel = mmr_select_jaccard(
                    query=query_text,
                    texts=texts,
                    top_k=k,
                    lambda_mult=self.mmr_lambda,
                )
            else:
                # Matrice de similarite pairwise sur tokens (Jaccard fallback)
                texts = [pair[0].text for pair in reranked_docs]
                tok_sets = [_tokenize_simple(t) for t in texts]
                n = len(texts)
                sim_mat = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i, n):
                        a, b = tok_sets[i], tok_sets[j]
                        if not a or not b:
                            v = 0.0
                        else:
                            u = len(a | b)
                            v = len(a & b) / u if u else 0.0
                        sim_mat[i][j] = v
                        sim_mat[j][i] = v
                mmr_sel = mmr_select_from_scores(
                    relevance=rerank_scores,
                    pairwise_similarity=sim_mat,
                    top_k=k,
                    lambda_mult=self.mmr_lambda,
                )
            selected_indices = mmr_sel.indices
            mmr_penalties = list(mmr_sel.diversity_penalties)
            trace.stages_used.append("mmr")
        else:
            selected_indices = list(range(min(k, len(reranked_docs))))
            mmr_penalties = [0.0] * len(selected_indices)

        # -- Construction EnhancedResult --
        final: List[EnhancedResult] = []
        for final_rank, idx in enumerate(selected_indices):
            if idx >= len(reranked_docs):
                continue
            hr, rerank_score = reranked_docs[idx]
            meta = dense_meta_by_id.get(hr.doc_id, {})
            dense_rank = hr.ranks.get("dense")
            bm25_rank = hr.ranks.get("bm25")

            final.append(
                EnhancedResult(
                    doc_id=hr.doc_id,
                    text=hr.text,
                    ticker=str(meta.get("ticker", ticker)),
                    date_iso=str(meta.get("date_iso", "")),
                    doc_type=str(meta.get("doc_type", "article")),
                    cosine_score=hr.scores.get("dense", 0.0),
                    temporal_score=float(meta.get("temporal_score", 0.0)),
                    bm25_score=hr.scores.get("bm25", 0.0),
                    rrf_score=hr.rrf_score,
                    rerank_score=float(rerank_score),
                    dense_rank=dense_rank,
                    bm25_rank=bm25_rank,
                    rrf_rank=hr.final_rank,
                    rerank_rank=rank_by_id.get(hr.doc_id),
                    final_rank=final_rank,
                    entity_mentioned=self.entity_checker(hr.text, ticker, company_name)
                    if self.use_entity_filter
                    else True,
                    mmr_diversity_penalty=mmr_penalties[final_rank] if final_rank < len(mmr_penalties) else 0.0,
                    days_old=float(meta.get("days_old", 0.0)),
                    metadata={},
                )
            )

        trace.mmr_final = len(final)
        return final, trace


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test sans ChromaDB reel : on simule via un RAG mock.
    class MockRAGStore:
        def __init__(self):
            self._initialized = True
            self._docs = {
                "aapl": [
                    ("d1", "Apple Q3 revenue 89.5B beat estimates on services margin expansion"),
                    ("d2", "Apple AAPL reports quarterly earnings above Wall Street consensus"),
                    ("d3", "Microsoft Azure cloud revenue grew 25%"),  # mauvais ticker
                    ("d4", "Apple services segment reaches 75% gross margin record"),
                    ("d5", "Apple iPhone sales decline 5% in China as macro pressures rise"),
                    ("d6", "Apple announces 90B share buyback program for AAPL shareholders"),
                ]
            }

        def _ensure_initialized(self):
            return True

        def _get_or_create_collection(self, ticker):
            class FakeColl:
                def __init__(self, docs):
                    self._docs = docs

                def count(self):
                    return len(self._docs)

                def get(self, include=None):
                    return {
                        "ids": [d[0] for d in self._docs],
                        "documents": [d[1] for d in self._docs],
                        "metadatas": [{} for _ in self._docs],
                    }

            return FakeColl(self._docs.get(ticker.lower(), []))

        def query(self, ticker, query_text, k, doc_types=None, lambda_override=None):
            # Retourne pseudo-RAGResult factices
            from dataclasses import dataclass

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
                doc: _Doc
                cosine_score: float
                temporal_score: float
                days_old: float

            docs = self._docs.get(ticker.lower(), [])
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

    store = MockRAGStore()
    enh = EnhancedRetriever(
        rag_store=store,
        use_reranker=True,  # tombera en fallback Jaccard si pas de modele
        use_mmr=True,
        use_entity_filter=True,
        use_bm25=True,
        mmr_lambda=0.7,
    )

    query = "Apple earnings Q3"
    results, trace = enh.retrieve("AAPL", query, k=3, company_name="Apple")
    logger.info(f"Trace : {trace.summary()}")
    logger.info(f"Stages used : {trace.stages_used}")
    logger.info("\nTop-3 results :")
    for r in results:
        logger.info(f"  {r.summary()}")
        logger.info(f"    [{r.doc_id}] {r.text[:70]}")

    # Validation : d3 (Microsoft) doit etre filtre
    ids = [r.doc_id for r in results]
    assert "d3" not in ids, "Entity filter echec : d3 (Microsoft) retenu"
    logger.info("\nOK - d3 (Microsoft) correctement filtre par entity_checker")
