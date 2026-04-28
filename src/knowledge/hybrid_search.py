"""
hybrid_search.py — Hybrid retrieval BM25 + dense avec Reciprocal Rank Fusion.

PROBLEME
--------
Les embeddings denses (MiniLM, BGE) sont tres bons pour les paraphrases et
la similarite semantique, mais ils perdent souvent la precision lexicale :
  - Ticker exact (AAPL vs APPL/AAP)
  - Entite nommee rare (CEO peu connu)
  - Chiffres precis (89.5B vs 9.5B)

Le BM25 est l'inverse : excellent sur les matchs exacts et les tokens
rares (grace a IDF), mauvais sur les paraphrases.

IDEE (hybrid retrieval)
-----------------------
Combiner les deux et FUSIONNER les resultats. Plusieurs strategies :
  - Weighted score : alpha*bm25 + (1-alpha)*dense (necessite normalisation)
  - Reciprocal Rank Fusion (RRF) : somme 1/(k+rang_i) sur chaque source
  - CombSUM / CombMNZ : sommer ou ponderer par le nombre de sources matches

On implemente **RRF** (Cormack et al. 2009) car :
  1. Pas besoin de normaliser les scores (ils sont sur echelles differentes)
  2. Robuste empiriquement (BEIR, MS MARCO, TREC)
  3. Un seul hyperparam k (defaut 60 dans le papier)

IMPLEMENTATION
--------------
- BM25 : rank_bm25 si disponible, sinon implementation pure Python.
- Dense : on recoit les resultats denses deja tries (du RAG existant).
- RRF : fusion.

REFERENCES
----------
- Cormack, G. et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods." SIGIR 2009.
- Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance
  Framework: BM25 and Beyond." Foundations and Trends in IR.
- Ma, X. et al. (2024). "Fine-Tuning LLaMA for Multi-Stage Text Retrieval."
  (hybrid retrieval baseline).
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# Hyperparam RRF : k=60 par defaut (cf. Cormack et al. 2009)
RRF_K_DEFAULT = 60


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    """Resultat d'un retrieval (dense ou lexical)."""

    doc_id: str
    text: str
    score: float
    rank: int = 0
    source: str = ""  # "bm25" | "dense" | "hybrid"
    metadata: Dict = field(default_factory=dict)


@dataclass
class HybridResult:
    doc_id: str
    text: str
    rrf_score: float
    ranks: Dict[str, int] = field(default_factory=dict)  # {"bm25": 3, "dense": 1}
    scores: Dict[str, float] = field(default_factory=dict)
    final_rank: int = 0

    def summary(self) -> str:
        rank_str = " ".join(f"{k}#{v}" for k, v in sorted(self.ranks.items()))
        return f"#{self.final_rank} rrf={self.rrf_score:.4f} {rank_str}"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


class BM25Index:
    """
    Index BM25 pur Python (fallback si rank_bm25 indisponible).

    Formule BM25Plus (Lv & Zhai 2011) :
        score(q, d) = sum_t [ IDF(t) * (tf(t,d) * (k1+1)) / (tf + k1*(1 - b + b * len_d / avg_len)) ]
        IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs_tokens: List[List[str]] = []
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._df: Dict[str, int] = {}
        self._N: int = 0
        self._idf: Dict[str, float] = {}
        self._use_external = False
        self._external_bm25 = None

        # Tenter rank_bm25
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._BM25Okapi = BM25Okapi
            self._use_external = True
        except ImportError:
            self._BM25Okapi = None

    def build(self, documents: Sequence[Tuple[str, str]]) -> None:
        """
        Args:
            documents : list[(doc_id, text)]
        """
        self._doc_ids = [d[0] for d in documents]
        self._doc_texts = [d[1] for d in documents]
        self._docs_tokens = [_tokenize(d[1]) for d in documents]
        self._doc_lens = [len(t) for t in self._docs_tokens]
        self._N = len(documents)
        self._avgdl = sum(self._doc_lens) / self._N if self._N > 0 else 0.0

        if self._use_external and self._BM25Okapi is not None:
            try:
                self._external_bm25 = self._BM25Okapi(self._docs_tokens, k1=self.k1, b=self.b)
                return
            except Exception as e:
                logger.warning("[Hybrid] rank_bm25 build echec : %s — fallback pur", e)
                self._use_external = False

        # Fallback pur Python : precompute IDF
        df: Dict[str, int] = {}
        for tokens in self._docs_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._idf = {t: math.log((self._N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        """Retourne top_k SearchHit tries par score decroissant."""
        if self._N == 0:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        if self._use_external and self._external_bm25 is not None:
            try:
                scores = self._external_bm25.get_scores(q_tokens)
            except Exception as e:
                logger.warning("[Hybrid] rank_bm25 score echec : %s — fallback", e)
                self._use_external = False
                scores = self._score_fallback(q_tokens)
        else:
            scores = self._score_fallback(q_tokens)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        indexed = indexed[:top_k]

        results: List[SearchHit] = []
        for rank, (i, s) in enumerate(indexed):
            if s <= 0.0:
                continue
            results.append(
                SearchHit(
                    doc_id=self._doc_ids[i],
                    text=self._doc_texts[i],
                    score=float(s),
                    rank=rank,
                    source="bm25",
                )
            )
        return results

    def _score_fallback(self, q_tokens: List[str]) -> List[float]:
        """Score BM25 pur Python."""
        scores: List[float] = [0.0] * self._N
        if self._avgdl == 0.0:
            return scores

        for i, doc_tokens in enumerate(self._docs_tokens):
            if not doc_tokens:
                continue
            tf_counter = Counter(doc_tokens)
            doc_len = self._doc_lens[i]
            acc = 0.0
            for t in q_tokens:
                tf = tf_counter.get(t, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(t, 0.0)
                norm = 1.0 - self.b + self.b * (doc_len / self._avgdl)
                acc += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
            scores[i] = acc
        return scores


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Dict[str, Sequence[SearchHit]],
    k: int = RRF_K_DEFAULT,
    weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
) -> List[HybridResult]:
    """
    Fusion de plusieurs rankings par RRF.

    Formule :
        RRF_score(d) = sum_i [ w_i / (k + rank_i(d)) ]

    Args:
        rankings : dict {"source_name": list[SearchHit]}
                   chaque SearchHit doit avoir un `rank` coherent (0 = meilleur)
        k        : constante RRF (60 defaut, cf. Cormack 2009)
        weights  : ponderation par source (None = equal). ex: {"dense": 0.6, "bm25": 0.4}
        top_k    : tronque a top_k resultats. None = tout.

    Returns:
        List[HybridResult] triee par rrf_score decroissant.
    """
    if not rankings:
        return []

    if weights is None:
        weights = {src: 1.0 for src in rankings}

    acc: Dict[str, HybridResult] = {}

    for source, hits in rankings.items():
        w = weights.get(source, 1.0)
        for hit in hits:
            rank = hit.rank
            contrib = w / (k + rank + 1)  # +1 pour rank 1-based dans RRF classique

            if hit.doc_id not in acc:
                acc[hit.doc_id] = HybridResult(
                    doc_id=hit.doc_id,
                    text=hit.text,
                    rrf_score=0.0,
                )
            entry = acc[hit.doc_id]
            entry.rrf_score += contrib
            entry.ranks[source] = rank
            entry.scores[source] = hit.score

    results = sorted(acc.values(), key=lambda h: h.rrf_score, reverse=True)
    for i, r in enumerate(results):
        r.final_rank = i

    if top_k is not None and top_k > 0:
        results = results[:top_k]
    return results


# ---------------------------------------------------------------------------
# Hybrid searcher (glue code)
# ---------------------------------------------------------------------------


class HybridSearcher:
    """
    Orchestrateur pour retrieval hybride.

    Usage :
        hs = HybridSearcher()
        hs.index_corpus(docs)  # docs = list[(doc_id, text)]
        dense_hits = [SearchHit(..., source="dense", rank=...), ...]  # via RAG existant
        hybrid = hs.search("query", dense_hits=dense_hits, top_k=5)
    """

    def __init__(
        self,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = RRF_K_DEFAULT,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._corpus_indexed = False

    def index_corpus(self, documents: Sequence[Tuple[str, str]]) -> None:
        self.bm25.build(documents)
        self._corpus_indexed = True
        logger.info("[Hybrid] BM25 index build : %d docs", len(documents))

    def search(
        self,
        query: str,
        dense_hits: Optional[Sequence[SearchHit]] = None,
        top_k: int = 10,
        bm25_top_k: int = 50,
    ) -> List[HybridResult]:
        """
        Fusion BM25 + dense.

        Args:
            query         : texte de la requete
            dense_hits    : resultats d'un retrieval dense (ChromaDB/RAG), deja tries
            top_k         : nombre final de resultats apres fusion
            bm25_top_k    : nombre de candidats BM25 avant fusion (K_bm25 > top_k)
        """
        rankings: Dict[str, Sequence[SearchHit]] = {}

        # BM25 hits
        if self._corpus_indexed:
            bm25_hits = self.bm25.search(query, top_k=bm25_top_k)
            rankings["bm25"] = bm25_hits
        else:
            logger.debug("[Hybrid] BM25 non indexe, fusion dense seule")

        # Dense hits
        if dense_hits:
            dense_norm = []
            for i, h in enumerate(dense_hits):
                h2 = SearchHit(
                    doc_id=h.doc_id,
                    text=h.text,
                    score=h.score,
                    rank=h.rank if h.rank else i,
                    source="dense",
                    metadata=dict(h.metadata) if h.metadata else {},
                )
                dense_norm.append(h2)
            rankings["dense"] = dense_norm

        if not rankings:
            return []

        weights = {"bm25": self.bm25_weight, "dense": self.dense_weight}
        return reciprocal_rank_fusion(
            rankings,
            k=self.rrf_k,
            weights=weights,
            top_k=top_k,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus = [
        ("d1", "Apple reports Q3 revenue of 89.5 billion USD beating estimates"),
        ("d2", "Microsoft Azure cloud revenue grew 25% in the quarter"),
        ("d3", "Apple iPhone sales decline 5% in Greater China region"),
        ("d4", "Federal Reserve holds rates steady at 5.25% amid inflation"),
        ("d5", "Apple services segment reaches record 75% gross margin"),
        ("d6", "Tesla Q3 delivers 466K vehicles missing analyst estimates"),
        ("d7", "Apple announces 90B share buyback program for shareholders"),
        ("d8", "Nvidia data center revenue surges 400% on AI demand"),
    ]

    hs = HybridSearcher()
    hs.index_corpus(corpus)

    query = "Apple earnings Q3"

    # Simuler des dense_hits (eg. du RAG ChromaDB)
    # Ici on simule : dense match = d1, d5, d3 (semantic) mais pas d7 (lexical rare)
    dense_hits = [
        SearchHit(doc_id="d1", text=corpus[0][1], score=0.92, rank=0, source="dense"),
        SearchHit(doc_id="d5", text=corpus[4][1], score=0.78, rank=1, source="dense"),
        SearchHit(doc_id="d3", text=corpus[2][1], score=0.71, rank=2, source="dense"),
    ]

    # BM25 devrait ramener d1, d3, d5, d7 (match lexical "Apple")
    logger.info("=== BM25 only ===")
    bm25_hits = hs.bm25.search(query, top_k=5)
    for h in bm25_hits:
        logger.info(f"  #{h.rank}  [{h.doc_id}]  score={h.score:.3f}  {h.text[:60]}")

    logger.info("\n=== Hybrid (RRF) ===")
    hybrid = hs.search(query, dense_hits=dense_hits, top_k=5)
    for r in hybrid:
        logger.info(f"  {r.summary()}  {r.doc_id}")

    # Test que d1 est en tete (match sur les deux sources)
    logger.info("\n=== Validation : top result doit etre d1 (Apple Q3 revenue) ===")
    assert hybrid[0].doc_id == "d1", f"Expected d1 first, got {hybrid[0].doc_id}"
    # d7 (buyback) doit remonter grace a BM25 meme s'il n'est pas en dense_hits
    top_ids = [r.doc_id for r in hybrid]
    assert "d7" in top_ids, "BM25 should surface d7 (buyback Apple lexical match)"
    logger.info("OK - BM25 remonte d7 (buyback Apple) et d1 reste #1")
