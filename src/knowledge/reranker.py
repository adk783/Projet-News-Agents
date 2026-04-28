"""
reranker.py — Cross-encoder reranking pour RAG.

PROBLEME
--------
Les embeddings bi-encoder (MiniLM, bge-small, ...) compressent chaque
document en un vecteur independant du query. C'est rapide a l'inference
(~ms pour 1M docs) mais la qualite de la mesure de pertinence est limitee :
on ne sait pas si le "match" est profond (causal, factuel) ou superficiel
(co-occurence de tokens).

IDEE (cross-encoder)
--------------------
Un cross-encoder prend (query, doc) en INPUT COMMUN et calcule un score
de pertinence dense via full cross-attention. C'est ~100x plus lent mais
~30-50% plus precis sur les benchmarks (nDCG, MRR).

Pattern two-stage classique :
  1. Retrieval (bi-encoder) : top-K candidats (K=50-200)
  2. Reranking (cross-encoder) : trie ces candidats, garde top-k (k=5-10)

REFERENCES
----------
- Nogueira, R. & Cho, K. (2019). "Passage Re-ranking with BERT." arXiv:1901.04085
- Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks." EMNLP 2019.
- Xiao, S. et al. (2024). "C-Pack: Packaged Resources To Advance General
  Chinese Embedding." (BGE-reranker-v2-m3 : SOTA multilingual 2024).
- Thakur, N. et al. (2021). "BEIR: Heterogeneous Benchmark for Zero-shot
  IR." NeurIPS 2021.

MODELES SUPPORTES
-----------------
- BAAI/bge-reranker-v2-m3 (recommande, multilingue, 568M params)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (leger, anglais uniquement)
- Fallback deterministic : similarite Jaccard sur tokens si aucun modele

USAGE
-----
    rr = CrossEncoderReranker()  # lazy load
    reranked = rr.rerank(query, documents, top_k=5)
    # documents = list[tuple[id, text]] ou list[str]
    # reranked = list[RerankResult] avec score et rang
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# Modeles par ordre de preference
DEFAULT_RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",  # SOTA multilingue 2024
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fallback leger anglais
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class RerankResult:
    doc_id: str
    text: str
    score: float  # Score de pertinence du cross-encoder
    original_rank: int  # Rang avant reranking
    new_rank: int  # Rang apres reranking

    def summary(self) -> str:
        return f"#{self.new_rank} (was #{self.original_rank}) score={self.score:.3f}"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Reranker two-stage avec cross-encoder.

    Initialisation lazy : le modele n'est charge qu'au premier appel a
    `.rerank()`. Si aucun modele dispo, degrade vers un scorer deterministic
    base sur la similarite Jaccard + longueur.

    Thread-safety : non. Utiliser un reranker par thread si parallelise.
    """

    def __init__(
        self,
        model_candidates: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        # Distinguer "None -> defaults" de "[] -> aucun modele voulu"
        if model_candidates is None:
            self.model_candidates = list(DEFAULT_RERANKER_MODELS)
        else:
            self.model_candidates = list(model_candidates)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device  # None = auto (cuda si dispo)

        self._model = None
        self._model_name: Optional[str] = None
        self._init_attempted = False
        self._is_fallback = False

    # -- Initialisation --

    def _ensure_model(self) -> bool:
        """Charge le premier modele disponible. Retourne True si un vrai
        modele est charge, False si on utilise le fallback."""
        if self._init_attempted:
            return self._model is not None and not self._is_fallback

        self._init_attempted = True

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:
            logger.warning(
                "[Rerank] sentence-transformers non installe. "
                "Fallback Jaccard active. pip install sentence-transformers"
            )
            self._is_fallback = True
            return False

        for name in self.model_candidates:
            try:
                self._model = CrossEncoder(
                    name,
                    max_length=self.max_length,
                    device=self.device,
                )
                self._model_name = name
                logger.info("[Rerank] Cross-encoder charge : %s", name)
                return True
            except Exception as e:
                logger.warning("[Rerank] Echec chargement %s : %s", name, e)
                continue

        logger.warning("[Rerank] Aucun modele cross-encoder disponible — fallback Jaccard")
        self._is_fallback = True
        return False

    # -- Scoring --

    def _score_with_model(self, query: str, texts: Sequence[str]) -> List[float]:
        """Appel cross-encoder reel."""
        pairs = [(query, t[: self.max_length * 4]) for t in texts]
        try:
            scores = self._model.predict(  # type: ignore[union-attr]
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning("[Rerank] Erreur predict cross-encoder : %s — fallback", e)
            self._is_fallback = True
            return self._score_with_fallback(query, texts)

    def _score_with_fallback(self, query: str, texts: Sequence[str]) -> List[float]:
        """
        Fallback deterministic base sur Jaccard + longueur + bonus bigrams.

        Pas aussi bon qu'un cross-encoder, mais reproductible, rapide, et
        capture une partie de la similarite token-level.
        """

        def tokenize(s: str) -> set:
            import re

            return set(re.findall(r"[a-z0-9]+", s.lower()))

        def bigrams(s: str) -> set:
            toks = s.lower().split()
            return set(zip(toks, toks[1:])) if len(toks) > 1 else set()

        q_tokens = tokenize(query)
        q_bigrams = bigrams(query)
        if not q_tokens:
            return [0.0] * len(texts)

        scores = []
        for text in texts:
            t_tokens = tokenize(text)
            if not t_tokens:
                scores.append(0.0)
                continue

            inter = len(q_tokens & t_tokens)
            union = len(q_tokens | t_tokens)
            jaccard = inter / union if union else 0.0

            t_bigrams = bigrams(text)
            bg_bonus = 0.0
            if q_bigrams and t_bigrams:
                bg_inter = len(q_bigrams & t_bigrams)
                bg_bonus = 0.2 * (bg_inter / len(q_bigrams))

            # Legere penalite si tres court ou tres long
            length_factor = 1.0
            wc = len(text.split())
            if wc < 10:
                length_factor = 0.7
            elif wc > 500:
                length_factor = 0.85

            scores.append((jaccard + bg_bonus) * length_factor)

        return scores

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Score brut d'une liste de textes par rapport a un query."""
        if not texts:
            return []
        self._ensure_model()
        if self._model is not None and not self._is_fallback:
            return self._score_with_model(query, texts)
        return self._score_with_fallback(query, texts)

    # -- Reranking --

    def rerank(
        self,
        query: str,
        documents: Union[Sequence[str], Sequence[Tuple[str, str]]],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank une liste de documents.

        Args:
            query     : texte de la requete
            documents : soit list[str], soit list[tuple(doc_id, text)]
            top_k     : garde les top-k seulement. None = garde tout.

        Returns:
            List[RerankResult] triee par score decroissant.
        """
        if not documents:
            return []

        # Normaliser en list[(doc_id, text)]
        pairs: List[Tuple[str, str]] = []
        for i, d in enumerate(documents):
            if isinstance(d, tuple) and len(d) == 2:
                pairs.append((str(d[0]), str(d[1])))
            else:
                pairs.append((f"doc_{i}", str(d)))

        texts = [p[1] for p in pairs]
        scores = self.score(query, texts)

        indexed = list(enumerate(zip(pairs, scores)))  # [(orig_rank, ((id, text), score))]
        indexed.sort(key=lambda x: x[1][1], reverse=True)

        results: List[RerankResult] = []
        for new_rank, (orig_rank, ((doc_id, text), score)) in enumerate(indexed):
            results.append(
                RerankResult(
                    doc_id=doc_id,
                    text=text,
                    score=float(score),
                    original_rank=orig_rank,
                    new_rank=new_rank,
                )
            )

        if top_k is not None and top_k > 0:
            results = results[:top_k]
        return results

    def info(self) -> dict:
        """Info sur l'etat actuel du reranker."""
        return {
            "model_name": self._model_name,
            "is_fallback": self._is_fallback,
            "init_attempted": self._init_attempted,
            "candidates": list(self.model_candidates),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "Apple Q3 earnings beat expectations"
    docs = [
        ("d1", "Microsoft reports strong Azure growth in cloud segment"),
        ("d2", "Apple quarterly revenue exceeds estimates driven by services"),
        ("d3", "Federal Reserve signals rate pause in June meeting"),
        ("d4", "Apple Q3 earnings beat Wall Street consensus by 5%"),
        ("d5", "Tim Cook announces new iPhone launch event"),
    ]

    rr = CrossEncoderReranker()
    results = rr.rerank(query, docs, top_k=3)
    logger.info(f"Reranker info: {rr.info()}")
    logger.info(f"Query: {query}\n")
    for r in results:
        logger.info(f"  {r.summary()}  [{r.doc_id}]  {r.text[:70]}")

    # Test avec list[str] simple
    logger.info("\n-- Test list[str] --")
    r2 = rr.rerank(
        "buyback dividend",
        [
            "Apple announces $90B share buyback and dividend increase",
            "Tesla launches new model",
            "Microsoft Azure grew 25%",
        ],
    )
    for r in r2:
        logger.info(f"  {r.summary()}  {r.text[:60]}")
