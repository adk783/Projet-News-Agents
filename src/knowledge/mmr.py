"""
mmr.py — Maximal Marginal Relevance (MMR) diversification.

PROBLEME
--------
Le top-k d'un retrieval cosinus peut contenir plusieurs documents
quasi-identiques (meme depeche reprise par 5 sources, doublons, paraphrases).
Le LLM consomme alors du contexte sans gagner d'information nouvelle.

IDEE (Carbonell & Goldstein 1998)
----------------------------------
MMR selectionne iterativement le doc qui maximise :
    MMR(d) = λ * Sim(d, query) - (1 - λ) * max_{s in selected} Sim(d, s)

Lambda λ (diversity ratio) :
  - λ = 1.0 : 100% pertinence (aucune diversite)
  - λ = 0.7 : bon compromis (defaut BEIR)
  - λ = 0.5 : forte diversite
  - λ = 0.0 : 100% diversite (aucune pertinence)

IMPLEMENTATION
--------------
On fournit :
  - les vecteurs d'embedding des documents (ou on les calcule)
  - le vecteur du query
  - un lambda

On retourne les top-k indices selectionnes dans l'ordre MMR.

REFERENCES
----------
- Carbonell, J. & Goldstein, J. (1998). "The Use of MMR, Diversity-Based
  Reranking for Reordering Documents and Producing Summaries." SIGIR 1998.
- Goldstein, J. & Carbonell, J. (1998). "Summarization: Using MMR for
  Diversity-Based Reranking and Evaluating Summaries." TIPSTER 1998.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class MMRSelection:
    indices: List[int]  # Indices selectionnes, dans l'ordre MMR
    relevance_scores: List[float]  # Sim au query pour chaque selectionne
    diversity_penalties: List[float]  # max sim a deja-selectionne, pour chaque iteration

    def summary(self) -> str:
        return (
            f"MMR selected {len(self.indices)} items, "
            f"avg_rel={sum(self.relevance_scores) / max(1, len(self.relevance_scores)):.3f} "
            f"avg_div_penalty={sum(self.diversity_penalties) / max(1, len(self.diversity_penalties)):.3f}"
        )


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Similarite cosinus entre deux vecteurs."""
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _similarity_matrix(
    doc_vectors: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Matrice NxN de similarites cosinus entre documents."""
    n = len(doc_vectors)
    sim = [[0.0] * n for _ in range(n)]
    norms = [_norm(v) for v in doc_vectors]
    for i in range(n):
        for j in range(i, n):
            if norms[i] == 0.0 or norms[j] == 0.0:
                val = 0.0
            else:
                val = _dot(doc_vectors[i], doc_vectors[j]) / (norms[i] * norms[j])
            sim[i][j] = val
            sim[j][i] = val
    return sim


def _query_similarities(
    query_vector: Sequence[float],
    doc_vectors: Sequence[Sequence[float]],
) -> List[float]:
    qn = _norm(query_vector)
    if qn == 0.0:
        return [0.0] * len(doc_vectors)
    out = []
    for v in doc_vectors:
        dn = _norm(v)
        if dn == 0.0:
            out.append(0.0)
        else:
            out.append(_dot(query_vector, v) / (qn * dn))
    return out


# ---------------------------------------------------------------------------
# MMR core
# ---------------------------------------------------------------------------


def mmr_select(
    query_vector: Sequence[float],
    doc_vectors: Sequence[Sequence[float]],
    top_k: int,
    lambda_mult: float = 0.7,
    pre_computed_rel: Optional[Sequence[float]] = None,
    pre_computed_sim_matrix: Optional[Sequence[Sequence[float]]] = None,
) -> MMRSelection:
    """
    Selection MMR.

    Args:
        query_vector         : embedding du query
        doc_vectors          : embeddings des documents candidats
        top_k                : nombre de docs a selectionner
        lambda_mult          : ratio pertinence/diversite [0,1]. 0.7 = defaut.
        pre_computed_rel     : sim query<->doc deja calculees
        pre_computed_sim_matrix : matrice NxN deja calculee (evite le recalcul)

    Returns:
        MMRSelection avec les indices selectionnes dans l'ordre.
    """
    n = len(doc_vectors)
    if n == 0 or top_k <= 0:
        return MMRSelection(indices=[], relevance_scores=[], diversity_penalties=[])

    top_k = min(top_k, n)
    lam = max(0.0, min(1.0, lambda_mult))

    # Sim au query
    rel = list(pre_computed_rel) if pre_computed_rel is not None else _query_similarities(query_vector, doc_vectors)

    # Matrice de similarites inter-docs (utilisee a la demande)
    sim_matrix = [list(row) for row in pre_computed_sim_matrix] if pre_computed_sim_matrix is not None else None

    def doc_doc_sim(i: int, j: int) -> float:
        if sim_matrix is not None:
            return sim_matrix[i][j]
        return cosine_similarity(doc_vectors[i], doc_vectors[j])

    selected: List[int] = []
    rel_scores_selected: List[float] = []
    div_penalties: List[float] = []
    remaining = set(range(n))

    # Premiere selection : max pertinence
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    rel_scores_selected.append(rel[first])
    div_penalties.append(0.0)  # rien a comparer encore
    remaining.discard(first)

    # Iterations MMR
    while len(selected) < top_k and remaining:
        best_idx = -1
        best_score = -float("inf")
        best_max_sim = 0.0

        for i in remaining:
            max_sim_to_sel = max(doc_doc_sim(i, s) for s in selected)
            mmr_score = lam * rel[i] - (1.0 - lam) * max_sim_to_sel
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
                best_max_sim = max_sim_to_sel

        if best_idx < 0:
            break

        selected.append(best_idx)
        rel_scores_selected.append(rel[best_idx])
        div_penalties.append(best_max_sim)
        remaining.discard(best_idx)

    return MMRSelection(
        indices=selected,
        relevance_scores=rel_scores_selected,
        diversity_penalties=div_penalties,
    )


# ---------------------------------------------------------------------------
# Variante : MMR via scores pre-calcules (si pas d'embeddings sous la main)
# ---------------------------------------------------------------------------


def mmr_select_from_scores(
    relevance: Sequence[float],
    pairwise_similarity: Sequence[Sequence[float]],
    top_k: int,
    lambda_mult: float = 0.7,
) -> MMRSelection:
    """
    Variante MMR sans embeddings : on passe directement les scores
    pre-calcules (utile quand on a deja la matrice d'un reranker/Jaccard).

    Args:
        relevance           : rel[i] = score query<->doc[i]
        pairwise_similarity : NxN sim[i][j] = score doc[i]<->doc[j]
        top_k               : nombre d'items a selectionner
        lambda_mult         : 0.7 defaut

    Returns:
        MMRSelection
    """
    n = len(relevance)
    if n == 0 or top_k <= 0:
        return MMRSelection(indices=[], relevance_scores=[], diversity_penalties=[])
    if len(pairwise_similarity) != n:
        raise ValueError("pairwise_similarity must be NxN aligned with relevance")

    top_k = min(top_k, n)
    lam = max(0.0, min(1.0, lambda_mult))

    selected: List[int] = []
    rel_scores: List[float] = []
    div_penalties: List[float] = []
    remaining = set(range(n))

    first = max(remaining, key=lambda i: relevance[i])
    selected.append(first)
    rel_scores.append(relevance[first])
    div_penalties.append(0.0)
    remaining.discard(first)

    while len(selected) < top_k and remaining:
        best_i = -1
        best_score = -float("inf")
        best_max_sim = 0.0

        for i in remaining:
            max_sim = max(pairwise_similarity[i][s] for s in selected)
            score = lam * relevance[i] - (1.0 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
                best_max_sim = max_sim

        if best_i < 0:
            break

        selected.append(best_i)
        rel_scores.append(relevance[best_i])
        div_penalties.append(best_max_sim)
        remaining.discard(best_i)

    return MMRSelection(
        indices=selected,
        relevance_scores=rel_scores,
        diversity_penalties=div_penalties,
    )


# ---------------------------------------------------------------------------
# Jaccard-based diversification (pas d'embedding requis)
# ---------------------------------------------------------------------------


def _tokenize_simple(s: str) -> set:
    import re

    return set(re.findall(r"[a-z0-9]+", s.lower()))


def mmr_select_jaccard(
    query: str,
    texts: Sequence[str],
    top_k: int,
    lambda_mult: float = 0.7,
) -> MMRSelection:
    """
    MMR en Jaccard pur — pas besoin d'embedding. Moins precis mais zero
    dependance et instantane. Utile comme fallback dans les environnements
    sans GPU / sentence-transformers.
    """
    n = len(texts)
    if n == 0 or top_k <= 0:
        return MMRSelection(indices=[], relevance_scores=[], diversity_penalties=[])

    q_toks = _tokenize_simple(query)
    doc_toks = [_tokenize_simple(t) for t in texts]

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    rel = [jaccard(q_toks, d) for d in doc_toks]
    sim_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = jaccard(doc_toks[i], doc_toks[j])
            sim_mat[i][j] = v
            sim_mat[j][i] = v

    return mmr_select_from_scores(rel, sim_mat, top_k, lambda_mult=lambda_mult)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario : 5 docs sur Apple dont 3 quasi-identiques (reprises de depeche)
    docs = [
        "Apple Q3 revenue 89.5B beat estimates",
        "Apple Q3 revenue beats at 89.5 billion usd",  # quasi-doublon
        "Apple quarterly revenue 89.5B exceeds consensus",  # quasi-doublon
        "Microsoft Azure grew 25% in cloud segment",
        "Federal Reserve pauses rate hike in June",
    ]
    query = "Apple earnings results"

    # MMR Jaccard pur
    logger.info("=== MMR Jaccard (lambda=0.7) ===")
    sel = mmr_select_jaccard(query, docs, top_k=3, lambda_mult=0.7)
    logger.info(f"Selected indices : {sel.indices}")
    for i, idx in enumerate(sel.indices):
        print(
            f"  #{i}  [idx={idx}]  rel={sel.relevance_scores[i]:.2f}  "
            f"div_pen={sel.diversity_penalties[i]:.2f}  {docs[idx][:60]}"
        )
    print(sel.summary())

    logger.info("\n=== MMR Jaccard (lambda=0.3, diversite forte) ===")
    sel2 = mmr_select_jaccard(query, docs, top_k=3, lambda_mult=0.3)
    for i, idx in enumerate(sel2.indices):
        print(
            f"  #{i}  [idx={idx}]  rel={sel2.relevance_scores[i]:.2f}  "
            f"div_pen={sel2.diversity_penalties[i]:.2f}  {docs[idx][:60]}"
        )

    # MMR avec vecteurs factices
    logger.info("\n=== MMR avec embeddings fictifs ===")
    import random

    random.seed(42)
    query_vec = [random.random() for _ in range(8)]
    doc_vecs = [[random.random() for _ in range(8)] for _ in range(5)]
    sel3 = mmr_select(query_vec, doc_vecs, top_k=3, lambda_mult=0.7)
    logger.info(f"Selected : {sel3.indices}")
    print(sel3.summary())
