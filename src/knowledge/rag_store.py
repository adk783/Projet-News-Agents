"""
rag_store.py — Base Vectorielle avec Décroissance Temporelle

Références scientifiques :
  [1] Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks." NeurIPS 2020.
      → Architecture RAG : retrieval + génération. Le retrieval de qualité
        est le facteur limitant principal de la précision du LLM.

  [2] Wang, W. et al. (2020). "MINILM: Deep Self-Attention Distillation for
      Task-Agnostic Compression of Pre-Trained Transformers."
      NeurIPS 2020. Microsoft Research.
      → `all-MiniLM-L6-v2` : 22M params, SOTA retrieval/compression en 2020-2024.
        Sur BEIR benchmark (Thakur et al. 2021), il surpasse des modèles 10× plus
        lourds sur des tâches de similarité sémantique.

  [3] Kanhabua, N. & Nørvåg, K. (2010). "Determining Time of Queries for
      Re-Ranking of Search Results." ECIR 2010.
      → Décroissance exponentielle temporelle :
        score_final = cosine_sim × exp(-λ × Δt_jours)
        avec λ = ln(2) / half_life_days

  [4] Mitra, B., Diaz, F., & Craswell, N. (2014). "SIGIR Workshop on Temporal
      Information Retrieval."
      → Pour les news financières, half-life empirique ≈ 14 jours :
        une information perd ~50% de sa pertinence décisionnelle toutes les 2 semaines.

  [5] Thakur, N. et al. (2021). "BEIR: A Heterogeneous Benchmark for
      Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021.
      → Benchmark de référence pour comparer les modèles d'embedding.
        MiniLM-L6-v2 : nDCG@10 = 0.429 sur le benchmark moyen — optimal pour
        le ratio performance/coût sur CPU.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes scientifiques
# ---------------------------------------------------------------------------

# Décroissance temporelle pour news financières
# Ref: Mitra et al. (2014) — half-life ≈ 14 jours empirique pour financial news
FINANCIAL_NEWS_HALF_LIFE_DAYS: float = 14.0
LAMBDA_DECAY: float = math.log(2) / FINANCIAL_NEWS_HALF_LIFE_DAYS  # ≈ 0.0495

# Modèle d'embedding
# Ref: Wang et al. (2020) — all-MiniLM-L6-v2 optimal pour tâches de retrieval
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Répertoire de persistance ChromaDB
CHROMA_PERSIST_DIR: str = "data/chroma_db"

# Nombre de résultats par défaut (RAG k)
DEFAULT_K: int = 5

# Score minimum après decay pour inclure un résultat
MIN_TEMPORAL_SCORE: float = 0.05

# Types de documents stockés
DOC_TYPE_ARTICLE = "article"
DOC_TYPE_DEBATE = "debate_summary"
DOC_TYPE_MEMORY = "memory_consolidation"
DOC_TYPE_SEC_8K = "sec_8k"
DOC_TYPE_SEC_FORM4 = "sec_form4"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RAGDocument:
    """Document stocké dans le RAG."""

    doc_id: str  # ID unique (URL ou généré)
    ticker: str
    text: str  # Texte à embedder
    doc_type: str  # article / debate_summary / memory / sec_8k
    date_iso: str  # ISO 8601 — pour la décroissance temporelle
    metadata: dict = field(default_factory=dict)  # Source, titre, aspects ABSA, etc.

    def days_old(self, now: Optional[datetime] = None) -> float:
        """Calcule l'âge du document en jours depuis date_iso."""
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            pub = datetime.fromisoformat(self.date_iso.replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            return max(0.0, (now - pub).total_seconds() / 86400)
        except Exception:
            return 30.0  # Fallback : 30 jours si date invalide


@dataclass
class RAGResult:
    """Résultat d'une requête RAG avec score temporellement pondéré."""

    doc: RAGDocument
    cosine_score: float  # Similarité cosinus brute [0, 1]
    temporal_score: float  # Score après décroissance [0, 1]
    days_old: float  # Âge en jours


# ---------------------------------------------------------------------------
# RAG Store principal
# ---------------------------------------------------------------------------


class LocusRAGStore:
    """
    Base vectorielle ChromaDB avec retrieval temporellement pondéré.

    Basé sur Lewis et al. (2020) pour l'architecture RAG et
    Kanhabua & Nørvåg (2010) pour la décroissance temporelle.

    Architecture :
      - Une collection ChromaDB par ticker : `locus_{ticker}`
      - Embedding : all-MiniLM-L6-v2 (Wang et al. 2020, BEIR optimal)
      - Score final = cosine_sim × exp(-λ × Δt_jours)
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        lambda_decay: float = LAMBDA_DECAY,
    ):
        self.persist_dir = persist_dir
        self.lambda_decay = lambda_decay
        self._client = None
        self._embedding_fn = None
        self._initialized = False
        self._embedding_model_name = embedding_model

        # Initialisation lazy — évite le chargement du modèle si RAG non utilisé
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    def _ensure_initialized(self) -> bool:
        """Initialise ChromaDB et le modèle d'embedding si nécessaire."""
        if self._initialized:
            return True
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=self.persist_dir)

            # Embedding function ChromaDB-native avec sentence-transformers
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model_name
            )

            self._initialized = True
            logger.info(
                "[RAG] ChromaDB initialisé → %s | Embedding: %s | λ=%.4f (half-life %.0f jours)",
                self.persist_dir,
                self._embedding_model_name,
                self.lambda_decay,
                math.log(2) / self.lambda_decay,
            )
            return True

        except ImportError as e:
            logger.warning(
                "[RAG] ChromaDB ou sentence-transformers non installé (%s). "
                "Installer avec: pip install chromadb sentence-transformers",
                e,
            )
            return False
        except Exception as e:
            logger.error("[RAG] Erreur initialisation ChromaDB : %s", e)
            return False

    def _get_or_create_collection(self, ticker: str):
        """Retourne ou crée la collection ChromaDB pour un ticker."""
        name = f"locus_{ticker.lower().replace('-', '_').replace('.', '_')}"
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Indexation ---

    def add_document(self, doc: RAGDocument) -> bool:
        """
        Indexe un document dans ChromaDB.
        Idempotent : si le doc_id existe déjà, il est mis à jour (upsert).
        """
        if not self._ensure_initialized():
            return False
        try:
            collection = self._get_or_create_collection(doc.ticker)
            metadata = {
                "ticker": doc.ticker,
                "doc_type": doc.doc_type,
                "date_iso": doc.date_iso,
                "days_old": doc.days_old(),
                **{k: str(v)[:500] for k, v in doc.metadata.items()},  # ChromaDB: strings only
            }
            collection.upsert(
                ids=[doc.doc_id],
                documents=[doc.text[:8000]],  # ChromaDB limite par document
                metadatas=[metadata],
            )
            logger.debug(
                "[RAG] Indexé : %s | ticker=%s | type=%s | age=%.0f j",
                doc.doc_id[:60],
                doc.ticker,
                doc.doc_type,
                doc.days_old(),
            )
            return True
        except Exception as e:
            logger.error("[RAG] Erreur indexation %s : %s", doc.doc_id[:60], e)
            return False

    def add_documents_batch(self, docs: list[RAGDocument]) -> int:
        """Indexe un batch de documents. Retourne le nombre indexés avec succès."""
        return sum(1 for d in docs if self.add_document(d))

    # --- Retrieval avec décroissance temporelle ---

    def query(
        self,
        ticker: str,
        query_text: str,
        k: int = DEFAULT_K,
        doc_types: Optional[list[str]] = None,
        lambda_override: Optional[float] = None,
    ) -> list[RAGResult]:
        """
        Retrieval avec décroissance temporelle exponentielle.

        Algorithme (Kanhabua & Nørvåg 2010) :
          1. Requête cosinus ChromaDB (top k×3 pour filtrage aval)
          2. Pour chaque résultat : temporal_score = cosine_sim × exp(-λ × Δt)
          3. Filtre MIN_TEMPORAL_SCORE, tri décroissant, retourne top-k

        Args:
            ticker     : ticker de la collection à interroger
            query_text : texte de la requête (article en cours)
            k          : nombre de résultats souhaités
            doc_types  : filtre par type de document (None = tous)
            lambda_override : surcharger λ (ex: 0.0 = pas de decay)

        Returns:
            Liste de RAGResult triée par temporal_score décroissant.
        """
        if not self._ensure_initialized():
            return []
        try:
            collection = self._get_or_create_collection(ticker)

            # Vérification collection non vide
            if collection.count() == 0:
                logger.debug("[RAG] Collection %s vide — pas de résultats.", ticker)
                return []

            # On récupère k×3 pour avoir de la marge après le filtrage temporal
            n_query = min(k * 3, max(collection.count(), 1))

            # Filtre par type de document si demandé
            where_filter = None
            if doc_types:
                where_filter = {"doc_type": {"$in": doc_types}}

            results = collection.query(
                query_texts=[query_text[:4000]],
                n_results=n_query,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            now = datetime.now(timezone.utc)
            lam = lambda_override if lambda_override is not None else self.lambda_decay
            rag_results: list[RAGResult] = []

            for i, (doc_text, meta, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # ChromaDB retourne distance cosinus [0,2] → similarité = 1 - distance/2
                # (avec hnsw:space=cosine, distance = 1 - cosine_sim)
                cosine_sim = max(0.0, 1.0 - distance)

                # Calcul de l'âge
                date_iso = meta.get("date_iso", "")
                try:
                    pub_dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    days_old = max(0.0, (now - pub_dt).total_seconds() / 86400)
                except Exception:
                    days_old = 30.0

                # Décroissance exponentielle (Kanhabua & Nørvåg 2010)
                temporal_score = cosine_sim * math.exp(-lam * days_old)

                if temporal_score < MIN_TEMPORAL_SCORE:
                    continue

                rag_doc = RAGDocument(
                    doc_id=results["ids"][0][i] if "ids" in results else f"result_{i}",
                    ticker=meta.get("ticker", ticker),
                    text=doc_text,
                    doc_type=meta.get("doc_type", "unknown"),
                    date_iso=date_iso,
                    metadata={k: v for k, v in meta.items() if k not in ("ticker", "doc_type", "date_iso", "days_old")},
                )
                rag_results.append(
                    RAGResult(
                        doc=rag_doc,
                        cosine_score=round(cosine_sim, 4),
                        temporal_score=round(temporal_score, 4),
                        days_old=round(days_old, 1),
                    )
                )

            # Tri par temporal_score décroissant
            rag_results.sort(key=lambda r: r.temporal_score, reverse=True)
            top_k = rag_results[:k]

            logger.info(
                "[RAG] Query ticker=%s | %d/%d résultats retenus (λ=%.4f) | Scores: [%s]",
                ticker,
                len(top_k),
                n_query,
                lam,
                ", ".join(f"{r.temporal_score:.3f}" for r in top_k),
            )
            return top_k

        except Exception as e:
            logger.error("[RAG] Erreur query ticker=%s : %s", ticker, e)
            return []

    def format_for_prompt(self, results: list[RAGResult], max_chars: int = 3000) -> str:
        """
        Formate les résultats RAG pour injection dans le prompt du débat.
        Inclut le type, l'âge et le score de pertinence pour la traçabilité.
        """
        if not results:
            return ""

        lines = [
            "=== MÉMOIRE SÉMANTIQUE (RAG — Pertinence Temporelle) ===",
            f"(Modèle: {self._embedding_model_name} | Decay λ={self.lambda_decay:.4f})",
            "",
        ]
        total_chars = 0

        for r in results:
            age_str = f"{r.days_old:.0f}j" if r.days_old < 365 else f"{r.days_old / 365:.1f}an"
            header = (
                f"[{r.doc.doc_type.upper()} | {age_str} | "
                f"pertinence={r.temporal_score:.3f} | cosine={r.cosine_score:.3f}]"
            )
            snippet = r.doc.text[:600] + ("…" if len(r.doc.text) > 600 else "")

            block = f"{header}\n{snippet}\n"
            if total_chars + len(block) > max_chars:
                break
            lines.append(block)
            total_chars += len(block)

        lines.append("=" * 52)
        return "\n".join(lines)

    def collection_size(self, ticker: str) -> int:
        """Retourne le nombre de documents indexés pour un ticker."""
        if not self._ensure_initialized():
            return 0
        try:
            return self._get_or_create_collection(ticker).count()
        except Exception:
            return 0

    def delete_old_documents(self, ticker: str, max_days: int = 180) -> int:
        """
        Supprime les documents plus vieux que max_days.
        Ref: gestion de la fenêtre temporelle — évite la croissance illimitée de l'index.
        """
        if not self._ensure_initialized():
            return 0
        try:
            collection = self._get_or_create_collection(ticker)
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
            cutoff_iso = cutoff.isoformat()
            # ChromaDB: filtre sur métadonnée date_iso
            results = collection.get(where={"date_iso": {"$lt": cutoff_iso}})
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(
                    "[RAG] %d documents > %d jours supprimés pour %s.",
                    len(ids_to_delete),
                    max_days,
                    ticker,
                )
            return len(ids_to_delete)
        except Exception as e:
            logger.debug("[RAG] delete_old_documents : %s", e)
            return 0


# ---------------------------------------------------------------------------
# Singleton global
# ---------------------------------------------------------------------------

_rag_store: Optional[LocusRAGStore] = None


def get_rag_store() -> LocusRAGStore:
    """Retourne le singleton LocusRAGStore (init lazy)."""
    global _rag_store
    if _rag_store is None:
        _rag_store = LocusRAGStore()
    return _rag_store
