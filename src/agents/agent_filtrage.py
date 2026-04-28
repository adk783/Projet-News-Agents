import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.func import entrypoint, task
from transformers import pipeline

logger = logging.getLogger(__name__)

_scout_pipeline = None


def _get_scout_pipeline():
    """Charge et retourne le pipeline DistilRoBERTa."""
    global _scout_pipeline
    if _scout_pipeline is None:
        logger.info("Chargement du modèle DistilRoBERTa Financial...")
        _scout_pipeline = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            truncation=True,
            device="cpu",
        )
    return _scout_pipeline


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
)


@task
def filtrer_actualite_complete(texte_article: str) -> Optional[tuple[str, float]]:
    """
    Analyse l'intégralité du texte par fenêtrage glissant.
    Retourne un tuple (label, score) si un signal positive/negative est détecté, None sinon.
    Modèle : distilroberta-finetuned-financial-news
    - "positive" : contexte haussier (bons résultats, hausse prévue...)
    - "negative" : contexte baissier (chute du titre, mauvais résultats...)
    - None       : article rejeté (neutre ou score de confiance insuffisant)
    """
    if not texte_article:
        return None

    scout = _get_scout_pipeline()
    chunks = text_splitter.split_text(texte_article)
    logger.info("Article divisé en %d segment(s) pour analyse.", len(chunks))

    scores: dict[str, float] = {"positive": 0.0, "negative": 0.0}
    best: dict[str, tuple[str, float]] = {}  # label → (label, best_score)

    for i, chunk in enumerate(chunks):
        resultat = scout(chunk)[0]
        label = resultat["label"].lower()
        score = resultat["score"]
        logger.debug("Segment %d/%d → %s (%.2f%%)", i + 1, len(chunks), label, score * 100)

        if label in ("positive", "negative") and score > 0.70:
            scores[label] += score
            if label not in best or score > best[label][1]:
                best[label] = (label, score)

    if not any(v > 0 for v in scores.values()):
        logger.info("Aucun impact stratégique détecté sur les %d segments.", len(chunks))
        return None

    dominant_label = max(scores, key=lambda k: scores[k])
    dominant_score = best[dominant_label][1]
    logger.info(
        "Signal dominant : %s (score cumulé=%.2f, meilleur chunk=%.2f%%).",
        dominant_label,
        scores[dominant_label],
        dominant_score * 100,
    )
    return (dominant_label, dominant_score)


@entrypoint()
def workflow_filtrer_actualite(texte: str) -> Optional[tuple[str, float]]:
    """Retourne (label, score) si signal détecté, None sinon."""
    return filtrer_actualite_complete(texte).result()
