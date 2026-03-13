import logging
from typing import Optional

from langgraph.func import task, entrypoint
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
            device="cpu"
        )
    return _scout_pipeline

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

@task
def filtrer_actualite_complete(texte_article: str) -> Optional[tuple[str, float]]:
    """
    Analyse l'intégralité du texte par fenêtrage glissant.
    Retourne un tuple (label, score) si un signal positive/negative est détecté, None sinon.
    Modèle : distilroberta-finetuned-financial-news (meilleur sur JeanBaptiste, accuracy 0.88).
    - "positive" : contexte haussier (bons résultats, hausse prévue...)
    - "negative" : contexte baissier (chute du titre, mauvais résultats...)
    - None       : article rejeté (neutre ou score de confiance insuffisant)
    """
    if not texte_article:
        return None

    scout = _get_scout_pipeline()
    chunks = text_splitter.split_text(texte_article)
    logger.info("Article divisé en %d segment(s) pour analyse.", len(chunks))

    for i, chunk in enumerate(chunks):
        resultat = scout(chunk)[0]
        label = resultat['label']
        score = resultat['score']

        if label in ("positive", "negative") and score > 0.70:
            logger.info("Signal majeur trouvé au segment %d/%d (%s à %.2f%%).", i + 1, len(chunks), label, score * 100)
            return (label, score)

    logger.info("Aucun impact stratégique détecté sur les %d segments.", len(chunks))
    return None

@entrypoint()
def workflow_filtrer_actualite(texte: str) -> Optional[tuple[str, float]]:
    """Retourne (label, score) si signal détecté, None sinon."""
    return filtrer_actualite_complete(texte).result()