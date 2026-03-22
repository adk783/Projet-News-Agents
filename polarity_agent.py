import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class PolarityAgent:
    def __init__(self):
        logger.info("Chargement du modèle de polarité (ProsusAI/finbert)...")
        self.pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert"
        )
        
    def predict(self, text):
        """
        Prédit la polarité du texte.
        Retourne : (polarity (1, 0, -1), confidence_score, label_str)
        """
        text = text[:1500] if text else ""
        if not text:
            return 0, 0.0, "neutral"
            
        result = self.pipeline(text)[0]
        label = result["label"]
        score = result["score"]

        if label == "positive":
            polarity = 1
        elif label == "negative":
            polarity = -1
        else:
            polarity = 0
            
        return polarity, score, label
