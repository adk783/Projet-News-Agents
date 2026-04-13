import logging

import torch
from transformers import pipeline


logger = logging.getLogger(__name__)


class PolarityAgent:
    def __init__(self):
        logger.info("Chargement du modele de polarite (ProsusAI/finbert)...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=self.device,
        )

    @staticmethod
    def _convert_result(result):
        label = result["label"].lower()
        score = float(result["score"])

        if label == "positive":
            polarity = 1
        elif label == "negative":
            polarity = -1
        else:
            polarity = 0

        return polarity, score, label

    def predict(self, text):
        """
        Predit la polarite du texte.
        Retourne : (polarity (1, 0, -1), confidence_score, label_str)
        """
        text = text[:1500] if text else ""
        if not text:
            return 0, 0.0, "neutral"

        result = self.pipeline(text)[0]
        return self._convert_result(result)

    def predict_batch(self, texts, batch_size=16):
        """Prediction par batch pour accelerer les etapes de training."""
        if not texts:
            return []

        prepared_texts = [(text[:1500] if text else "") for text in texts]
        results = []

        for start in range(0, len(prepared_texts), batch_size):
            batch = prepared_texts[start:start + batch_size]
            safe_batch = [text if text else "neutral" for text in batch]
            batch_results = self.pipeline(safe_batch, batch_size=batch_size, truncation=True)

            for original_text, result in zip(batch, batch_results):
                if not original_text:
                    results.append((0, 0.0, "neutral"))
                else:
                    results.append(self._convert_result(result))

        return results
