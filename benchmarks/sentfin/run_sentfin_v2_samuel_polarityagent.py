import argparse
import os
import sys
import time

from benchmark_utils import (
    add_common_args,
    build_input,
    evaluate_and_save,
    load_sentfin,
    normalize_label,
)


def load_samuel_agent(repo_root=None):
    """
    Charge le PolarityAgent de la branche samuel.
    Si repo_root est donné, il doit pointer vers le dossier du repo en branche samuel.
    """
    if repo_root:
        repo_root = os.path.abspath(repo_root)
        sys.path.insert(0, repo_root)

    try:
        from polarity_agent import PolarityAgent
        return PolarityAgent(), "samuel_polarity_agent"
    except Exception as e:
        print("Impossible d'importer PolarityAgent depuis la branche Samuel.")
        print(f"Erreur : {e}")
        print("Fallback : utilisation directe de ProsusAI/finbert.")
        from transformers import pipeline

        class FinbertFallback:
            def __init__(self):
                self.clf = pipeline("text-classification", model="ProsusAI/finbert")

            def predict(self, text):
                result = self.clf(text, truncation=True)[0]
                label = result["label"].lower()
                score = float(result["score"])

                if label == "positive":
                    polarity = 1
                elif label == "negative":
                    polarity = -1
                else:
                    polarity = 0

                return polarity, score, label

        return FinbertFallback(), "samuel_polarity_fallback_finbert"


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--repo-root", default=None, help="Chemin vers le repo GitHub en branche samuel")
    args = parser.parse_args()

    output_file = args.output or "sentfin_v2_samuel_polarityagent_predictions.csv"
    pred_col = "pred_samuel_polarityagent"

    df = load_sentfin(args.input, limit=args.limit)

    agent, model_name = load_samuel_agent(args.repo_root)

    predictions = []
    raw_polarities = []
    raw_labels = []
    confidence_scores = []
    times = []

    for i, row in df.iterrows():
        model_input = build_input(row["text"], row["entity"])

        start = time.perf_counter()
        polarity, confidence, label = agent.predict(model_input)
        elapsed = time.perf_counter() - start

        pred = normalize_label(label)

        predictions.append(pred)
        raw_polarities.append(polarity)
        raw_labels.append(label)
        confidence_scores.append(confidence)
        times.append(elapsed)

        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(df)} lignes traitées...")

    df[pred_col] = predictions
    df["samuel_raw_polarity"] = raw_polarities
    df["samuel_raw_label"] = raw_labels
    df["samuel_confidence"] = confidence_scores
    df["inference_time_sec"] = times

    evaluate_and_save(df, pred_col, output_file, model_name=model_name)


if __name__ == "__main__":
    main()