import argparse
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

from benchmark_utils import (
    add_common_args,
    build_input,
    evaluate_and_save,
    load_sentfin,
    normalize_label,
)


MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


def predict_poc_filtrage(text, threshold=0.70):
    """
    Reproduit la logique centrale de src/agents/agent_filtrage.py :
    - découpage en chunks
    - modèle DistilRoBERTa financial news sentiment
    - si signal positive/negative avec confiance > threshold : on garde
    - sinon : neutral
    """
    if not text:
        return "neutral", None, 0.0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    if not chunks:
        return "neutral", None, 0.0

    clf = predict_poc_filtrage.clf

    scores = {"positive": 0.0, "negative": 0.0}
    best = {}

    for chunk in chunks:
        result = clf(chunk, truncation=True)[0]
        label = str(result["label"]).lower()
        score = float(result["score"])

        if label in ("positive", "negative") and score > threshold:
            scores[label] += score

            if label not in best or score > best[label][1]:
                best[label] = (label, score)

    if not any(v > 0 for v in scores.values()):
        return "neutral", None, 0.0

    dominant_label = max(scores, key=lambda k: scores[k])
    dominant_score = best[dominant_label][1]

    return dominant_label, dominant_label, dominant_score


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    output_file = args.output or "sentfin_v2_poc_filtrage_agents_predictions.csv"
    pred_col = "pred_poc_filtrage_agents"

    df = load_sentfin(args.input, limit=args.limit)

    print(f"Chargement du modèle POC-Filtrage-Agents : {MODEL_NAME}")
    predict_poc_filtrage.clf = pipeline(
        "text-classification",
        model=MODEL_NAME,
        truncation=True,
        device="cpu"
    )

    predictions = []
    raw_labels = []
    confidence_scores = []
    times = []

    for i, row in df.iterrows():
        model_input = build_input(row["text"], row["entity"])

        start = time.perf_counter()
        pred_label, raw_label, confidence = predict_poc_filtrage(
            model_input,
            threshold=args.threshold
        )
        elapsed = time.perf_counter() - start

        pred = normalize_label(pred_label)

        predictions.append(pred)
        raw_labels.append(raw_label)
        confidence_scores.append(confidence)
        times.append(elapsed)

        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(df)} lignes traitées...")

    df[pred_col] = predictions
    df["poc_raw_label"] = raw_labels
    df["poc_confidence"] = confidence_scores
    df["inference_time_sec"] = times

    evaluate_and_save(df, pred_col, output_file, model_name="poc_filtrage_agents_distilroberta")


if __name__ == "__main__":
    main()