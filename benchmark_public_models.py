"""
Benchmarks publics des outputs du projet
=======================================

Genere des benchmarks publics pour :
- polarity (classification sentiment)
- uncertainty (proxy benchmarks)

Les autres scores additionnels disposent deja de leurs benchmarks dans :
- fundamental_strength_model/benchmarks.json
- litigious_model/benchmarks.json
"""

import json
import os
import hashlib
from collections import Counter

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from polarity_agent import PolarityAgent
from uncertainty_agent import UncertaintyAgent


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark_results")
POLARITY_BENCHMARK_FILE = os.path.join(BENCHMARK_DIR, "polarity_benchmarks.json")
UNCERTAINTY_BENCHMARK_FILE = os.path.join(BENCHMARK_DIR, "uncertainty_benchmarks.json")

PHRASEBANK_DATASET = "financial_phrasebank"
PHRASEBANK_CONFIG = "sentences_allagree"
FIQA_DATASET = "TheFinAI/fiqa-sentiment-classification"
NICKY_DATASET = "NickyNicky/Finance_sentiment_and_topic_classification_En"
NICKY_HOLDOUT_MODULO = 5
NICKY_BENCHMARK_BUCKET = 4

POLARITY_LABEL_ORDER = ["negative", "neutral", "positive"]


def ensure_output_dir():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def stable_bucket(text: str, modulo: int = NICKY_HOLDOUT_MODULO) -> int:
    payload = (text or "").encode("utf-8", errors="ignore")
    return int(hashlib.md5(payload).hexdigest(), 16) % modulo


def sample_balanced(df: pd.DataFrame, label_column: str, max_per_class: int = None) -> pd.DataFrame:
    counts = Counter(df[label_column].tolist())
    min_count = min(counts.values())
    sample_size = min_count if max_per_class is None else min(min_count, max_per_class)
    return (
        df.groupby(label_column, group_keys=False)
        .sample(n=sample_size, random_state=42)
        .reset_index(drop=True)
    )


def fiqa_score_to_sentiment(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def evaluate_multiclass_labels(true_labels, pred_labels, label_order):
    return {
        "dataset_size": int(len(true_labels)),
        "accuracy": round(float(accuracy_score(true_labels, pred_labels)), 6),
        "macro_f1": round(float(f1_score(true_labels, pred_labels, average="macro")), 6),
        "class_distribution": dict(Counter(true_labels)),
        "prediction_distribution": dict(Counter(pred_labels)),
        "label_order": label_order,
    }


def evaluate_binary_scores(true_labels, scores, threshold_values=None):
    if threshold_values is None:
        threshold_values = [i / 100.0 for i in range(1, 96)]

    predictions_at_05 = [1 if score >= 0.5 else 0 for score in scores]
    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = 0.0

    for threshold in threshold_values:
        threshold_predictions = [1 if score >= threshold else 0 for score in scores]
        threshold_f1 = f1_score(true_labels, threshold_predictions)
        if threshold_f1 > best_f1:
            best_f1 = threshold_f1
            best_threshold = threshold
            best_accuracy = accuracy_score(true_labels, threshold_predictions)

    positive_scores = [score for score, label in zip(scores, true_labels) if label == 1]
    negative_scores = [score for score, label in zip(scores, true_labels) if label == 0]

    return {
        "dataset_size": int(len(true_labels)),
        "positive_share": round(float(sum(true_labels) / len(true_labels)), 6),
        "roc_auc": round(float(roc_auc_score(true_labels, scores)), 6),
        "average_precision": round(float(average_precision_score(true_labels, scores)), 6),
        "accuracy_at_0_5": round(float(accuracy_score(true_labels, predictions_at_05)), 6),
        "f1_at_0_5": round(float(f1_score(true_labels, predictions_at_05)), 6),
        "best_threshold": round(float(best_threshold), 6),
        "accuracy_best_threshold": round(float(best_accuracy), 6),
        "f1_best_threshold": round(float(best_f1), 6),
        "mean_positive_score": round(float(sum(positive_scores) / len(positive_scores)), 6),
        "mean_negative_score": round(float(sum(negative_scores) / len(negative_scores)), 6),
    }


def build_polarity_phrasebank_benchmark() -> pd.DataFrame:
    ds = load_dataset(
        PHRASEBANK_DATASET,
        PHRASEBANK_CONFIG,
        split="train",
        trust_remote_code=True,
    )
    label_names = ds.features["label"].names
    rows = []
    for row in ds:
        rows.append(
            {
                "text": row["sentence"],
                "label": label_names[row["label"]],
            }
        )
    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 20]
    return sample_balanced(df, "label", max_per_class=700)


def build_polarity_fiqa_benchmark() -> pd.DataFrame:
    fiqa = load_dataset(FIQA_DATASET)
    rows = []
    for split_name in ["train", "valid", "test"]:
        for row in fiqa[split_name]:
            rows.append(
                {
                    "text": row["sentence"],
                    "label": fiqa_score_to_sentiment(float(row["score"])),
                }
            )
    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 20].drop_duplicates(subset=["text"])
    return sample_balanced(df, "label", max_per_class=450)


def build_uncertainty_nicky_proxy_benchmark() -> pd.DataFrame:
    ds = load_dataset(NICKY_DATASET, split="train", trust_remote_code=True).to_pandas()
    ds = ds[ds["task_type"] == "topic_classification"].copy()
    ds["text"] = ds["user_prompt"].fillna("").astype(str).str.strip()
    ds = ds[ds["text"].str.len() > 20]

    high_uncertainty_topics = {
        "Macro",
        "Politics",
        "Fed | Central Banks",
        "Markets",
        "General News | Opinion",
        "Currencies",
        "Stock Movement",
    }
    low_uncertainty_topics = {
        "Earnings",
        "Dividend",
        "Financials",
        "Analyst Update",
        "Company | Product News",
        "Personnel Change",
    }

    df = ds[ds["answer"].isin(high_uncertainty_topics | low_uncertainty_topics)].copy()
    df["bucket"] = df["text"].apply(stable_bucket)
    df = df[df["bucket"] == NICKY_BENCHMARK_BUCKET].copy()
    df["label"] = df["answer"].isin(high_uncertainty_topics).astype(int)
    return sample_balanced(df[["text", "label"]], "label", max_per_class=800)


def build_uncertainty_fiqa_proxy_benchmark() -> pd.DataFrame:
    fiqa = load_dataset(FIQA_DATASET)
    high_uncertainty_aspects = {
        "Corporate/Risks",
        "Corporate/Regulatory",
        "Corporate/M&A/Proposed Merger",
        "Stock/Price Action/Volatility/Short Selling",
        "Stock/Price Action/Bearish/Bearish Behavior",
        "Stock/Price Action/Bullish/Bullish Behavior",
    }
    low_uncertainty_aspects = {
        "Corporate/Sales",
        "Corporate/Sales/Deal",
        "Corporate/Dividend Policy",
        "Corporate/Appointment",
        "Stock/Coverage/AnalystRatings/Upgrade",
    }

    rows = []
    for split_name in ["valid", "test"]:
        split_df = fiqa[split_name].to_pandas()
        split_df["text"] = split_df["sentence"].fillna("").astype(str).str.strip()
        split_df = split_df[split_df["text"].str.len() > 20]
        split_df = split_df[
            split_df["aspect"].isin(high_uncertainty_aspects | low_uncertainty_aspects)
        ].copy()
        split_df["label"] = split_df["aspect"].isin(high_uncertainty_aspects).astype(int)
        rows.append(split_df[["text", "label"]])

    df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["text"])
    return sample_balanced(df, "label")


def run_polarity_benchmarks():
    agent = PolarityAgent()
    benchmarks = {}

    phrasebank_df = build_polarity_phrasebank_benchmark()
    phrasebank_predictions = agent.predict_batch(phrasebank_df["text"].tolist(), batch_size=32)
    phrasebank_pred_labels = [result[2] for result in phrasebank_predictions]
    benchmarks["phrasebank_sentiment"] = evaluate_multiclass_labels(
        phrasebank_df["label"].tolist(),
        phrasebank_pred_labels,
        POLARITY_LABEL_ORDER,
    )

    fiqa_df = build_polarity_fiqa_benchmark()
    fiqa_predictions = agent.predict_batch(fiqa_df["text"].tolist(), batch_size=32)
    fiqa_pred_labels = [result[2] for result in fiqa_predictions]
    benchmarks["fiqa_sentiment"] = evaluate_multiclass_labels(
        fiqa_df["label"].tolist(),
        fiqa_pred_labels,
        POLARITY_LABEL_ORDER,
    )

    with open(POLARITY_BENCHMARK_FILE, "w", encoding="utf-8") as handle:
        json.dump(benchmarks, handle, indent=2)
    return benchmarks


def run_uncertainty_benchmarks():
    agent = UncertaintyAgent(model_path="./uncertainty_model")
    benchmarks = {}

    nicky_df = build_uncertainty_nicky_proxy_benchmark()
    nicky_scores = agent.predict_batch(nicky_df["text"].tolist(), batch_size=16)
    benchmarks["nicky_topic_uncertainty_proxy"] = evaluate_binary_scores(
        nicky_df["label"].tolist(),
        nicky_scores,
    )

    fiqa_df = build_uncertainty_fiqa_proxy_benchmark()
    fiqa_scores = agent.predict_batch(fiqa_df["text"].tolist(), batch_size=16)
    benchmarks["fiqa_aspect_uncertainty_proxy"] = evaluate_binary_scores(
        fiqa_df["label"].tolist(),
        fiqa_scores,
    )

    with open(UNCERTAINTY_BENCHMARK_FILE, "w", encoding="utf-8") as handle:
        json.dump(benchmarks, handle, indent=2)
    return benchmarks


def main():
    ensure_output_dir()
    print("Running polarity public benchmarks...")
    polarity_benchmarks = run_polarity_benchmarks()
    print(json.dumps(polarity_benchmarks, indent=2))

    print("Running uncertainty public proxy benchmarks...")
    uncertainty_benchmarks = run_uncertainty_benchmarks()
    print(json.dumps(uncertainty_benchmarks, indent=2))


if __name__ == "__main__":
    main()
