"""
Fundamental Strength Agent - Training and inference
==================================================

This agent estimates how strongly a financial text discusses business
fundamentals: earnings, revenues, margins, free cash flow, dividends,
guidance, debt profile, capital discipline, backlog, and similar signals.

Output:
    score in [0, 1]
    0.0 = little fundamental signal
    1.0 = strong fundamental signal

Training strategy:
    - local project articles weak-labeled with heuristics
    - synthetic examples for stability
    - larger external financial corpora from public datasets

Public benchmarks:
    - finance topic benchmark
    - finance aspect benchmark
"""

import argparse
import json
import logging
import math
import os
import re
import sqlite3
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_OUTPUT_DIR = "./fundamental_strength_model"
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "model.joblib")
METRICS_FILE = os.path.join(MODEL_OUTPUT_DIR, "metrics.json")
BENCHMARK_FILE = os.path.join(MODEL_OUTPUT_DIR, "benchmarks.json")
CONFIG_FILE = os.path.join(MODEL_OUTPUT_DIR, "config.json")

PHRASEBANK_DATASET = "financial_phrasebank"
PHRASEBANK_CONFIG = "sentences_allagree"
NICKY_DATASET = "NickyNicky/Finance_sentiment_and_topic_classification_En"
FIQA_DATASET = "TheFinAI/fiqa-sentiment-classification"

logger = logging.getLogger("FundamentalStrengthAgent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)


def load_fundamental_lexicon() -> set:
    return {
        "backlog", "beat", "beats", "bookings", "buyback", "buybacks", "cash",
        "cashflow", "cashflows", "contract", "contracts", "demand", "debt",
        "deleveraging", "dividend", "dividends", "earnings", "ebit", "ebitda",
        "efficiency", "expansion", "fcf", "free", "growth", "guidance",
        "improvement", "improvements", "income", "leverage", "margin", "margins",
        "order", "orders", "pipeline", "profit", "profitability", "profitable",
        "raise", "raised", "reaffirmed", "repurchase", "revenue", "revenues",
        "sales", "shareholder", "strong", "subscription", "subscriptions",
        "visibility", "wins", "yield",
    }


def _clip_score(score: float) -> float:
    return round(max(0.0, min(float(score), 1.0)), 4)


def compute_fundamental_strength_score(text: str, lexicon: set) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", (text or "").lower())
    if not words:
        return 0.0

    text_lower = text.lower()
    n_words = len(words)

    positive_count = sum(1 for word in words if word in lexicon)
    lexicon_ratio = positive_count / n_words
    lexicon_score = min(lexicon_ratio / 0.025, 1.0)

    strong_patterns = [
        r"\braised guidance\b",
        r"\breaffirmed guidance\b",
        r"\bbeat(?:s)? expectations\b",
        r"\brecord revenue\b",
        r"\brecord earnings\b",
        r"\bmargin expansion\b",
        r"\bfree cash flow\b",
        r"\bstrong demand\b",
        r"\bdouble-digit growth\b",
        r"\border backlog\b",
        r"\bshare repurchase\b",
        r"\bdividend increase\b",
        r"\bdebt reduction\b",
        r"\bprofitable growth\b",
        r"\boperating leverage\b",
    ]
    strong_hits = sum(1 for pattern in strong_patterns if re.search(pattern, text_lower))
    strong_signal = min(strong_hits / 4.0, 1.0)

    numeric_positive_patterns = [
        r"\bup \d+(?:\.\d+)?%\b",
        r"\bgrew \d+(?:\.\d+)?%\b",
        r"\bgrowth of \d+(?:\.\d+)?%\b",
        r"\bmargin(?:s)? .*? expanded\b",
        r"\bcash flow .*? increased\b",
        r"\brevenue .*? rose\b",
        r"\bearnings .*? rose\b",
    ]
    numeric_hits = sum(
        1 for pattern in numeric_positive_patterns if re.search(pattern, text_lower)
    )
    numeric_signal = min(numeric_hits / 3.0, 1.0)

    weakness_patterns = [
        r"\bmissed expectations\b",
        r"\bcut guidance\b",
        r"\bmargin pressure\b",
        r"\bcash burn\b",
        r"\bweak demand\b",
        r"\bdeclining sales\b",
        r"\bfalling revenue\b",
        r"\bprofit warning\b",
        r"\bnegative free cash flow\b",
        r"\bleverage concerns\b",
        r"\bdebt burden\b",
    ]
    weakness_hits = sum(1 for pattern in weakness_patterns if re.search(pattern, text_lower))
    weakness_penalty = min(weakness_hits / 4.0, 0.8)

    raw_score = (
        0.50 * lexicon_score
        + 0.30 * strong_signal
        + 0.20 * numeric_signal
        - 0.35 * weakness_penalty
    )
    raw_score = max(0.0, min(raw_score, 1.0))
    return _clip_score(math.sqrt(raw_score))


def _normalize_text_frame(df: pd.DataFrame, text_column: str, source_name: str) -> pd.DataFrame:
    normalized = df.copy()
    normalized["text"] = normalized[text_column].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["text"].str.len() > 20]
    normalized = normalized[["text"]].copy()
    normalized["source"] = source_name
    return normalized


def _sample_per_source(df: pd.DataFrame, max_samples_per_source: int = None) -> pd.DataFrame:
    if not max_samples_per_source:
        return df.reset_index(drop=True)

    sampled_frames = []
    for _, source_df in df.groupby("source"):
        if len(source_df) > max_samples_per_source:
            sampled_frames.append(
                source_df.sample(n=max_samples_per_source, random_state=42)
            )
        else:
            sampled_frames.append(source_df)
    return pd.concat(sampled_frames, ignore_index=True)


def generate_weak_labels(db_path: str, lexicon: set, max_samples: int = None) -> pd.DataFrame:
    logger.info(f"Loading articles from {db_path}...")

    df = pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        query = """
            SELECT title, content
            FROM articles
            WHERE content IS NOT NULL AND LENGTH(content) > 100
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as exc:
        logger.warning(f"Could not read database ({exc}). Falling back to synthetic data.")

    if df.empty:
        synthetic_df = _generate_synthetic_data()
        synthetic_df["text"] = synthetic_df["content"]
        synthetic_df["score"] = synthetic_df["text"].apply(
            lambda text: compute_fundamental_strength_score(text, lexicon)
        )
        synthetic_df["source"] = "synthetic_seed"
        return synthetic_df[["text", "score", "source"]]

    df["text"] = df["content"].fillna(df["title"]).fillna("")
    df["score"] = df["text"].apply(
        lambda text: compute_fundamental_strength_score(text, lexicon)
    )
    df["source"] = "local_articles"
    df = df[["text", "score", "source"]]

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    logger.info(f"{len(df)} real articles weak-labeled.")
    return df


def load_external_weak_training_corpus(max_samples_per_source: int = None) -> pd.DataFrame:
    frames = []

    phrasebank = load_dataset(
        PHRASEBANK_DATASET,
        PHRASEBANK_CONFIG,
        split="train",
        trust_remote_code=True,
    ).to_pandas()
    frames.append(_normalize_text_frame(phrasebank, "sentence", "financial_phrasebank"))

    nicky = load_dataset(
        NICKY_DATASET,
        split="train",
        trust_remote_code=True,
    ).to_pandas()
    nicky = nicky[nicky["task_type"] == "sentiment_analysis"].copy()
    frames.append(_normalize_text_frame(nicky, "user_prompt", "nicky_sentiment"))

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = _sample_per_source(df, max_samples_per_source=max_samples_per_source)
    return df


def build_public_benchmarks() -> Dict[str, pd.DataFrame]:
    benchmarks = {}

    nicky = load_dataset(
        NICKY_DATASET,
        split="train",
        trust_remote_code=True,
    ).to_pandas()
    nicky = nicky[nicky["task_type"] == "topic_classification"].copy()
    nicky["text"] = nicky["user_prompt"].fillna("").astype(str).str.strip()
    nicky = nicky[nicky["text"].str.len() > 20]

    positive_topics = {
        "Earnings",
        "Financials",
        "Dividend",
        "Treasuries | Corporate Debt",
    }
    negative_topics = {
        "Legal | Regulation",
        "Politics",
        "Macro",
        "Fed | Central Banks",
        "Currencies",
        "Markets",
        "Personnel Change",
        "Stock Movement",
    }

    topic_benchmark = nicky[nicky["answer"].isin(positive_topics | negative_topics)].copy()
    topic_benchmark["label"] = topic_benchmark["answer"].isin(positive_topics).astype(int)
    min_count = int(topic_benchmark["label"].value_counts().min())
    topic_benchmark = topic_benchmark.groupby("label", group_keys=False).sample(
        n=min(min_count, 1200),
        random_state=42,
    )
    benchmarks["nicky_topic_fundamentals"] = topic_benchmark[["text", "label"]]

    fiqa = load_dataset(FIQA_DATASET)
    fiqa_rows = []
    positive_aspects = {
        "Corporate/Sales",
        "Corporate/Sales/Deal",
        "Corporate/Dividend Policy",
    }
    negative_aspects = {
        "Corporate/Appointment",
        "Corporate/Regulatory",
        "Stock/Technical Analysis",
        "Stock/Price Action",
        "Stock/Coverage/AnalystRatings/Upgrade",
        "Stock/Signal/Buy Signal",
    }
    for split_name in ["train", "valid", "test"]:
        split_df = fiqa[split_name].to_pandas()
        split_df["text"] = split_df["sentence"].fillna("").astype(str).str.strip()
        split_df = split_df[split_df["text"].str.len() > 20]
        split_df = split_df[split_df["aspect"].isin(positive_aspects | negative_aspects)].copy()
        split_df["label"] = split_df["aspect"].isin(positive_aspects).astype(int)
        fiqa_rows.append(split_df[["text", "label"]])

    fiqa_benchmark = pd.concat(fiqa_rows, ignore_index=True).drop_duplicates(subset=["text"])
    min_count = int(fiqa_benchmark["label"].value_counts().min())
    fiqa_benchmark = fiqa_benchmark.groupby("label", group_keys=False).sample(
        n=min_count,
        random_state=42,
    )
    benchmarks["fiqa_aspect_fundamentals"] = fiqa_benchmark[["text", "label"]]
    return benchmarks


def evaluate_binary_benchmark(
    benchmark_df: pd.DataFrame,
    pipeline: Pipeline,
) -> Dict[str, float]:
    texts = benchmark_df["text"].tolist()
    labels = benchmark_df["label"].astype(int).tolist()
    scores = [_clip_score(score) for score in pipeline.predict(texts)]
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = 0.0
    for threshold_idx in range(5, 96):
        threshold = threshold_idx / 100.0
        threshold_predictions = [1 if score >= threshold else 0 for score in scores]
        threshold_f1 = f1_score(labels, threshold_predictions)
        if threshold_f1 > best_f1:
            best_f1 = threshold_f1
            best_threshold = threshold
            best_accuracy = accuracy_score(labels, threshold_predictions)

    positive_scores = [score for score, label in zip(scores, labels) if label == 1]
    negative_scores = [score for score, label in zip(scores, labels) if label == 0]

    return {
        "dataset_size": int(len(labels)),
        "positive_share": round(float(sum(labels) / len(labels)), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "average_precision": round(float(average_precision_score(labels, scores)), 6),
        "accuracy_at_0_5": round(float(accuracy_score(labels, predictions)), 6),
        "f1_at_0_5": round(float(f1_score(labels, predictions)), 6),
        "best_threshold": round(float(best_threshold), 6),
        "accuracy_best_threshold": round(float(best_accuracy), 6),
        "f1_best_threshold": round(float(best_f1), 6),
        "mean_positive_score": round(float(sum(positive_scores) / len(positive_scores)), 6),
        "mean_negative_score": round(float(sum(negative_scores) / len(negative_scores)), 6),
    }


def _generate_synthetic_data() -> pd.DataFrame:
    high_strength = [
        "The company reported record revenue growth, margin expansion, and strong free cash flow while raising full-year guidance.",
        "Management reaffirmed demand strength, announced a share buyback, and delivered double-digit earnings growth with lower leverage.",
        "Backlog reached a record high, operating margins improved, and the board approved a dividend increase after a strong quarter.",
        "The firm beat expectations on revenue and EBITDA, expanded gross margin, and reduced debt thanks to robust cash generation.",
        "Customer bookings accelerated, free cash flow turned strongly positive, and management raised profitability targets for next year.",
        "Recurring revenue grew sharply, churn remained low, and the company delivered profitable growth across all key segments.",
        "Orders surged, backlog improved, and operating leverage helped lift earnings well above consensus forecasts.",
        "The company posted strong sales growth, record margins, and clear guidance visibility supported by large signed contracts.",
    ]

    medium_strength = [
        "Revenue came in line with expectations and margins were stable, while management kept guidance unchanged.",
        "The quarter was solid overall, with modest growth in sales and steady cash generation despite mixed segment trends.",
        "Management highlighted resilient demand and stable profitability, but stopped short of raising full-year targets.",
        "The business delivered acceptable earnings with some efficiency gains, though growth remained moderate.",
        "Orders were healthy and the balance sheet remains stable, but margin improvement was limited.",
        "Analysts described the update as balanced, with decent revenue visibility but no major upside catalyst.",
        "Cash flow improved slightly and debt was manageable, though management remains cautious on near-term acceleration.",
        "The company met expectations on earnings and maintained a disciplined capital allocation policy.",
    ]

    low_strength = [
        "The company missed expectations, cut guidance, and warned about margin pressure and weaker customer demand.",
        "Revenue declined, free cash flow remained negative, and management flagged ongoing leverage concerns.",
        "The quarter showed deteriorating profitability, softer bookings, and no visibility on a near-term recovery.",
        "Management reported weak sales trends, lower margins, and a heavier debt burden after disappointing results.",
        "The firm posted negative free cash flow, shrinking demand, and a profit warning ahead of the next quarter.",
        "Operating performance weakened sharply as revenue fell and management lowered forecasts for the rest of the year.",
        "The company cited cash burn, declining orders, and weak execution after another quarter below expectations.",
        "Sales contracted and margin pressure intensified, forcing management to cut targets and slow investments.",
    ]

    rows = [{"content": text} for text in high_strength + medium_strength + low_strength]
    return pd.DataFrame(rows)


def train_model(
    db_path: str = "news_database.db",
    output_dir: str = MODEL_OUTPUT_DIR,
    alpha: float = 1.0,
    max_samples: int = None,
):
    os.makedirs(output_dir, exist_ok=True)
    lexicon = load_fundamental_lexicon()

    local_df = generate_weak_labels(db_path, lexicon, max_samples=max_samples)

    synthetic_df = _generate_synthetic_data()
    synthetic_df["text"] = synthetic_df["content"]
    synthetic_df["score"] = synthetic_df["text"].apply(
        lambda text: compute_fundamental_strength_score(text, lexicon)
    )
    synthetic_df["source"] = "synthetic_examples"
    synthetic_df = synthetic_df[["text", "score", "source"]]

    logger.info("Loading larger external financial corpora...")
    external_df = load_external_weak_training_corpus()
    external_df["score"] = external_df["text"].apply(
        lambda text: compute_fundamental_strength_score(text, lexicon)
    )

    df = pd.concat([local_df, synthetic_df, external_df], ignore_index=True)
    df = df.dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    logger.info(f"Training dataset size: {len(df)}")
    logger.info(f"  mean score = {df['score'].mean():.4f}")
    logger.info(f"  min score  = {df['score'].min():.4f}")
    logger.info(f"  max score  = {df['score'].max():.4f}")

    x_train, x_eval, y_train, y_eval = train_test_split(
        df["text"],
        df["score"],
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=12000,
                    sublinear_tf=True,
                ),
            ),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )

    pipeline.fit(x_train, y_train)
    eval_predictions = [_clip_score(score) for score in pipeline.predict(x_eval)]

    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(len(x_train)),
        "eval_size": int(len(x_eval)),
        "training_sources": df["source"].value_counts().to_dict(),
        "label_mean": round(float(df["score"].mean()), 6),
        "label_min": round(float(df["score"].min()), 6),
        "label_max": round(float(df["score"].max()), 6),
        "mae": round(float(mean_absolute_error(y_eval, eval_predictions)), 6),
        "rmse": round(float(math.sqrt(mean_squared_error(y_eval, eval_predictions))), 6),
        "r2": round(float(r2_score(y_eval, eval_predictions)), 6),
        "model_type": "tfidf_ridge_regression",
        "feature": "fundamental_strength",
    }

    logger.info("Eval metrics:")
    logger.info(f"  MAE  = {metrics['mae']}")
    logger.info(f"  RMSE = {metrics['rmse']}")
    logger.info(f"  R2   = {metrics['r2']}")

    pipeline.fit(df["text"], df["score"])

    logger.info("Running public benchmark tests...")
    benchmark_results = {}
    for name, benchmark_df in build_public_benchmarks().items():
        benchmark_results[name] = evaluate_binary_benchmark(benchmark_df, pipeline)
        logger.info(
            f"  {name}: auc={benchmark_results[name]['roc_auc']}, "
            f"f1@0.5={benchmark_results[name]['f1_at_0_5']}"
        )

    metrics["benchmarks"] = benchmark_results

    dump(pipeline, MODEL_FILE)

    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with open(BENCHMARK_FILE, "w", encoding="utf-8") as handle:
        json.dump(benchmark_results, handle, indent=2)

    with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "lexicon_size": len(lexicon),
                "model_file": MODEL_FILE,
                "benchmark_file": BENCHMARK_FILE,
                "external_sources": [
                    PHRASEBANK_DATASET,
                    NICKY_DATASET,
                ],
            },
            handle,
            indent=2,
        )

    logger.info(f"Saved model to {MODEL_FILE}")
    logger.info(f"Saved metrics to {METRICS_FILE}")
    logger.info(f"Saved benchmark results to {BENCHMARK_FILE}")
    return pipeline, metrics


class FundamentalStrengthAgent:
    def __init__(self, model_path: str = MODEL_OUTPUT_DIR, fallback_to_heuristic: bool = False):
        self.model_file = os.path.join(model_path, "model.joblib")
        self.fallback_to_heuristic = fallback_to_heuristic
        self.lexicon = load_fundamental_lexicon()
        self.pipeline = None

        if os.path.exists(self.model_file):
            logger.info(f"Loading fundamental strength model from {self.model_file}")
            self.pipeline = load(self.model_file)
        elif not fallback_to_heuristic:
            raise FileNotFoundError(
                f"Fundamental strength model not found at {self.model_file}. Run training first."
            )
        else:
            logger.warning(
                "Fundamental strength model not found. Falling back to heuristic weak-label score."
            )

    def predict(self, text: str) -> float:
        if not text:
            return 0.0
        if self.pipeline is None:
            return compute_fundamental_strength_score(text, self.lexicon)
        return _clip_score(self.pipeline.predict([text])[0])

    def predict_batch(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
        if self.pipeline is None:
            return [compute_fundamental_strength_score(text, self.lexicon) for text in texts]
        predictions = self.pipeline.predict(texts)
        return [_clip_score(score) for score in predictions]


def main():
    parser = argparse.ArgumentParser(description="Fundamental strength agent")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="train to build the model, predict to score one text",
    )
    parser.add_argument("--db", default="news_database.db", help="SQLite database path")
    parser.add_argument("--text", default=None, help="Text to score in predict mode")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional training cap")
    args = parser.parse_args()

    if args.mode == "train":
        _, metrics = train_model(db_path=args.db, max_samples=args.max_samples)
        print(json.dumps(metrics, indent=2))
        return

    if not args.text:
        raise ValueError("--text is required in predict mode")

    agent = FundamentalStrengthAgent(fallback_to_heuristic=True)
    score = agent.predict(args.text)
    print(f"fundamental_strength = {score:.4f}")


if __name__ == "__main__":
    main()
