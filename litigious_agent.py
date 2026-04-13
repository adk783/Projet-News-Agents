"""
Litigious Agent - Training and inference
=======================================

This agent adds a complementary legal / regulatory risk score to the
existing sentiment vector.

Model choice:
    TF-IDF + Ridge regression (scikit-learn)

Why this model:
    - different from FinBERT
    - lightweight and fast to train
    - easy to evaluate with MAE / RMSE / R2
    - simple to integrate into the current architecture

Training data:
    - local project articles weak-labeled with heuristics
    - synthetic legal-risk examples
    - larger public financial corpora
    - a legal-domain corpus to give the model more legal vocabulary

Benchmarks:
    - finance topic benchmark
    - finance aspect benchmark
    - legal-domain stress benchmark
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


MODEL_OUTPUT_DIR = "./litigious_model"
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "model.joblib")
METRICS_FILE = os.path.join(MODEL_OUTPUT_DIR, "metrics.json")
BENCHMARK_FILE = os.path.join(MODEL_OUTPUT_DIR, "benchmarks.json")
CONFIG_FILE = os.path.join(MODEL_OUTPUT_DIR, "config.json")
PROMPT_PREFIX = "Assess the legal and regulatory risk in the following financial text: "

PHRASEBANK_DATASET = "financial_phrasebank"
PHRASEBANK_CONFIG = "sentences_allagree"
NICKY_DATASET = "NickyNicky/Finance_sentiment_and_topic_classification_En"
FIQA_DATASET = "TheFinAI/fiqa-sentiment-classification"
LEXGLUE_DATASET = "lex_glue"
LEXGLUE_CONFIG = "ledgar"

logger = logging.getLogger("LitigiousAgent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)


def load_litigious_lexicon() -> set:
    return {
        "action", "actions", "allegation", "allegations", "appeal", "appeals",
        "arbitration", "attorney", "attorneys", "audit", "breach", "charges",
        "claim", "claims", "class", "complaint", "complaints", "compliance",
        "compliant", "consent", "court", "courts", "covenant", "covenants",
        "criminal", "damages", "default", "defaults", "defendant",
        "defendants", "dispute", "disputes", "doj", "enforcement", "fined",
        "fine", "fines", "fraud", "fraudulent", "ftc", "governmental",
        "indictment", "injunction", "inquiry", "investigation",
        "investigations", "judicial", "judge", "jury", "lawsuit",
        "lawsuits", "legal", "legislation", "liability", "liabilities",
        "litigate", "litigation", "noncompliance", "penalty", "penalties",
        "plaintiff", "plaintiffs", "probe", "proceeding", "proceedings",
        "prosecution", "regulation", "regulations", "regulatory", "remedy",
        "restriction", "restrictions", "sanction", "sanctions", "sec",
        "settlement", "settlements", "subpoena", "sued", "sues", "suit",
        "tribunal", "trial", "violation", "violations", "waiver", "waivers",
    }


def _clip_score(score: float) -> float:
    return round(max(0.0, min(float(score), 1.0)), 4)


def compute_litigious_score(text: str, lexicon: set) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", (text or "").lower())
    if not words:
        return 0.0

    text_lower = text.lower()
    n_words = len(words)

    litigious_count = sum(1 for word in words if word in lexicon)
    lexicon_ratio = litigious_count / n_words
    lexicon_score = min(lexicon_ratio / 0.02, 1.0)

    legal_patterns = [
        r"\blawsuit\b",
        r"\blawsuits\b",
        r"\bclass action\b",
        r"\bcourt\b",
        r"\bplaintiff\b",
        r"\bdefendant\b",
        r"\bsettlement\b",
        r"\btrial\b",
        r"\bsubpoena\b",
        r"\barbitration\b",
        r"\binjunction\b",
        r"\bcomplaint filed\b",
        r"\bdamages\b",
        r"\bappeal\b",
    ]
    legal_hits = sum(1 for pattern in legal_patterns if re.search(pattern, text_lower))
    legal_score = min(legal_hits / 4.0, 1.0)

    regulatory_patterns = [
        r"\bsec\b",
        r"\bdoj\b",
        r"\bftc\b",
        r"\bantitrust\b",
        r"\binvestigation\b",
        r"\bprobe\b",
        r"\benforcement\b",
        r"\bregulator\b",
        r"\bregulatory review\b",
        r"\bpenalt(?:y|ies)\b",
        r"\bsanction(?:s)?\b",
        r"\bcompliance failure\b",
        r"\bconsent decree\b",
        r"\bwhistleblower\b",
    ]
    regulatory_hits = sum(
        1 for pattern in regulatory_patterns if re.search(pattern, text_lower)
    )
    regulatory_score = min(regulatory_hits / 4.0, 1.0)

    constraint_patterns = [
        r"\bcovenant(?:s)?\b",
        r"\bcovenant breach\b",
        r"\bliquidity pressure\b",
        r"\bliquidity shortfall\b",
        r"\brestricted cash\b",
        r"\bwaiver request\b",
        r"\bdefault notice\b",
        r"\brefinancing risk\b",
        r"\bchapter 11\b",
        r"\brestructuring support\b",
    ]
    constraint_hits = sum(
        1 for pattern in constraint_patterns if re.search(pattern, text_lower)
    )
    constraint_score = min(constraint_hits / 3.0, 1.0)

    raw_score = (
        0.45 * lexicon_score
        + 0.25 * legal_score
        + 0.20 * regulatory_score
        + 0.10 * constraint_score
    )
    raw_score = max(0.0, min(raw_score, 1.0))
    return _clip_score(math.sqrt(raw_score))


def _normalize_text_frame(
    df: pd.DataFrame,
    text_column: str,
    source_name: str,
    extra_columns: List[str] = None,
) -> pd.DataFrame:
    normalized = df.copy()
    normalized["text"] = normalized[text_column].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["text"].str.len() > 20]
    keep_columns = ["text"]
    if extra_columns:
        keep_columns.extend(extra_columns)
    normalized = normalized[keep_columns].copy()
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
        logger.warning("No real articles found. Using synthetic legal-risk samples only.")
        synthetic_df = _generate_synthetic_data()
        synthetic_df["text"] = synthetic_df["content"]
        synthetic_df["score"] = synthetic_df["text"].apply(
            lambda text: compute_litigious_score(text, lexicon)
        )
        synthetic_df["source"] = "synthetic_seed"
        return synthetic_df[["text", "score", "source"]]

    df["text"] = df["content"].fillna(df["title"]).fillna("")
    df["score"] = df["text"].apply(lambda text: compute_litigious_score(text, lexicon))
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

    nicky_topics = load_dataset(
        NICKY_DATASET,
        split="train",
        trust_remote_code=True,
    ).to_pandas()
    nicky_topics = nicky_topics[nicky_topics["task_type"] == "topic_classification"].copy()
    frames.append(
        _normalize_text_frame(
            nicky_topics,
            "user_prompt",
            "nicky_topic",
            extra_columns=["answer"],
        )
    )

    fiqa = load_dataset(FIQA_DATASET)
    fiqa_train = fiqa["train"].to_pandas()
    frames.append(
        _normalize_text_frame(
            fiqa_train,
            "sentence",
            "fiqa_train_aspect",
            extra_columns=["aspect"],
        )
    )

    ledgar_train = load_dataset(
        LEXGLUE_DATASET,
        LEXGLUE_CONFIG,
        split="train",
        trust_remote_code=True,
    ).to_pandas()
    ledgar_train = _normalize_text_frame(ledgar_train, "text", "lexglue_ledgar_train")
    ledgar_train = _sample_per_source(ledgar_train, max_samples_per_source=6000)
    frames.append(ledgar_train)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = _sample_per_source(df, max_samples_per_source=max_samples_per_source)
    return df


def build_public_benchmarks() -> Dict[str, pd.DataFrame]:
    benchmarks = {}

    fiqa = load_dataset(FIQA_DATASET)
    fiqa_rows = []
    positive_aspects = {
        "Corporate/Regulatory",
        "Corporate/Risks",
        "Corporate/Risks/Product Recall",
    }
    negative_aspects = {
        "Corporate/Sales",
        "Corporate/Sales/Deal",
        "Corporate/Dividend Policy",
        "Corporate/Appointment",
        "Stock/Technical Analysis",
        "Stock/Price Action",
    }
    for split_name in ["valid", "test"]:
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
    benchmarks["fiqa_aspect_legal_risk"] = fiqa_benchmark[["text", "label"]]

    ledgar_valid = load_dataset(
        LEXGLUE_DATASET,
        LEXGLUE_CONFIG,
        split="validation",
        trust_remote_code=True,
    ).to_pandas()
    ledgar_valid = _normalize_text_frame(ledgar_valid, "text", "lexglue_ledgar_validation")
    ledgar_valid["label"] = 1

    phrasebank = load_dataset(
        PHRASEBANK_DATASET,
        PHRASEBANK_CONFIG,
        split="train",
        trust_remote_code=True,
    ).to_pandas()
    phrasebank = _normalize_text_frame(phrasebank, "sentence", "financial_phrasebank_negative")
    phrasebank["label"] = 0
    phrasebank = phrasebank.sample(n=min(len(phrasebank), len(ledgar_valid), 1200), random_state=42)
    ledgar_valid = ledgar_valid.sample(n=len(phrasebank), random_state=42)
    benchmarks["legal_domain_stress"] = pd.concat(
        [ledgar_valid[["text", "label"]], phrasebank[["text", "label"]]],
        ignore_index=True,
    )
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
    for threshold_idx in range(1, 51):
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
    high_risk = [
        "The company faces a class action lawsuit alleging misleading disclosures about revenue quality and covenant compliance.",
        "Regulators opened an antitrust investigation while the DOJ requested documents related to the merger review.",
        "Management disclosed a subpoena from the SEC and warned about possible penalties, sanctions, and civil claims.",
        "The borrower is negotiating covenant waivers after a default notice and a growing liquidity shortfall.",
        "Shareholders filed a complaint seeking damages after a product safety recall triggered multiple legal proceedings.",
        "The group entered Chapter 11 discussions after creditors challenged its debt restructuring and collateral package.",
        "The firm announced a settlement reserve after patent litigation and arbitration proceedings intensified across markets.",
        "A whistleblower report triggered a regulatory probe into accounting practices, compliance failures, and governance controls.",
    ]

    medium_risk = [
        "The company said regulatory review could delay the transaction, although management expects no material remedy.",
        "Executives mentioned ongoing discussions with regulators and external counsel regarding data privacy obligations.",
        "The issuer is monitoring covenant headroom and may request a waiver if cash generation weakens further.",
        "The business remains exposed to legal disputes with suppliers, though no formal court action has been launched.",
        "Management disclosed a preliminary inquiry into marketing practices and said cooperation with authorities is ongoing.",
        "The board noted moderate litigation exposure related to employment disputes and vendor claims.",
        "The company is reassessing compliance controls after a minor penalty from a foreign regulator.",
        "Analysts flagged refinancing risk because tighter debt restrictions may limit flexibility next quarter.",
    ]

    low_risk = [
        "The company reported strong revenue growth, reaffirmed guidance, and highlighted stable customer demand across segments.",
        "Management raised full-year forecasts after margin expansion and better-than-expected operating cash flow.",
        "The board approved a dividend increase following record quarterly earnings and a healthy balance sheet.",
        "The firm completed a factory expansion on schedule and expects the new capacity to support growth.",
        "Analysts praised the latest results, citing robust execution and low operational uncertainty.",
        "The company generated solid free cash flow and reduced leverage while expanding into new markets.",
        "Leadership reiterated that the merger integration remains on track with no major execution issues.",
        "The issuer announced a new product launch backed by signed customer contracts and visible demand.",
    ]

    rows = [{"content": text} for text in high_risk + medium_risk + low_risk]
    return pd.DataFrame(rows)


def train_model(
    db_path: str = "news_database.db",
    output_dir: str = MODEL_OUTPUT_DIR,
    alpha: float = 1.0,
    max_samples: int = None,
):
    os.makedirs(output_dir, exist_ok=True)
    lexicon = load_litigious_lexicon()

    local_df = generate_weak_labels(db_path, lexicon, max_samples=max_samples)

    synthetic_df = _generate_synthetic_data()
    synthetic_df["text"] = synthetic_df["content"]
    synthetic_df["score"] = synthetic_df["text"].apply(
        lambda text: compute_litigious_score(text, lexicon)
    )
    synthetic_df["source"] = "synthetic_examples"
    synthetic_df = synthetic_df[["text", "score", "source"]]

    logger.info("Loading larger external corpora...")
    external_df = load_external_weak_training_corpus()
    external_df["score"] = external_df["text"].apply(
        lambda text: compute_litigious_score(text, lexicon)
    )
    nicky_topic_priors = {
        "Legal | Regulation": 0.92,
        "Treasuries | Corporate Debt": 0.34,
        "Politics": 0.22,
        "M&A | Investments": 0.16,
        "Macro": 0.10,
        "Fed | Central Banks": 0.08,
    }
    fiqa_aspect_priors = {
        "Corporate/Regulatory": 0.92,
        "Corporate/Risks": 0.74,
        "Corporate/Risks/Product Recall": 0.82,
        "Corporate/Sales": 0.05,
        "Corporate/Sales/Deal": 0.08,
        "Corporate/Dividend Policy": 0.04,
        "Corporate/Appointment": 0.05,
    }
    external_df.loc[
        external_df["source"] == "nicky_topic",
        "score",
    ] = external_df.loc[
        external_df["source"] == "nicky_topic",
    ].apply(
        lambda row: max(
            row["score"],
            nicky_topic_priors.get(str(row.get("answer", "")).strip(), 0.03),
        ),
        axis=1,
    )
    external_df.loc[
        external_df["source"] == "fiqa_train_aspect",
        "score",
    ] = external_df.loc[
        external_df["source"] == "fiqa_train_aspect",
    ].apply(
        lambda row: max(
            row["score"],
            fiqa_aspect_priors.get(str(row.get("aspect", "")).strip(), 0.03),
        ),
        axis=1,
    )
    external_df.loc[
        external_df["source"] == "lexglue_ledgar_train",
        "score",
    ] = external_df.loc[
        external_df["source"] == "lexglue_ledgar_train",
        "score",
    ].apply(lambda score: max(score, 0.78))

    df = pd.concat([local_df, synthetic_df, external_df], ignore_index=True)
    df = df.dropna(subset=["text", "score"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

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
        "feature": "litigious",
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
                "prompt_prefix": PROMPT_PREFIX,
                "lexicon_size": len(lexicon),
                "model_file": MODEL_FILE,
                "benchmark_file": BENCHMARK_FILE,
                "external_sources": [
                    PHRASEBANK_DATASET,
                    NICKY_DATASET,
                    f"{LEXGLUE_DATASET}/{LEXGLUE_CONFIG}",
                ],
            },
            handle,
            indent=2,
        )

    logger.info(f"Saved model to {MODEL_FILE}")
    logger.info(f"Saved metrics to {METRICS_FILE}")
    logger.info(f"Saved benchmark results to {BENCHMARK_FILE}")
    return pipeline, metrics


class LitigiousAgent:
    def __init__(self, model_path: str = MODEL_OUTPUT_DIR, fallback_to_heuristic: bool = False):
        self.model_dir = model_path
        self.model_file = os.path.join(model_path, "model.joblib")
        self.fallback_to_heuristic = fallback_to_heuristic
        self.lexicon = load_litigious_lexicon()
        self.pipeline = None

        if os.path.exists(self.model_file):
            logger.info(f"Loading litigious model from {self.model_file}")
            self.pipeline = load(self.model_file)
        elif not fallback_to_heuristic:
            raise FileNotFoundError(
                f"Litigious model not found at {self.model_file}. Run training first."
            )
        else:
            logger.warning(
                "Litigious model not found. Falling back to heuristic weak-label score."
            )

    def predict(self, text: str) -> float:
        if not text:
            return 0.0

        if self.pipeline is None:
            return compute_litigious_score(text, self.lexicon)

        return _clip_score(self.pipeline.predict([text])[0])

    def predict_batch(self, texts: List[str]) -> List[float]:
        if not texts:
            return []

        if self.pipeline is None:
            return [compute_litigious_score(text, self.lexicon) for text in texts]

        predictions = self.pipeline.predict(texts)
        return [_clip_score(score) for score in predictions]


def main():
    parser = argparse.ArgumentParser(description="Litigious risk agent")
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

    agent = LitigiousAgent(fallback_to_heuristic=True)
    score = agent.predict(args.text)
    print(f"litigious_score = {score:.4f}")


if __name__ == "__main__":
    main()
