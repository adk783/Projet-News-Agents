"""
Market impact processing pipeline
=================================

Modes disponibles :

1. supervised_external
   - entraine la classification finale sur un benchmark public Bearish / Neutral / Bullish
   - utilise le clustering KMeans comme signal auxiliaire
   - applique ensuite le modele sur les articles sourcés localement

2. pseudo_cluster
   - ancien mode PoC qui entraine le classifieur sur des pseudo-labels KMeans

Le mode par defaut est supervised_external.
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fundamental_strength_agent import FundamentalStrengthAgent
from litigious_agent import LitigiousAgent
from polarity_agent import PolarityAgent
from uncertainty_agent import (
    UncertaintyAgent,
    compute_uncertainty_score,
    download_lm_uncertainty_lexicon,
)


DB_PATH = "news_database.db"
MODEL_DIR = "./market_impact_model"
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "classifier.joblib")
CLUSTER_FILE = os.path.join(MODEL_DIR, "kmeans.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
FEATURES_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
RAW_SUPERVISED_DATASET_FILE = os.path.join(MODEL_DIR, "market_impact_supervised_dataset.csv")
SUPERVISED_FEATURES_FILE = os.path.join(MODEL_DIR, "market_impact_supervised_features.csv")

MARKET_IMPACT_DATASET = "benstaf/FNSPID-nasdaq-100-post2019-1newsperrow"
MARKET_IMPACT_DATASET_URL = "https://huggingface.co/datasets/benstaf/FNSPID-nasdaq-100-post2019-1newsperrow"
FNSPID_PRICE_DATASET = "beachside1234/FNSPID"
FNSPID_PRICE_DATASET_URL = "https://huggingface.co/datasets/beachside1234/FNSPID"
FNSPID_FORWARD_TRADING_DAYS = 2
FNSPID_RETURN_THRESHOLD = 0.01
FNSPID_MAX_ROWS_PER_CLASS = 4000

LABEL_BULLISH = "Bullish"
LABEL_NEUTRAL = "Neutral"
LABEL_BEARISH = "Bearish"
FINAL_LABEL_ORDER = [LABEL_BEARISH, LABEL_NEUTRAL, LABEL_BULLISH]


def load_source_feature_table(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            a.url,
            a.ticker,
            a.sector,
            a.industry,
            a.title,
            a.content,
            s.polarity,
            s.polarity_conf,
            s.uncertainty,
            COALESCE(s.litigious, 0.0) AS litigious,
            COALESCE(s.fundamental_strength, 0.0) AS fundamental_strength
        FROM article_scores s
        JOIN articles a ON s.url = a.url
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        raise ValueError("No processed articles found in article_scores.")
    df["text"] = (
        df["title"].fillna("").astype(str)
        + ". "
        + df["content"].fillna("").astype(str).str[:2200]
    ).str.strip()
    df["source_ticker"] = df["ticker"].fillna("UNKNOWN").astype(str)
    return df


def build_numeric_features(df: pd.DataFrame):
    enriched = df.copy()
    enriched["risk_adjusted_sentiment"] = enriched["polarity"] * enriched["polarity_conf"]
    enriched["headline_conviction"] = enriched["polarity_conf"] * (1.0 - enriched["uncertainty"])
    # `fundamental_strength` measures how much the text discusses fundamentals.
    # For market impact, we need a directional signal: strong fundamentals in a
    # bearish article should hurt, not help. We therefore sign it with sentiment.
    enriched["fundamental_impact"] = (
        enriched["fundamental_strength"] * enriched["risk_adjusted_sentiment"]
    )
    enriched["business_quality_score"] = (
        0.70 * enriched["fundamental_impact"]
        + 0.30 * enriched["headline_conviction"]
    )
    enriched["risk_pressure"] = (0.55 * enriched["uncertainty"]) + (0.45 * enriched["litigious"])
    enriched["market_signal_score"] = (
        1.00 * enriched["risk_adjusted_sentiment"]
        + 1.10 * enriched["fundamental_impact"]
        + 0.20 * enriched["headline_conviction"]
        - 0.85 * enriched["uncertainty"]
        - 0.35 * enriched["litigious"]
    )

    feature_columns = [
        "polarity",
        "polarity_conf",
        "uncertainty",
        "litigious",
        "fundamental_strength",
        "risk_adjusted_sentiment",
        "headline_conviction",
        "fundamental_impact",
        "business_quality_score",
        "risk_pressure",
        "market_signal_score",
    ]

    enriched[feature_columns] = enriched[feature_columns].fillna(0.0)
    return enriched, feature_columns


def choose_stratify_labels(labels: pd.Series):
    counts = labels.value_counts()
    if counts.min() < 2:
        return None
    return labels


_PRICE_CACHE = {}


def load_price_history(symbol: str):
    symbol = str(symbol).strip()
    if not symbol:
        return None
    if symbol in _PRICE_CACHE:
        return _PRICE_CACHE[symbol]

    price_df = None
    for candidate in [symbol.upper(), symbol.lower()]:
        try:
            path = hf_hub_download(
                FNSPID_PRICE_DATASET,
                f"Stock_price/full_history/{candidate}.parquet",
                repo_type="dataset",
            )
            raw_df = pd.read_parquet(path)
            price_column = "adj close" if "adj close" in raw_df.columns else "close"
            raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
            price_df = (
                raw_df.dropna(subset=["date", price_column])
                .sort_values("date")
                .drop_duplicates(subset=["date"])
                [["date", price_column]]
                .rename(columns={price_column: "price"})
                .reset_index(drop=True)
            )
            break
        except Exception:
            continue

    _PRICE_CACHE[symbol] = price_df
    return price_df


def compute_forward_return(symbol: str, article_date, horizon_days: int):
    price_df = load_price_history(symbol)
    if price_df is None or price_df.empty:
        return np.nan

    dates = price_df["date"].values.astype("datetime64[ns]")
    start_idx = int(np.searchsorted(dates, np.datetime64(article_date), side="left"))
    end_idx = start_idx + horizon_days
    if start_idx < 0 or end_idx >= len(price_df):
        return np.nan

    start_price = float(price_df.iloc[start_idx]["price"])
    end_price = float(price_df.iloc[end_idx]["price"])
    if start_price == 0.0 or np.isnan(start_price) or np.isnan(end_price):
        return np.nan
    return (end_price / start_price) - 1.0


def return_to_market_impact(forward_return: float) -> str:
    if forward_return > FNSPID_RETURN_THRESHOLD:
        return LABEL_BULLISH
    if forward_return < -FNSPID_RETURN_THRESHOLD:
        return LABEL_BEARISH
    return LABEL_NEUTRAL


def build_fnspid_text(row: pd.Series) -> str:
    title = str(row.get("Article_title") or "").strip()
    summary = (
        str(row.get("Lexrank_summary") or "").strip()
        or str(row.get("Textrank_summary") or "").strip()
        or str(row.get("Lsa_summary") or "").strip()
    )
    article = str(row.get("Article") or "").strip()
    body = summary or article[:2200]
    return f"{title}. {body}".strip()[:2500]


def balanced_time_split(df: pd.DataFrame) -> pd.DataFrame:
    balanced_parts = []
    for label in FINAL_LABEL_ORDER:
        subset = df[df["market_impact_label"] == label].sort_values("article_date")
        if len(subset) > FNSPID_MAX_ROWS_PER_CLASS:
            subset = subset.sample(FNSPID_MAX_ROWS_PER_CLASS, random_state=42)
        balanced_parts.append(subset)

    balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sort_values("article_date")
        .reset_index(drop=True)
    )
    split_index = int(len(balanced) * 0.80)
    balanced["source_split"] = "train"
    balanced.loc[balanced.index >= split_index, "source_split"] = "validation"
    return balanced


def load_supervised_external_dataset() -> pd.DataFrame:
    article_df = load_dataset(MARKET_IMPACT_DATASET, split="train").to_pandas()
    article_df["article_date"] = (
        pd.to_datetime(article_df["Date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    article_df = article_df.dropna(subset=["article_date", "Stock_symbol"]).copy()

    forward_returns = []
    for row in article_df[["Stock_symbol", "article_date"]].itertuples(index=False):
        forward_returns.append(
            compute_forward_return(
                row.Stock_symbol,
                row.article_date,
                FNSPID_FORWARD_TRADING_DAYS,
            )
        )

    article_df["forward_return"] = forward_returns
    article_df = article_df.dropna(subset=["forward_return"]).copy()
    article_df["market_impact_label"] = article_df["forward_return"].apply(
        return_to_market_impact
    )
    article_df["text"] = article_df.apply(build_fnspid_text, axis=1)
    article_df = article_df[article_df["text"].str.len() > 20].copy()
    article_df = article_df.drop_duplicates(subset=["Url", "Stock_symbol"])
    article_df = balanced_time_split(article_df)

    df = article_df[
        [
            "text",
            "market_impact_label",
            "source_split",
            "Stock_symbol",
            "Date",
            "Url",
            "forward_return",
        ]
    ].rename(
        columns={
            "Stock_symbol": "source_ticker",
            "Date": "source_date",
            "Url": "source_url",
        }
    )
    df["source_dataset"] = MARKET_IMPACT_DATASET
    df["source_label"] = df["forward_return"]
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df.reset_index(drop=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    df.to_csv(RAW_SUPERVISED_DATASET_FILE, index=False, encoding="utf-8")
    return df


def build_supervised_training_features(force_recompute: bool = False) -> pd.DataFrame:
    if os.path.exists(SUPERVISED_FEATURES_FILE) and not force_recompute:
        cached_df = pd.read_csv(SUPERVISED_FEATURES_FILE)
        if (
            "forward_return" in cached_df.columns
            and "source_dataset" in cached_df.columns
            and set(cached_df["source_dataset"].dropna().unique()) == {MARKET_IMPACT_DATASET}
        ):
            return cached_df

    dataset_df = load_supervised_external_dataset()
    texts = dataset_df["text"].tolist()

    polarity_agent = PolarityAgent()

    uncertainty_agent = None
    uncertainty_lexicon = None
    if os.path.exists("./uncertainty_model"):
        uncertainty_agent = UncertaintyAgent(model_path="./uncertainty_model")
    else:
        uncertainty_lexicon = download_lm_uncertainty_lexicon()

    litigious_agent = LitigiousAgent(
        model_path="./litigious_model",
        fallback_to_heuristic=True,
    )
    fundamental_strength_agent = FundamentalStrengthAgent(
        model_path="./fundamental_strength_model",
        fallback_to_heuristic=True,
    )

    polarity_results = polarity_agent.predict_batch(texts, batch_size=32)
    dataset_df["polarity"] = [result[0] for result in polarity_results]
    dataset_df["polarity_conf"] = [result[1] for result in polarity_results]
    dataset_df["polarity_label"] = [result[2] for result in polarity_results]

    if uncertainty_agent:
        dataset_df["uncertainty"] = uncertainty_agent.predict_batch(texts, batch_size=32)
    else:
        dataset_df["uncertainty"] = [
            compute_uncertainty_score(text, uncertainty_lexicon) for text in texts
        ]

    dataset_df["litigious"] = litigious_agent.predict_batch(texts)
    dataset_df["fundamental_strength"] = fundamental_strength_agent.predict_batch(texts)
    dataset_df.to_csv(SUPERVISED_FEATURES_FILE, index=False, encoding="utf-8")
    return dataset_df


def fit_kmeans_auxiliary_features(
    df: pd.DataFrame,
    base_feature_columns,
    label_column: str,
):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[base_feature_columns])

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(x_scaled)

    cluster_distance_columns = []
    distances = kmeans.transform(x_scaled)
    for idx in range(distances.shape[1]):
        column_name = f"cluster_distance_{idx}"
        df[column_name] = distances[:, idx]
        cluster_distance_columns.append(column_name)

    cluster_sizes = df["cluster_id"].value_counts()
    if len(cluster_sizes) == 3 and cluster_sizes.min() > 1:
        sil_score = float(silhouette_score(x_scaled, df["cluster_id"]))
    else:
        sil_score = 0.0

    cluster_profiles = []
    cluster_label_map = {}
    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]
        label_counts = cluster_df[label_column].value_counts()
        majority_label = label_counts.idxmax()
        majority_share = float(label_counts.max() / len(cluster_df))
        cluster_label_map[int(cluster_id)] = majority_label
        cluster_profiles.append(
            {
                "cluster_id": int(cluster_id),
                "support": int(len(cluster_df)),
                "market_impact_label": majority_label,
                "majority_share": round(majority_share, 6),
                "cluster_score": float(cluster_df["market_signal_score"].mean()),
                "mean_polarity": float(cluster_df["polarity"].mean()),
                "mean_confidence": float(cluster_df["polarity_conf"].mean()),
                "mean_uncertainty": float(cluster_df["uncertainty"].mean()),
                "mean_litigious": float(cluster_df["litigious"].mean()),
                "mean_fundamental_strength": float(cluster_df["fundamental_strength"].mean()),
                "mean_fundamental_impact": float(cluster_df["fundamental_impact"].mean()),
                "mean_business_quality": float(cluster_df["business_quality_score"].mean()),
                "mean_risk_pressure": float(cluster_df["risk_pressure"].mean()),
                "label_distribution": label_counts.to_dict(),
            }
        )

    df["cluster_label"] = df["cluster_id"].map(cluster_label_map)
    cluster_alignment_accuracy = float(accuracy_score(df[label_column], df["cluster_label"]))
    cluster_alignment_macro_f1 = float(
        f1_score(df[label_column], df["cluster_label"], average="macro")
    )

    cluster_metrics = {
        "silhouette_score": round(sil_score, 6),
        "cluster_alignment_accuracy": round(cluster_alignment_accuracy, 6),
        "cluster_alignment_macro_f1": round(cluster_alignment_macro_f1, 6),
    }

    return df, scaler, kmeans, cluster_distance_columns, cluster_profiles, cluster_label_map, cluster_metrics


def add_cluster_features(df: pd.DataFrame, scaler, kmeans, base_feature_columns, cluster_label_map):
    enriched = df.copy()
    x_scaled = scaler.transform(enriched[base_feature_columns])
    enriched["cluster_id"] = kmeans.predict(x_scaled)
    distances = kmeans.transform(x_scaled)

    cluster_distance_columns = []
    for idx in range(distances.shape[1]):
        column_name = f"cluster_distance_{idx}"
        enriched[column_name] = distances[:, idx]
        cluster_distance_columns.append(column_name)

    enriched["cluster_label"] = enriched["cluster_id"].map(cluster_label_map)
    return enriched, cluster_distance_columns


def build_market_impact_classifier(numeric_columns):
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("numeric", StandardScaler(), numeric_columns),
                        (
                            "ticker",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["source_ticker"],
                        ),
                        (
                            "text",
                            TfidfVectorizer(
                                max_features=10000,
                                ngram_range=(1, 2),
                                min_df=3,
                                max_df=0.90,
                                sublinear_tf=True,
                                stop_words="english",
                            ),
                            "text",
                        ),
                    ],
                    sparse_threshold=0.30,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    C=0.20,
                    solver="saga",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def persist_results(
    db_path: str,
    scored_df: pd.DataFrame,
    cluster_profiles,
    metrics: dict,
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS market_impact_predictions (
            url TEXT PRIMARY KEY,
            cluster_id INTEGER,
            market_impact_label TEXT,
            market_signal_score REAL,
            prob_bearish REAL,
            prob_neutral REAL,
            prob_bullish REAL,
            generated_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS market_impact_cluster_profiles (
            cluster_id INTEGER PRIMARY KEY,
            market_impact_label TEXT,
            support INTEGER,
            cluster_score REAL,
            mean_polarity REAL,
            mean_confidence REAL,
            mean_uncertainty REAL,
            mean_litigious REAL,
            silhouette_score REAL,
            classifier_accuracy REAL,
            classifier_macro_f1 REAL,
            generated_at TEXT
        )
        """
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    for _, row in scored_df.iterrows():
        cursor.execute(
            """
            INSERT OR REPLACE INTO market_impact_predictions (
                url,
                cluster_id,
                market_impact_label,
                market_signal_score,
                prob_bearish,
                prob_neutral,
                prob_bullish,
                generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["url"],
                int(row["cluster_id"]),
                row["market_impact_label"],
                float(row["market_signal_score"]),
                float(row["prob_bearish"]),
                float(row["prob_neutral"]),
                float(row["prob_bullish"]),
                generated_at,
            ),
        )

    for profile in cluster_profiles:
        cursor.execute(
            """
            INSERT OR REPLACE INTO market_impact_cluster_profiles (
                cluster_id,
                market_impact_label,
                support,
                cluster_score,
                mean_polarity,
                mean_confidence,
                mean_uncertainty,
                mean_litigious,
                silhouette_score,
                classifier_accuracy,
                classifier_macro_f1,
                generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(profile["cluster_id"]),
                profile["market_impact_label"],
                int(profile["support"]),
                float(profile["cluster_score"]),
                float(profile["mean_polarity"]),
                float(profile["mean_confidence"]),
                float(profile["mean_uncertainty"]),
                float(profile["mean_litigious"]),
                float(metrics["silhouette_score"]),
                float(metrics["accuracy"]),
                float(metrics["macro_f1"]),
                generated_at,
            ),
        )

    conn.commit()
    conn.close()


def run_supervised_external_pipeline(
    db_path: str = DB_PATH,
    force_recompute_training_features: bool = False,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading external supervised financial datasets...")
    training_df = build_supervised_training_features(
        force_recompute=force_recompute_training_features
    )
    training_df, base_feature_columns = build_numeric_features(training_df)
    train_df = training_df[training_df["source_split"] == "train"].copy()
    benchmark_df = training_df[training_df["source_split"] == "validation"].copy()
    if train_df.empty or benchmark_df.empty:
        raise ValueError("The market impact benchmark must contain train and validation splits.")

    print("Running KMeans auxiliary clustering on supervised feature space...")
    (
        train_df,
        cluster_scaler,
        kmeans,
        cluster_distance_columns,
        cluster_profiles,
        cluster_label_map,
        cluster_metrics,
    ) = fit_kmeans_auxiliary_features(
        train_df,
        base_feature_columns,
        "market_impact_label",
    )
    benchmark_df, _ = add_cluster_features(
        benchmark_df,
        cluster_scaler,
        kmeans,
        base_feature_columns,
        cluster_label_map,
    )

    numeric_classifier_columns = base_feature_columns + cluster_distance_columns
    classifier_feature_columns = ["text", "source_ticker"] + numeric_classifier_columns
    for frame in [train_df, benchmark_df]:
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["source_ticker"] = frame["source_ticker"].fillna("UNKNOWN").astype(str)

    print("Training multiclass classifier on public market-impact labels...")
    x_train = train_df[classifier_feature_columns]
    y_train = train_df["market_impact_label"]
    x_test = benchmark_df[classifier_feature_columns]
    y_test = benchmark_df["market_impact_label"]

    classifier = build_market_impact_classifier(numeric_classifier_columns)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    train_majority_label = y_train.value_counts().idxmax()
    validation_majority_label = y_test.value_counts().idxmax()
    train_majority_pred = [train_majority_label] * len(y_test)
    validation_majority_pred = [validation_majority_label] * len(y_test)

    source_df = load_source_feature_table(db_path)
    source_df, _ = build_numeric_features(source_df)
    source_df, _ = add_cluster_features(
        source_df,
        cluster_scaler,
        kmeans,
        base_feature_columns,
        cluster_label_map,
    )
    source_df["text"] = source_df["text"].fillna("").astype(str)
    source_df["source_ticker"] = source_df["source_ticker"].fillna("UNKNOWN").astype(str)
    source_x = source_df[classifier_feature_columns]
    source_pred = classifier.predict(source_x)
    source_prob = classifier.predict_proba(source_x)

    probability_df = pd.DataFrame(source_prob, columns=classifier.classes_)
    for label in FINAL_LABEL_ORDER:
        if label not in probability_df.columns:
            probability_df[label] = 0.0

    source_df["prob_bearish"] = probability_df[LABEL_BEARISH].values
    source_df["prob_neutral"] = probability_df[LABEL_NEUTRAL].values
    source_df["prob_bullish"] = probability_df[LABEL_BULLISH].values
    source_df["market_impact_label"] = source_pred

    metrics = {
        "dataset_size": int(len(training_df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "output_name": "market_impact_label",
        "label_source": MARKET_IMPACT_DATASET,
        "benchmark_dataset_url": MARKET_IMPACT_DATASET_URL,
        "price_dataset": FNSPID_PRICE_DATASET,
        "price_dataset_url": FNSPID_PRICE_DATASET_URL,
        "benchmark_train_split": "train",
        "benchmark_test_split": "validation",
        "forward_trading_days": FNSPID_FORWARD_TRADING_DAYS,
        "return_threshold": FNSPID_RETURN_THRESHOLD,
        "classifier_type": "hybrid_tfidf_numeric_logistic_regression",
        "classifier_inputs": {
            "text": "TF-IDF word ngrams from title and article/summary text",
            "ticker": "one-hot encoded source ticker",
            "numeric_vectors": numeric_classifier_columns,
        },
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "log_loss": round(float(log_loss(y_test, y_prob, labels=classifier.classes_)), 6),
        "uniform_random_baseline_accuracy": round(1.0 / len(FINAL_LABEL_ORDER), 6),
        "train_majority_baseline_label": train_majority_label,
        "train_majority_baseline_accuracy": round(
            float(accuracy_score(y_test, train_majority_pred)),
            6,
        ),
        "train_majority_baseline_macro_f1": round(
            float(f1_score(y_test, train_majority_pred, average="macro")),
            6,
        ),
        "validation_majority_label": validation_majority_label,
        "validation_majority_accuracy": round(
            float(accuracy_score(y_test, validation_majority_pred)),
            6,
        ),
        "validation_majority_macro_f1": round(
            float(f1_score(y_test, validation_majority_pred, average="macro")),
            6,
        ),
        "silhouette_score": cluster_metrics["silhouette_score"],
        "cluster_alignment_accuracy": cluster_metrics["cluster_alignment_accuracy"],
        "cluster_alignment_macro_f1": cluster_metrics["cluster_alignment_macro_f1"],
        "class_distribution": y_train.value_counts().to_dict(),
        "benchmark_class_distribution": y_test.value_counts().to_dict(),
        "source_prediction_distribution": pd.Series(source_pred).value_counts().to_dict(),
        "cluster_profiles": cluster_profiles,
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=FINAL_LABEL_ORDER,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=FINAL_LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "feature_columns": classifier_feature_columns,
        "numeric_classifier_columns": numeric_classifier_columns,
        "base_feature_columns": base_feature_columns,
        "cluster_feature_columns": cluster_distance_columns,
        "training_datasets": train_df["source_dataset"].value_counts().to_dict(),
        "label_mapping": {
            MARKET_IMPACT_DATASET: {
                f"forward_return > {FNSPID_RETURN_THRESHOLD}": LABEL_BULLISH,
                f"forward_return < -{FNSPID_RETURN_THRESHOLD}": LABEL_BEARISH,
                f"|forward_return| <= {FNSPID_RETURN_THRESHOLD}": LABEL_NEUTRAL,
            }
        },
    }

    dump(classifier, CLASSIFIER_FILE)
    dump(
        {
            "scaler": cluster_scaler,
            "kmeans": kmeans,
            "base_feature_columns": base_feature_columns,
            "cluster_feature_columns": cluster_distance_columns,
            "cluster_label_map": cluster_label_map,
        },
        CLUSTER_FILE,
    )

    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with open(FEATURES_FILE, "w", encoding="utf-8") as handle:
        json.dump(classifier_feature_columns, handle, indent=2)

    persist_results(db_path, source_df, cluster_profiles, metrics)

    print("Market impact processing complete (supervised_external).")
    print(f"  Training rows           : {len(train_df)}")
    print(f"  Benchmark rows          : {len(benchmark_df)}")
    print(f"  Source articles scored  : {len(source_df)}")
    print(f"  Cluster alignment acc   : {metrics['cluster_alignment_accuracy']:.4f}")
    print(f"  Classifier accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Classifier macro F1     : {metrics['macro_f1']:.4f}")
    print("  Source label distribution:")
    for label, count in pd.Series(source_pred).value_counts().items():
        print(f"    - {label}: {count}")

    return source_df, metrics


def map_clusters_to_labels(df: pd.DataFrame):
    cluster_summary = []
    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]
        cluster_score = float(cluster_df["market_signal_score"].mean())
        cluster_summary.append(
            {
                "cluster_id": int(cluster_id),
                "support": int(len(cluster_df)),
                "cluster_score": cluster_score,
                "mean_polarity": float(cluster_df["polarity"].mean()),
                "mean_confidence": float(cluster_df["polarity_conf"].mean()),
                "mean_uncertainty": float(cluster_df["uncertainty"].mean()),
                "mean_litigious": float(cluster_df["litigious"].mean()),
                "mean_fundamental_strength": float(cluster_df["fundamental_strength"].mean()),
                "mean_business_quality": float(cluster_df["business_quality_score"].mean()),
                "mean_risk_pressure": float(cluster_df["risk_pressure"].mean()),
            }
        )

    ordered_clusters = sorted(cluster_summary, key=lambda item: item["cluster_score"])
    label_map = {
        ordered_clusters[0]["cluster_id"]: LABEL_BEARISH,
        ordered_clusters[1]["cluster_id"]: LABEL_NEUTRAL,
        ordered_clusters[2]["cluster_id"]: LABEL_BULLISH,
    }

    df["market_impact_label"] = df["cluster_id"].map(label_map)
    cluster_profiles = pd.DataFrame(cluster_summary)
    cluster_profiles["market_impact_label"] = cluster_profiles["cluster_id"].map(label_map)
    return df, cluster_profiles


def run_pseudo_cluster_pipeline(db_path: str = DB_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading processed article vectors...")
    df = load_source_feature_table(db_path)
    df, feature_columns = build_numeric_features(df)

    print("Running clustering (K=3)...")
    cluster_scaler = StandardScaler()
    x_cluster = cluster_scaler.fit_transform(df[feature_columns])
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    df["cluster_id"] = kmeans.fit_predict(x_cluster)

    cluster_sizes = df["cluster_id"].value_counts()
    if len(cluster_sizes) == 3 and cluster_sizes.min() > 1:
        sil_score = float(silhouette_score(x_cluster, df["cluster_id"]))
    else:
        sil_score = 0.0

    df, cluster_profiles = map_clusters_to_labels(df)

    print("Training market impact classifier on pseudo-labels...")
    x = df[feature_columns]
    y = df["market_impact_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=choose_stratify_labels(y),
    )

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)

    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "output_name": "market_impact_label",
        "label_source": "kmeans_pseudo_labels",
        "classifier_type": "multiclass_logistic_regression",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "log_loss": round(float(log_loss(y_test, y_prob, labels=classifier.classes_)), 6),
        "silhouette_score": round(float(sil_score), 6),
        "class_distribution": y.value_counts().to_dict(),
        "cluster_profiles": cluster_profiles.to_dict(orient="records"),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=FINAL_LABEL_ORDER,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=FINAL_LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "feature_columns": feature_columns,
    }

    dump(classifier, CLASSIFIER_FILE)
    dump(
        {
            "scaler": cluster_scaler,
            "kmeans": kmeans,
            "feature_columns": feature_columns,
        },
        CLUSTER_FILE,
    )

    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with open(FEATURES_FILE, "w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)

    all_probabilities = classifier.predict_proba(x)
    probability_df = pd.DataFrame(all_probabilities, columns=classifier.classes_)
    for label in FINAL_LABEL_ORDER:
        if label not in probability_df.columns:
            probability_df[label] = 0.0

    df["prob_bearish"] = probability_df[LABEL_BEARISH].values
    df["prob_neutral"] = probability_df[LABEL_NEUTRAL].values
    df["prob_bullish"] = probability_df[LABEL_BULLISH].values
    df["market_impact_label"] = classifier.predict(x)

    persist_results(db_path, df, metrics["cluster_profiles"], metrics)

    print("Market impact processing complete (pseudo_cluster).")
    print(f"  Articles processed     : {len(df)}")
    print(f"  Silhouette score       : {metrics['silhouette_score']:.4f}")
    print(f"  Classifier accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Classifier macro F1    : {metrics['macro_f1']:.4f}")
    print("  Label distribution:")
    for label, count in df["market_impact_label"].value_counts().items():
        print(f"    - {label}: {count}")

    return df, metrics


def main():
    parser = argparse.ArgumentParser(description="Market impact processing pipeline")
    parser.add_argument(
        "--mode",
        choices=["supervised_external", "pseudo_cluster"],
        default="supervised_external",
        help="Mode d'entrainement du classifieur final",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="Chemin vers la base SQLite source",
    )
    parser.add_argument(
        "--force-recompute-training-features",
        action="store_true",
        help="Recalcule les features du dataset externe au lieu d'utiliser le cache CSV",
    )
    args = parser.parse_args()

    if args.mode == "pseudo_cluster":
        run_pseudo_cluster_pipeline(db_path=args.db)
    else:
        run_supervised_external_pipeline(
            db_path=args.db,
            force_recompute_training_features=args.force_recompute_training_features,
        )


if __name__ == "__main__":
    main()
