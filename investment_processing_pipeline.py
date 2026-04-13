"""
Investment processing pipeline
==============================

Modes disponibles :

1. supervised_external
   - entraine la classification finale sur de vrais datasets financiers etiquetes
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

import pandas as pd
from datasets import load_dataset
from joblib import dump
from sklearn.cluster import KMeans
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
from sklearn.preprocessing import StandardScaler

from fundamental_strength_agent import FundamentalStrengthAgent
from litigious_agent import LitigiousAgent
from polarity_agent import PolarityAgent
from uncertainty_agent import (
    UncertaintyAgent,
    compute_uncertainty_score,
    download_lm_uncertainty_lexicon,
)


DB_PATH = "news_database.db"
MODEL_DIR = "./investment_model"
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "classifier.joblib")
CLUSTER_FILE = os.path.join(MODEL_DIR, "kmeans.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
FEATURES_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
RAW_SUPERVISED_DATASET_FILE = os.path.join(MODEL_DIR, "external_supervised_dataset.csv")
SUPERVISED_FEATURES_FILE = os.path.join(MODEL_DIR, "external_supervised_features.csv")

LABEL_GOOD = "Good Investment"
LABEL_WATCHLIST = "Watchlist"
LABEL_BAD = "Do Not Invest"
FINAL_LABEL_ORDER = [LABEL_BAD, LABEL_WATCHLIST, LABEL_GOOD]


def load_source_feature_table(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            a.url,
            a.ticker,
            a.sector,
            a.industry,
            a.title,
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
    return df


def build_numeric_features(df: pd.DataFrame):
    enriched = df.copy()
    enriched["risk_adjusted_sentiment"] = enriched["polarity"] * enriched["polarity_conf"]
    enriched["headline_conviction"] = enriched["polarity_conf"] * (1.0 - enriched["uncertainty"])
    # `fundamental_strength` measures how much the text discusses fundamentals.
    # For investment scoring, we need a directional signal: strong fundamentals in
    # a negative article should hurt, not help. We therefore sign it with sentiment.
    enriched["fundamental_impact"] = (
        enriched["fundamental_strength"] * enriched["risk_adjusted_sentiment"]
    )
    enriched["business_quality_score"] = (
        0.70 * enriched["fundamental_impact"]
        + 0.30 * enriched["headline_conviction"]
    )
    enriched["risk_pressure"] = (0.55 * enriched["uncertainty"]) + (0.45 * enriched["litigious"])
    enriched["investment_profile_score"] = (
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
        "investment_profile_score",
    ]

    enriched[feature_columns] = enriched[feature_columns].fillna(0.0)
    return enriched, feature_columns


def choose_stratify_labels(labels: pd.Series):
    counts = labels.value_counts()
    if counts.min() < 2:
        return None
    return labels


def phrasebank_label_to_investment(label_name: str) -> str:
    mapping = {
        "positive": LABEL_GOOD,
        "neutral": LABEL_WATCHLIST,
        "negative": LABEL_BAD,
    }
    return mapping[label_name]


def fiqa_score_to_investment(score: float) -> str:
    if score >= 0.15:
        return LABEL_GOOD
    if score <= -0.15:
        return LABEL_BAD
    return LABEL_WATCHLIST


def load_supervised_external_dataset() -> pd.DataFrame:
    phrasebank = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        split="train",
        trust_remote_code=True,
    )
    phrasebank_rows = []
    label_names = phrasebank.features["label"].names
    for row in phrasebank:
        phrasebank_rows.append(
            {
                "text": row["sentence"],
                "investment_label": phrasebank_label_to_investment(label_names[row["label"]]),
                "source_dataset": "financial_phrasebank",
                "source_label": label_names[row["label"]],
            }
        )

    fiqa = load_dataset("TheFinAI/fiqa-sentiment-classification")
    fiqa_rows = []
    for split_name in ["train", "valid", "test"]:
        for row in fiqa[split_name]:
            fiqa_rows.append(
                {
                    "text": row["sentence"],
                    "investment_label": fiqa_score_to_investment(float(row["score"])),
                    "source_dataset": f"fiqa_{split_name}",
                    "source_label": float(row["score"]),
                }
            )

    df = pd.DataFrame(phrasebank_rows + fiqa_rows)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 20].drop_duplicates(subset=["text"]).reset_index(drop=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    df.to_csv(RAW_SUPERVISED_DATASET_FILE, index=False, encoding="utf-8")
    return df


def build_supervised_training_features(force_recompute: bool = False) -> pd.DataFrame:
    if os.path.exists(SUPERVISED_FEATURES_FILE) and not force_recompute:
        return pd.read_csv(SUPERVISED_FEATURES_FILE)

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
                "investment_label": majority_label,
                "majority_share": round(majority_share, 6),
                "cluster_score": float(cluster_df["investment_profile_score"].mean()),
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
        CREATE TABLE IF NOT EXISTS investment_recommendations (
            url TEXT PRIMARY KEY,
            cluster_id INTEGER,
            investment_label TEXT,
            investment_profile_score REAL,
            prob_do_not_invest REAL,
            prob_watchlist REAL,
            prob_good_investment REAL,
            generated_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_cluster_profiles (
            cluster_id INTEGER PRIMARY KEY,
            investment_label TEXT,
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
            INSERT OR REPLACE INTO investment_recommendations (
                url,
                cluster_id,
                investment_label,
                investment_profile_score,
                prob_do_not_invest,
                prob_watchlist,
                prob_good_investment,
                generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["url"],
                int(row["cluster_id"]),
                row["investment_label"],
                float(row["investment_profile_score"]),
                float(row["prob_do_not_invest"]),
                float(row["prob_watchlist"]),
                float(row["prob_good_investment"]),
                generated_at,
            ),
        )

    for profile in cluster_profiles:
        cursor.execute(
            """
            INSERT OR REPLACE INTO investment_cluster_profiles (
                cluster_id,
                investment_label,
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
                profile["investment_label"],
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

    print("Running KMeans auxiliary clustering on supervised feature space...")
    (
        training_df,
        cluster_scaler,
        kmeans,
        cluster_distance_columns,
        cluster_profiles,
        cluster_label_map,
        cluster_metrics,
    ) = fit_kmeans_auxiliary_features(
        training_df,
        base_feature_columns,
        "investment_label",
    )

    classifier_feature_columns = base_feature_columns + cluster_distance_columns

    print("Training multiclass classifier on external labels...")
    x = training_df[classifier_feature_columns]
    y = training_df["investment_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
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

    source_df = load_source_feature_table(db_path)
    source_df, _ = build_numeric_features(source_df)
    source_df, _ = add_cluster_features(
        source_df,
        cluster_scaler,
        kmeans,
        base_feature_columns,
        cluster_label_map,
    )
    source_x = source_df[classifier_feature_columns]
    source_pred = classifier.predict(source_x)
    source_prob = classifier.predict_proba(source_x)

    probability_df = pd.DataFrame(source_prob, columns=classifier.classes_)
    for label in FINAL_LABEL_ORDER:
        if label not in probability_df.columns:
            probability_df[label] = 0.0

    source_df["prob_do_not_invest"] = probability_df[LABEL_BAD].values
    source_df["prob_watchlist"] = probability_df[LABEL_WATCHLIST].values
    source_df["prob_good_investment"] = probability_df[LABEL_GOOD].values
    source_df["investment_label"] = source_pred

    metrics = {
        "dataset_size": int(len(training_df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "label_source": "external_supervised_financial_datasets",
        "classifier_type": "multiclass_logistic_regression",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "log_loss": round(float(log_loss(y_test, y_prob, labels=classifier.classes_)), 6),
        "silhouette_score": cluster_metrics["silhouette_score"],
        "cluster_alignment_accuracy": cluster_metrics["cluster_alignment_accuracy"],
        "cluster_alignment_macro_f1": cluster_metrics["cluster_alignment_macro_f1"],
        "class_distribution": y.value_counts().to_dict(),
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
        "base_feature_columns": base_feature_columns,
        "cluster_feature_columns": cluster_distance_columns,
        "training_datasets": training_df["source_dataset"].value_counts().to_dict(),
        "label_mapping": {
            "financial_phrasebank": {
                "positive": LABEL_GOOD,
                "neutral": LABEL_WATCHLIST,
                "negative": LABEL_BAD,
            },
            "fiqa": {
                "score >= 0.15": LABEL_GOOD,
                "-0.15 < score < 0.15": LABEL_WATCHLIST,
                "score <= -0.15": LABEL_BAD,
            },
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

    print("Investment processing complete (supervised_external).")
    print(f"  Training rows           : {len(training_df)}")
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
        cluster_score = float(cluster_df["investment_profile_score"].mean())
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
        ordered_clusters[0]["cluster_id"]: LABEL_BAD,
        ordered_clusters[1]["cluster_id"]: LABEL_WATCHLIST,
        ordered_clusters[2]["cluster_id"]: LABEL_GOOD,
    }

    df["investment_label"] = df["cluster_id"].map(label_map)
    cluster_profiles = pd.DataFrame(cluster_summary)
    cluster_profiles["investment_label"] = cluster_profiles["cluster_id"].map(label_map)
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

    print("Training investment classifier on pseudo-labels...")
    x = df[feature_columns]
    y = df["investment_label"]

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

    df["prob_do_not_invest"] = probability_df[LABEL_BAD].values
    df["prob_watchlist"] = probability_df[LABEL_WATCHLIST].values
    df["prob_good_investment"] = probability_df[LABEL_GOOD].values
    df["investment_label"] = classifier.predict(x)

    persist_results(db_path, df, metrics["cluster_profiles"], metrics)

    print("Investment processing complete (pseudo_cluster).")
    print(f"  Articles processed     : {len(df)}")
    print(f"  Silhouette score       : {metrics['silhouette_score']:.4f}")
    print(f"  Classifier accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Classifier macro F1    : {metrics['macro_f1']:.4f}")
    print("  Label distribution:")
    for label, count in df["investment_label"].value_counts().items():
        print(f"    - {label}: {count}")

    return df, metrics


def main():
    parser = argparse.ArgumentParser(description="Investment processing pipeline")
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
