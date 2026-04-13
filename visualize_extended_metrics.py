"""
Visualisations essentielles des metriques
========================================

Genere uniquement les graphes utiles pour comprendre :
- les benchmarks des modeles
- la matrice de confusion du benchmark final
- les profils finaux des classes
- la projection 2D et la confiance sur les 67 articles
"""

import json
import os
import sqlite3
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "metrics_plots")
DB_PATH = os.path.join(OUTPUT_DIR, "news_database.db")

MARKET_IMPACT_METRICS_PATH = os.path.join(OUTPUT_DIR, "market_impact_model", "metrics.json")
MARKET_IMPACT_CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "market_impact_model", "classifier.joblib")
UNCERTAINTY_METRICS_PATH = os.path.join(OUTPUT_DIR, "uncertainty_model", "metrics.json")
FUNDAMENTAL_METRICS_PATH = os.path.join(OUTPUT_DIR, "fundamental_strength_model", "metrics.json")
LITIGIOUS_METRICS_PATH = os.path.join(OUTPUT_DIR, "litigious_model", "metrics.json")
FUNDAMENTAL_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "fundamental_strength_model", "benchmarks.json")
LITIGIOUS_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "litigious_model", "benchmarks.json")
POLARITY_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results", "polarity_benchmarks.json")
UNCERTAINTY_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results", "uncertainty_benchmarks.json")
OUTPUT_METRICS_SUMMARY_CSV = os.path.join(PLOTS_DIR, "output_metrics_summary.csv")
FINAL_MODEL_NUMERIC_WEIGHTS_CSV = os.path.join(PLOTS_DIR, "final_model_numeric_feature_weights.csv")
BASE_OUTPUT_WEIGHTS_CSV = os.path.join(PLOTS_DIR, "base_output_weight_summary.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
    }
)

C_BG = "#fafbfc"
C_GRID = "#dde4ea"
C_POS = "#2ecc71"
C_NEG = "#e74c3c"
C_BLUE = "#3498db"
C_ORANGE = "#f39c12"
C_PURPLE = "#8e44ad"

LABEL_COLORS = {
    "Bullish": C_POS,
    "Neutral": C_ORANGE,
    "Bearish": C_NEG,
}
DISPLAY_LABEL_ORDER = ["Bullish", "Neutral", "Bearish"]
BENCHMARK_LABEL_ORDER = ["Bearish", "Neutral", "Bullish"]
RAW_OUTPUTS = [
    "polarity",
    "polarity_conf",
    "uncertainty",
    "litigious",
    "fundamental_strength",
]


def cleanup_existing_plots():
    for file_name in os.listdir(PLOTS_DIR):
        if file_name.lower().endswith(".png"):
            os.remove(os.path.join(PLOTS_DIR, file_name))


def add_explanation(fig, text, y_pos=0.02):
    fig.text(
        0.5,
        y_pos,
        text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
        style="italic",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#eef2f7",
            edgecolor="#bdc3c7",
            alpha=0.9,
        ),
        transform=fig.transFigure,
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_article_level_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            a.title,
            a.ticker,
            s.polarity,
            s.polarity_conf,
            s.uncertainty,
            COALESCE(s.litigious, 0.0) AS litigious,
            COALESCE(s.fundamental_strength, 0.0) AS fundamental_strength,
            mi.market_impact_label,
            mi.market_signal_score,
            mi.prob_bearish,
            mi.prob_neutral,
            mi.prob_bullish
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        LEFT JOIN market_impact_predictions mi ON s.url = mi.url
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["risk_adjusted_sentiment"] = df["polarity"] * df["polarity_conf"]
    df["headline_conviction"] = df["polarity_conf"] * (1.0 - df["uncertainty"])
    df["fundamental_impact"] = df["fundamental_strength"] * df["risk_adjusted_sentiment"]
    df["risk_pressure"] = (0.55 * df["uncertainty"]) + (0.45 * df["litigious"])
    df["assigned_class_probability"] = np.select(
        [
            df["market_impact_label"] == "Bullish",
            df["market_impact_label"] == "Neutral",
            df["market_impact_label"] == "Bearish",
        ],
        [
            df["prob_bullish"],
            df["prob_neutral"],
            df["prob_bearish"],
        ],
        default=np.nan,
    )
    return df


def label_with_n(display_name, benchmark_dict, key):
    n_value = benchmark_dict[key].get("dataset_size")
    if n_value is None:
        return display_name
    return f"{display_name}\n(n={n_value})"


def _plot_grouped_bars(ax, benchmark_dict, metric_a, metric_b, label_a, label_b, title, name_map=None):
    names = list(benchmark_dict.keys())
    display_names = [name_map.get(name, name) if name_map else name for name in names]
    values_a = [benchmark_dict[name][metric_a] for name in names]
    values_b = [benchmark_dict[name][metric_b] for name in names]

    x = np.arange(len(names))
    width = 0.34

    ax.bar(x - width / 2, values_a, width=width, color=C_BLUE, label=label_a, edgecolor="white", linewidth=1.2)
    ax.bar(x + width / 2, values_b, width=width, color=C_POS, label=label_b, edgecolor="white", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.grid(axis="y", color=C_GRID, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    for idx, value in enumerate(values_a):
        ax.text(idx - width / 2, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")
    for idx, value in enumerate(values_b):
        ax.text(idx + width / 2, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")


def plot_classifier_benchmarks(polarity_benchmarks, market_impact_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
    fig.suptitle("1. Benchmarks des classifications", fontsize=16, fontweight="bold", y=0.97)

    polarity_name_map = {
        "phrasebank_sentiment": label_with_n("PhraseBank\n(sentiment clair)", polarity_benchmarks, "phrasebank_sentiment"),
        "fiqa_sentiment": label_with_n("FiQA\n(sentiment plus dur)", polarity_benchmarks, "fiqa_sentiment"),
    }
    _plot_grouped_bars(
        axes[0],
        polarity_benchmarks,
        "accuracy",
        "macro_f1",
        "Accuracy",
        "Macro-F1",
        "Polarity (sentiment)",
        name_map=polarity_name_map,
    )

    market_impact_benchmark = {
        "model": {
            "accuracy": market_impact_metrics["accuracy"],
            "macro_f1": market_impact_metrics["macro_f1"],
        },
        "chance": {
            "accuracy": market_impact_metrics.get("uniform_random_baseline_accuracy", 1 / 3),
            "macro_f1": 1 / 3,
        },
        "majority": {
            "accuracy": market_impact_metrics.get("validation_majority_accuracy", 0.0),
            "macro_f1": market_impact_metrics.get("validation_majority_macro_f1", 0.0),
        },
    }
    _plot_grouped_bars(
        axes[1],
        market_impact_benchmark,
        "accuracy",
        "macro_f1",
        "Accuracy",
        "Macro-F1",
        "Market impact final",
        name_map={
            "model": f"Notre modele\nFNSPID\n(n={market_impact_metrics['test_size']})",
            "chance": "Hasard\n3 classes",
            "majority": "Majorite\nvalidation",
        },
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    add_explanation(
        fig,
        "Le benchmark final utilise FNSPID : le label vient du vrai rendement du titre apres la news. On compare aussi au hasard et a une baseline qui predit toujours la classe majoritaire.",
    )

    path = os.path.join(PLOTS_DIR, "1_classifier_benchmarks.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_fnspid_confusion_matrix(market_impact_metrics):
    matrix = np.array(market_impact_metrics["confusion_matrix"], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_pct = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=C_BG)
    fig.suptitle("8. Matrice de confusion FNSPID", fontsize=16, fontweight="bold", y=0.97)

    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=max(0.55, float(row_pct.max())))
    ax.set_xticks(np.arange(len(BENCHMARK_LABEL_ORDER)))
    ax.set_yticks(np.arange(len(BENCHMARK_LABEL_ORDER)))
    ax.set_xticklabels(BENCHMARK_LABEL_ORDER, fontsize=11, fontweight="bold")
    ax.set_yticklabels(BENCHMARK_LABEL_ORDER, fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediction du modele")
    ax.set_ylabel("Vrai label FNSPID, derive du rendement reel")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            pct_value = row_pct[row_idx, col_idx] * 100
            count_value = int(matrix[row_idx, col_idx])
            text_color = "white" if row_pct[row_idx, col_idx] > 0.38 else "#1f2d3d"
            ax.text(
                col_idx,
                row_idx,
                f"{count_value}\n{pct_value:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    for edge in np.arange(-0.5, len(BENCHMARK_LABEL_ORDER), 1):
        ax.axvline(edge, color="white", linewidth=2)
        ax.axhline(edge, color="white", linewidth=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% par vraie classe")

    fig.tight_layout(rect=[0, 0.13, 1, 0.92])
    add_explanation(
        fig,
        f"Diagonale = bonnes predictions. Accuracy={market_impact_metrics['accuracy']:.3f}, Macro-F1={market_impact_metrics['macro_f1']:.3f}. Ici on teste le vrai mouvement de prix, donc c'est beaucoup plus dur que du sentiment.",
    )

    path = os.path.join(PLOTS_DIR, "8_fnspid_confusion_matrix.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_score_benchmarks(uncertainty_benchmarks, fundamental_benchmarks, litigious_benchmarks):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor=C_BG)
    fig.suptitle("2. Benchmarks des scores continus", fontsize=16, fontweight="bold", y=0.97)

    uncertainty_name_map = {
        "nicky_topic_uncertainty_proxy": label_with_n("Nicky topics\n(proxy)", uncertainty_benchmarks, "nicky_topic_uncertainty_proxy"),
        "fiqa_aspect_uncertainty_proxy": label_with_n("FiQA aspects\n(proxy)", uncertainty_benchmarks, "fiqa_aspect_uncertainty_proxy"),
    }
    _plot_grouped_bars(
        axes[0],
        uncertainty_benchmarks,
        "roc_auc",
        "f1_best_threshold",
        "ROC-AUC",
        "Best F1",
        "Uncertainty",
        name_map=uncertainty_name_map,
    )

    fundamental_name_map = {
        "nicky_topic_fundamentals": label_with_n("Nicky topics", fundamental_benchmarks, "nicky_topic_fundamentals"),
        "fiqa_aspect_fundamentals": label_with_n("FiQA aspects", fundamental_benchmarks, "fiqa_aspect_fundamentals"),
    }
    _plot_grouped_bars(
        axes[1],
        fundamental_benchmarks,
        "roc_auc",
        "f1_best_threshold",
        "ROC-AUC",
        "Best F1",
        "Fundamental Strength",
        name_map=fundamental_name_map,
    )

    litigious_name_map = {
        "fiqa_aspect_legal_risk": label_with_n("FiQA legal\nrisk", litigious_benchmarks, "fiqa_aspect_legal_risk"),
        "legal_domain_stress": label_with_n("Legal domain\nstress", litigious_benchmarks, "legal_domain_stress"),
    }
    _plot_grouped_bars(
        axes[2],
        litigious_benchmarks,
        "roc_auc",
        "f1_best_threshold",
        "ROC-AUC",
        "Best F1",
        "Litigious",
        name_map=litigious_name_map,
    )

    fig.tight_layout(rect=[0, 0.16, 1, 0.93])
    add_explanation(
        fig,
        "ROC-AUC mesure si le score separe bien les cas positifs et negatifs. Best F1 est la meilleure conversion du score en oui/non. Les benchmarks uncertainty sont des proxies publics, pas des labels d'incertitude natifs.",
        y_pos=0.03,
    )

    path = os.path.join(PLOTS_DIR, "2_score_benchmarks.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def build_output_metrics_summary(
    polarity_benchmarks,
    uncertainty_benchmarks,
    fundamental_benchmarks,
    litigious_benchmarks,
    uncertainty_metrics,
    fundamental_metrics,
    litigious_metrics,
    market_impact_metrics,
):
    rows = [
        {
            "output": "polarity",
            "type": "classification",
            "training": "FinBERT pretrained",
            "benchmark_main": f"PhraseBank direct n={polarity_benchmarks['phrasebank_sentiment']['dataset_size']}",
            "main_score": f"Macro-F1 {polarity_benchmarks['phrasebank_sentiment']['macro_f1']:.3f}",
            "hard_score": f"FiQA Macro-F1 {polarity_benchmarks['fiqa_sentiment']['macro_f1']:.3f}",
            "verdict": "Tres fort sur sentiment clair, moyen sur FiQA",
        },
        {
            "output": "uncertainty",
            "type": "score [0,1]",
            "training": f"{uncertainty_metrics['augmented_dataset_size']} ex.",
            "benchmark_main": f"Nicky proxy n={uncertainty_benchmarks['nicky_topic_uncertainty_proxy']['dataset_size']}",
            "main_score": f"AUC {uncertainty_benchmarks['nicky_topic_uncertainty_proxy']['roc_auc']:.3f}",
            "hard_score": f"FiQA AUC {uncertainty_benchmarks['fiqa_aspect_uncertainty_proxy']['roc_auc']:.3f}",
            "verdict": "Bon signal proxy, pas un label natif",
        },
        {
            "output": "fundamental_strength",
            "type": "score [0,1]",
            "training": f"{fundamental_metrics['dataset_size']} ex.",
            "benchmark_main": f"FiQA aspects n={fundamental_benchmarks['fiqa_aspect_fundamentals']['dataset_size']}",
            "main_score": f"AUC {fundamental_benchmarks['fiqa_aspect_fundamentals']['roc_auc']:.3f}",
            "hard_score": f"R2 train/eval {fundamental_metrics['r2']:.3f}",
            "verdict": "Solide pour detecter les fondamentaux",
        },
        {
            "output": "litigious",
            "type": "score [0,1]",
            "training": f"{litigious_metrics['dataset_size']} ex.",
            "benchmark_main": f"Legal stress n={litigious_benchmarks['legal_domain_stress']['dataset_size']}",
            "main_score": f"AUC {litigious_benchmarks['legal_domain_stress']['roc_auc']:.3f}",
            "hard_score": f"FiQA legal AUC {litigious_benchmarks['fiqa_aspect_legal_risk']['roc_auc']:.3f}",
            "verdict": "Tres bon legal evident, plus faible sur legal subtil",
        },
        {
            "output": "market_impact_label",
            "type": "classification",
            "training": f"{market_impact_metrics['train_size']} train / {market_impact_metrics['test_size']} test",
            "benchmark_main": "FNSPID rendement reel",
            "main_score": f"Macro-F1 {market_impact_metrics['macro_f1']:.3f}",
            "hard_score": f"Accuracy {market_impact_metrics['accuracy']:.3f}",
            "verdict": "Tache tres dure, au-dessus du hasard",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_METRICS_SUMMARY_CSV, index=False, encoding="utf-8")
    return summary_df


def plot_output_metrics_scorecard(summary_df):
    fig, ax = plt.subplots(figsize=(19, 7), facecolor=C_BG)
    fig.suptitle("10. Scorecard des outputs principaux", fontsize=16, fontweight="bold", y=0.97)
    ax.axis("off")

    display_df = summary_df.rename(
        columns={
            "output": "Output",
            "type": "Type",
            "training": "Training / source",
            "benchmark_main": "Benchmark principal",
            "main_score": "Score principal",
            "hard_score": "Score complementaire",
            "verdict": "Lecture rapide",
        }
    )
    wrapped_df = display_df.copy()
    for column in wrapped_df.columns:
        wrapped_df[column] = wrapped_df[column].apply(
            lambda value: "\n".join(textwrap.wrap(str(value), width=24))
        )

    table = ax.table(
        cellText=wrapped_df.values,
        colLabels=wrapped_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.14, 0.10, 0.14, 0.17, 0.12, 0.15, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dee6")
        if row_idx == 0:
            cell.set_facecolor("#1f2d3d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f2f5f8")
            if col_idx == 0:
                cell.set_text_props(weight="bold", color="#1f2d3d")
            if col_idx == 4:
                cell.set_facecolor("#e8f7ef")

    add_explanation(
        fig,
        "Cette scorecard resume les outputs comparables : ce que mesure chaque output, sur quoi il est entraine ou benchmarke, et le score a citer rapidement.",
        y_pos=0.035,
    )

    path = os.path.join(PLOTS_DIR, "10_output_metrics_scorecard.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")
    print(f"  OK {OUTPUT_METRICS_SUMMARY_CSV}")


def plot_continuous_training_metrics(uncertainty_metrics, fundamental_metrics, litigious_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), facecolor=C_BG)
    fig.suptitle("11. Metriques d'entrainement des scores continus", fontsize=16, fontweight="bold", y=0.97)

    labels = ["uncertainty", "fundamental\nstrength", "litigious"]
    metrics_list = [uncertainty_metrics, fundamental_metrics, litigious_metrics]
    colors = [C_PURPLE, C_POS, C_NEG]

    r2_values = [metric["r2"] for metric in metrics_list]
    axes[0].bar(labels, r2_values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("R2 sur evaluation interne")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for idx, value in enumerate(r2_values):
        axes[0].text(idx, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")

    x = np.arange(len(labels))
    width = 0.34
    mae_values = [metric["mae"] for metric in metrics_list]
    rmse_values = [metric["rmse"] for metric in metrics_list]
    axes[1].bar(x - width / 2, mae_values, width, color=C_BLUE, label="MAE", edgecolor="white", linewidth=1.2)
    axes[1].bar(x + width / 2, rmse_values, width, color=C_ORANGE, label="RMSE", edgecolor="white", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Erreur moyenne des scores [0,1]")
    axes[1].set_ylim(0, max(rmse_values) + 0.08)
    axes[1].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="upper right")
    for idx, value in enumerate(mae_values):
        axes[1].text(idx - width / 2, value + 0.012, f"{value:.3f}", ha="center", fontweight="bold", fontsize=10)
    for idx, value in enumerate(rmse_values):
        axes[1].text(idx + width / 2, value + 0.012, f"{value:.3f}", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout(rect=[0, 0.15, 1, 0.93])
    add_explanation(
        fig,
        "R2 plus haut = meilleur. MAE/RMSE plus bas = meilleur. Ces metriques concernent les modeles de score continus, pas la classification finale FNSPID.",
        y_pos=0.035,
    )

    path = os.path.join(PLOTS_DIR, "11_continuous_training_metrics.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def extract_final_model_numeric_weights(market_impact_metrics):
    classifier = load(MARKET_IMPACT_CLASSIFIER_PATH)
    numeric_columns = market_impact_metrics["numeric_classifier_columns"]
    feature_names = list(classifier.named_steps["features"].get_feature_names_out())
    logistic = classifier.named_steps["classifier"]

    rows = []
    for class_index, class_name in enumerate(logistic.classes_):
        coefficients = logistic.coef_[class_index]
        for feature in numeric_columns:
            feature_name = f"numeric__{feature}"
            if feature_name not in feature_names:
                continue
            feature_index = feature_names.index(feature_name)
            coefficient = float(coefficients[feature_index])
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )

    weights_df = pd.DataFrame(rows)
    raw_summary_df = (
        weights_df[weights_df["feature"].isin(RAW_OUTPUTS)]
        .groupby("feature", as_index=False)
        .agg(mean_abs_weight=("abs_coefficient", "mean"))
        .sort_values("mean_abs_weight", ascending=False)
    )
    weights_df.to_csv(FINAL_MODEL_NUMERIC_WEIGHTS_CSV, index=False, encoding="utf-8")
    raw_summary_df.to_csv(BASE_OUTPUT_WEIGHTS_CSV, index=False, encoding="utf-8")
    return weights_df, raw_summary_df


def plot_final_model_output_weights(weights_df, raw_summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=C_BG)
    fig.suptitle("12. Poids des outputs dans le modele final", fontsize=16, fontweight="bold", y=0.97)

    colors = [
        C_POS if feature == "fundamental_strength"
        else C_NEG if feature == "litigious"
        else C_PURPLE if feature == "uncertainty"
        else C_BLUE
        for feature in raw_summary_df["feature"]
    ]
    axes[0].bar(
        raw_summary_df["feature"],
        raw_summary_df["mean_abs_weight"],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    axes[0].set_title("Importance moyenne des outputs de base")
    axes[0].set_ylabel("Coefficient absolu moyen")
    axes[0].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(axis="x", rotation=18)
    for idx, value in enumerate(raw_summary_df["mean_abs_weight"]):
        axes[0].text(idx, value + 0.01, f"{value:.3f}", ha="center", fontweight="bold")

    pivot_df = (
        weights_df[weights_df["feature"].isin(RAW_OUTPUTS)]
        .pivot(index="feature", columns="class", values="coefficient")
        .reindex(RAW_OUTPUTS)
        .fillna(0.0)
    )
    im = axes[1].imshow(pivot_df.values, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(np.arange(len(pivot_df.columns)))
    axes[1].set_xticklabels(pivot_df.columns, fontweight="bold")
    axes[1].set_yticks(np.arange(len(pivot_df.index)))
    axes[1].set_yticklabels(pivot_df.index, fontweight="bold")
    axes[1].set_title("Sens du poids par classe")
    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = pivot_df.iloc[row_idx, col_idx]
            axes[1].text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                color="#1f2d3d",
            )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="coefficient")

    fig.tight_layout(rect=[0, 0.15, 1, 0.93])
    add_explanation(
        fig,
        "Ces poids viennent de la regression logistique finale. Ils sont comparables car les features numeriques sont standardisees. Attention : le modele utilise aussi le texte TF-IDF et le ticker.",
        y_pos=0.035,
    )

    path = os.path.join(PLOTS_DIR, "12_final_model_output_weights.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")
    print(f"  OK {FINAL_MODEL_NUMERIC_WEIGHTS_CSV}")
    print(f"  OK {BASE_OUTPUT_WEIGHTS_CSV}")


def plot_benchmark_guide():
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), facecolor=C_BG)
    fig.suptitle("3. Guide de lecture des benchmarks", fontsize=16, fontweight="bold", y=0.98)

    panels = [
        (
            "PhraseBank",
            "Sentences financieres annotees par experts.\nUsage ici : tester le sentiment classique de polarity.",
        ),
        (
            "FiQA",
            "Headlines / phrases finance plus difficiles.\nUsage ici : tester sentiment, uncertainty proxy, fondamentaux et risque legal plus fins.",
        ),
        (
            "FNSPID Market Returns",
            "News financieres reliees aux prix. Ici, le label vient du rendement reel apres la news : Bullish, Neutral ou Bearish.",
        ),
        (
            "Nicky Topics",
            "Dataset de topics financiers.\nUsage ici : creer des benchmarks proxy. Exemple : Macro / Politics = plus incertain ; Earnings / Financials = plus fondamental.",
        ),
        (
            "Legal Domain Stress",
            "Vrais textes juridiques opposes a des textes non juridiques.\nUsage ici : verifier si litigious reconnait bien un langage legal tres marque.",
        ),
        (
            "Proxy Benchmark",
            "Le label n'est pas exactement la cible du modele, mais une approximation publique utile. C'est surtout le cas pour uncertainty.",
        ),
        (
            "Comment lire",
            "Plus pres de 1.0 = meilleur. Accuracy / Macro-F1 pour les classifications. ROC-AUC / Best F1 pour les scores continus.",
        ),
    ]

    for ax, (title, body) in zip(axes.flatten(), panels):
        ax.axis("off")
        ax.text(
            0.5,
            0.70,
            title,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#1f2d3d",
        )
        ax.text(
            0.5,
            0.34,
            textwrap.fill(body, width=34),
            ha="center",
            va="center",
            fontsize=11,
            color="#334e68",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#eef2f7",
                edgecolor="#bdc3c7",
                alpha=0.95,
            ),
        )
    for ax in axes.flatten()[len(panels):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    path = os.path.join(PLOTS_DIR, "3_benchmark_guide.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_benchmark_coverage(market_impact_metrics):
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=C_BG)
    fig.suptitle("7. Quel benchmark sert a tester quel output ?", fontsize=16, fontweight="bold", y=0.97)

    rows = [
        "Polarity",
        "Uncertainty",
        "Fundamental\nStrength",
        "Litigious",
        "Market Impact\nFinal",
    ]
    cols = [
        "PhraseBank",
        "FiQA",
        "Nicky Topics",
        "Legal Domain",
        "FNSPID\nMarket Returns",
    ]

    matrix = np.array(
        [
            [2, 2, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 0, 2, 0],
            [0, 0, 0, 0, 2],
        ]
    )

    cell_text = [
        ["sent.\n909", "sent.\n405", "", "", ""],
        ["", "proxy\n294", "proxy\n1600", "", ""],
        ["", "aspect\n208", "topic\n2400", "", ""],
        ["", "legal\n82", "", "stress\n2400", ""],
        ["", "", "", "", f"direct\n{market_impact_metrics['test_size']}"],
    ]

    cmap = matplotlib.colors.ListedColormap(["#eef2f7", "#f9c74f", "#2ecc71"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
    ax.set_yticklabels(rows, fontsize=11, fontweight="bold")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                cell_text[row_idx][col_idx],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if matrix[row_idx, col_idx] == 2 else "normal",
                color="#1f2d3d",
            )

    for edge in np.arange(-0.5, len(cols), 1):
        ax.axvline(edge, color="white", linewidth=2)
    for edge in np.arange(-0.5, len(rows), 1):
        ax.axhline(edge, color="white", linewidth=2)

    ax.set_title("Vert = benchmark direct, jaune = benchmark proxy, gris = non utilise", pad=16)
    ax.tick_params(top=False, bottom=True, left=True, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    add_explanation(
        fig,
        "Ce tableau dit explicitement a quoi correspond chaque benchmark. Exemple : Polarity est teste sur PhraseBank et FiQA, alors que Market Impact est teste sur FNSPID avec un label derive du mouvement reel du titre.",
    )

    path = os.path.join(PLOTS_DIR, "7_benchmark_coverage.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_output_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
    fig.suptitle("4. Vue d'ensemble des 67 articles", fontsize=16, fontweight="bold", y=0.97)

    counts = df["market_impact_label"].value_counts().reindex(DISPLAY_LABEL_ORDER).fillna(0)
    colors = [LABEL_COLORS[label] for label in DISPLAY_LABEL_ORDER]
    axes[0].bar(DISPLAY_LABEL_ORDER, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Repartition des classes finales Market Impact")
    axes[0].set_ylabel("Nombre d'articles")
    axes[0].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for idx, value in enumerate(counts.values):
        axes[0].text(idx, value + 0.8, str(int(value)), ha="center", fontweight="bold")

    score_names = ["uncertainty", "fundamental_strength", "litigious"]
    display_names = ["Uncertainty", "Fundamental\nStrength", "Litigious"]
    mean_scores = [float(df[column].mean()) for column in score_names]
    std_scores = [float(df[column].std()) for column in score_names]
    axes[1].bar(display_names, mean_scores, yerr=std_scores, capsize=8, color=[C_PURPLE, C_POS, C_NEG], edgecolor="white", linewidth=1.2)
    axes[1].set_title("Moyenne des scores sur les 67 articles")
    axes[1].set_ylim(0, max(mean_scores[i] + std_scores[i] for i in range(len(mean_scores))) + 0.12)
    axes[1].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for idx, value in enumerate(mean_scores):
        axes[1].text(idx, value + std_scores[idx] + 0.02, f"{value:.3f}", ha="center", fontweight="bold")

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    add_explanation(
        fig,
        "A gauche : combien d'articles tombent dans chaque classe Bearish / Neutral / Bullish. A droite : niveau moyen des scores sur le corpus. Les barres noires montrent la dispersion des scores.",
    )

    path = os.path.join(PLOTS_DIR, "4_output_overview.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_final_class_profiles(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=C_BG)
    fig.suptitle("5. Profils moyens des classes Market Impact", fontsize=16, fontweight="bold", y=0.98)

    profile_df = (
        df.groupby("market_impact_label")
        .agg(
            support=("market_impact_label", "size"),
            mean_polarity=("polarity", "mean"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_fundamental_impact=("fundamental_impact", "mean"),
            mean_risk_pressure=("risk_pressure", "mean"),
            mean_market_signal=("market_signal_score", "mean"),
        )
        .reindex(DISPLAY_LABEL_ORDER)
        .reset_index()
    )

    metrics = [
        ("support", "Nombre d'articles", True),
        ("mean_polarity", "Polarite moyenne", False),
        ("mean_uncertainty", "Uncertainty moyenne", False),
        ("mean_fundamental_impact", "Impact fondamental moyen", False),
        ("mean_risk_pressure", "Pression de risque moyenne", False),
        ("mean_market_signal", "Score signal marche moyen", False),
    ]

    for ax, (column, title, is_count) in zip(axes.flatten(), metrics):
        colors = [LABEL_COLORS[label] for label in profile_df["market_impact_label"]]
        values = profile_df[column].tolist()
        ax.bar(profile_df["market_impact_label"], values, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(title)
        ax.grid(axis="y", color=C_GRID, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if is_count:
            ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1.0)
            for idx, value in enumerate(values):
                ax.text(idx, value + 0.8, str(int(value)), ha="center", fontweight="bold")
        else:
            min_value = min(values)
            max_value = max(values)
            spread = max(max_value - min_value, 0.12)
            ax.set_ylim(min(0.0, min_value - spread * 0.18), max_value + spread * 0.22)
            for idx, value in enumerate(values):
                y_pos = value + (spread * 0.06 if value >= 0 else -spread * 0.08)
                ax.text(
                    idx,
                    y_pos,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontweight="bold",
                )

    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    add_explanation(
        fig,
        "Ce graphe montre directement les 67 articles classes. Bullish doit idealement etre plus positif, avec un impact fondamental plus haut et une pression de risque plus basse que Bearish.",
    )

    path = os.path.join(PLOTS_DIR, "5_final_class_profiles.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_market_impact_decision_map(df):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C_BG)
    fig.suptitle("6. Projection 2D simplifiee des 67 articles", fontsize=16, fontweight="bold", y=0.97)

    for label in DISPLAY_LABEL_ORDER:
        subset = df[df["market_impact_label"] == label]
        ax.scatter(
            subset["fundamental_impact"],
            subset["risk_pressure"],
            s=70 + (subset["assigned_class_probability"].fillna(0.0) * 70),
            alpha=0.8,
            c=LABEL_COLORS[label],
            edgecolors="white",
            linewidth=0.8,
            label=label,
        )

    ax.axvline(df["fundamental_impact"].mean(), color=C_BLUE, linestyle="--", linewidth=1.5)
    ax.axhline(df["risk_pressure"].mean(), color=C_PURPLE, linestyle="--", linewidth=1.5)
    ax.text(0.69, 0.08, "Impact positif / risque modere", transform=ax.transAxes, color=C_POS, fontsize=10)
    ax.text(0.06, 0.88, "Impact negatif / risque eleve", transform=ax.transAxes, color=C_NEG, fontsize=10)
    ax.set_xlabel("Impact fondamental signe")
    ax.set_ylabel("Pression de risque")
    ax.legend(loc="upper right")
    ax.grid(color=C_GRID, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    add_explanation(
        fig,
        "Projection simplifiee : chaque point est un article projete sur seulement 2 axes. Le modele final utilise aussi le texte TF-IDF, le ticker et toutes les features numeriques ; ce graphe n'est donc pas toute la decision.",
    )

    path = os.path.join(PLOTS_DIR, "6_decision_map.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_prediction_confidence(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
    fig.suptitle("9. Confiance des predictions sur les 67 articles", fontsize=16, fontweight="bold", y=0.97)

    confidence_data = [
        df.loc[df["market_impact_label"] == label, "assigned_class_probability"]
        .dropna()
        .values
        for label in DISPLAY_LABEL_ORDER
    ]
    box = axes[0].boxplot(
        confidence_data,
        tick_labels=DISPLAY_LABEL_ORDER,
        patch_artist=True,
        showmeans=True,
    )
    for patch, label in zip(box["boxes"], DISPLAY_LABEL_ORDER):
        patch.set_facecolor(LABEL_COLORS[label])
        patch.set_alpha(0.72)
    axes[0].set_title("Confiance dans la classe retenue")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Probabilite de la classe choisie")
    axes[0].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    probability_means = (
        df.groupby("market_impact_label")[["prob_bullish", "prob_neutral", "prob_bearish"]]
        .mean()
        .reindex(DISPLAY_LABEL_ORDER)
        .fillna(0.0)
    )
    bottom = np.zeros(len(DISPLAY_LABEL_ORDER))
    probability_columns = [
        ("prob_bullish", "Bullish", C_POS),
        ("prob_neutral", "Neutral", C_ORANGE),
        ("prob_bearish", "Bearish", C_NEG),
    ]
    x = np.arange(len(DISPLAY_LABEL_ORDER))
    for column, label, color in probability_columns:
        values = probability_means[column].values
        axes[1].bar(
            x,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            label=label,
        )
        bottom += values
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DISPLAY_LABEL_ORDER)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Probabilites moyennes par classe predite")
    axes[1].set_ylabel("Probabilite moyenne")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", color=C_GRID, alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    add_explanation(
        fig,
        "A gauche : si les boites sont basses, le modele hesite. A droite : on voit si une classe predite est vraiment dominee par sa probabilite ou si les trois probabilites restent proches.",
    )

    path = os.path.join(PLOTS_DIR, "9_prediction_confidence.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def main():
    print("Generation des visualisations essentielles...")
    cleanup_existing_plots()

    df = load_article_level_data()
    market_impact_metrics = load_json(MARKET_IMPACT_METRICS_PATH)
    uncertainty_metrics = load_json(UNCERTAINTY_METRICS_PATH)
    fundamental_metrics = load_json(FUNDAMENTAL_METRICS_PATH)
    litigious_metrics = load_json(LITIGIOUS_METRICS_PATH)
    fundamental_benchmarks = load_json(FUNDAMENTAL_BENCHMARKS_PATH)
    litigious_benchmarks = load_json(LITIGIOUS_BENCHMARKS_PATH)
    polarity_benchmarks = load_json(POLARITY_BENCHMARKS_PATH)
    uncertainty_benchmarks = load_json(UNCERTAINTY_BENCHMARKS_PATH)
    summary_df = build_output_metrics_summary(
        polarity_benchmarks,
        uncertainty_benchmarks,
        fundamental_benchmarks,
        litigious_benchmarks,
        uncertainty_metrics,
        fundamental_metrics,
        litigious_metrics,
        market_impact_metrics,
    )
    weights_df, raw_summary_df = extract_final_model_numeric_weights(market_impact_metrics)

    plot_classifier_benchmarks(polarity_benchmarks, market_impact_metrics)
    plot_fnspid_confusion_matrix(market_impact_metrics)
    plot_score_benchmarks(uncertainty_benchmarks, fundamental_benchmarks, litigious_benchmarks)
    plot_output_metrics_scorecard(summary_df)
    plot_continuous_training_metrics(uncertainty_metrics, fundamental_metrics, litigious_metrics)
    plot_final_model_output_weights(weights_df, raw_summary_df)
    plot_benchmark_guide()
    plot_output_overview(df)
    plot_final_class_profiles(df)
    plot_market_impact_decision_map(df)
    plot_prediction_confidence(df)
    plot_benchmark_coverage(market_impact_metrics)
    print("Visualisations terminees.")


if __name__ == "__main__":
    main()
