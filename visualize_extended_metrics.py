"""
Visualisations essentielles des metriques
========================================

Genere uniquement les graphes utiles pour comprendre :
- les benchmarks des modeles
- les profils finaux des classes
- la carte de decision sur les 67 articles
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


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "metrics_plots")
DB_PATH = os.path.join(OUTPUT_DIR, "news_database.db")

INVESTMENT_METRICS_PATH = os.path.join(OUTPUT_DIR, "investment_model", "metrics.json")
FUNDAMENTAL_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "fundamental_strength_model", "benchmarks.json")
LITIGIOUS_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "litigious_model", "benchmarks.json")
POLARITY_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results", "polarity_benchmarks.json")
UNCERTAINTY_BENCHMARKS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results", "uncertainty_benchmarks.json")

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
    "Good Investment": C_POS,
    "Watchlist": C_ORANGE,
    "Do Not Invest": C_NEG,
}


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
            ir.investment_label,
            ir.investment_profile_score,
            ir.prob_do_not_invest,
            ir.prob_watchlist,
            ir.prob_good_investment
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        LEFT JOIN investment_recommendations ir ON s.url = ir.url
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["risk_adjusted_sentiment"] = df["polarity"] * df["polarity_conf"]
    df["headline_conviction"] = df["polarity_conf"] * (1.0 - df["uncertainty"])
    df["fundamental_impact"] = df["fundamental_strength"] * df["risk_adjusted_sentiment"]
    df["risk_pressure"] = (0.55 * df["uncertainty"]) + (0.45 * df["litigious"])
    df["assigned_class_probability"] = np.select(
        [
            df["investment_label"] == "Good Investment",
            df["investment_label"] == "Watchlist",
            df["investment_label"] == "Do Not Invest",
        ],
        [
            df["prob_good_investment"],
            df["prob_watchlist"],
            df["prob_do_not_invest"],
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


def plot_classifier_benchmarks(polarity_benchmarks, investment_metrics):
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

    investment_benchmark = {
        "external_supervised_test": {
            "accuracy": investment_metrics["accuracy"],
            "macro_f1": investment_metrics["macro_f1"],
        }
    }
    _plot_grouped_bars(
        axes[1],
        investment_benchmark,
        "accuracy",
        "macro_f1",
        "Accuracy",
        "Macro-F1",
        "Investissement final",
        name_map={"external_supervised_test": f"Test externe\n(n={investment_metrics['test_size']})"},
    )

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    add_explanation(
        fig,
        "Ici, plus les barres sont hautes, mieux le modele classe correctement les textes. Le benchmark investissement est evalue sur un jeu de test externe, pas sur les 67 articles.",
    )

    path = os.path.join(PLOTS_DIR, "1_classifier_benchmarks.png")
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


def plot_benchmark_guide():
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), facecolor=C_BG)
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

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    path = os.path.join(PLOTS_DIR, "3_benchmark_guide.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_benchmark_coverage():
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=C_BG)
    fig.suptitle("7. Quel benchmark sert a tester quel output ?", fontsize=16, fontweight="bold", y=0.97)

    rows = [
        "Polarity",
        "Uncertainty",
        "Fundamental\nStrength",
        "Litigious",
        "Investment\nFinal",
    ]
    cols = [
        "PhraseBank",
        "FiQA",
        "Nicky Topics",
        "Legal Domain",
        "External Invest\nTest",
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
        ["", "", "", "", "test\n669"],
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
        "Ce tableau dit explicitement a quoi correspond chaque benchmark. Exemple : Polarity est teste sur PhraseBank et FiQA, alors que Uncertainty n'a que des proxies publics. Le nombre dans chaque case est la taille du test utilise.",
    )

    path = os.path.join(PLOTS_DIR, "7_benchmark_coverage.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_output_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
    fig.suptitle("4. Vue d'ensemble des 67 articles", fontsize=16, fontweight="bold", y=0.97)

    label_order = ["Good Investment", "Watchlist", "Do Not Invest"]
    counts = df["investment_label"].value_counts().reindex(label_order).fillna(0)
    colors = [LABEL_COLORS[label] for label in label_order]
    axes[0].bar(label_order, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Repartition des classes finales")
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
        "A gauche : combien d'articles tombent dans chaque decision finale. A droite : niveau moyen des scores sur le corpus. Les barres noires montrent la dispersion des scores.",
    )

    path = os.path.join(PLOTS_DIR, "4_output_overview.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_final_class_profiles(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=C_BG)
    fig.suptitle("5. Profils moyens des classes finales", fontsize=16, fontweight="bold", y=0.98)

    label_order = ["Good Investment", "Watchlist", "Do Not Invest"]
    profile_df = (
        df.groupby("investment_label")
        .agg(
            support=("investment_label", "size"),
            mean_polarity=("polarity", "mean"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_fundamental_impact=("fundamental_impact", "mean"),
            mean_risk_pressure=("risk_pressure", "mean"),
            mean_investment_score=("investment_profile_score", "mean"),
        )
        .reindex(label_order)
        .reset_index()
    )

    metrics = [
        ("support", "Nombre d'articles", True),
        ("mean_polarity", "Polarite moyenne", False),
        ("mean_uncertainty", "Uncertainty moyenne", False),
        ("mean_fundamental_impact", "Impact fondamental moyen", False),
        ("mean_risk_pressure", "Pression de risque moyenne", False),
        ("mean_investment_score", "Score d'investissement moyen", False),
    ]

    for ax, (column, title, is_count) in zip(axes.flatten(), metrics):
        colors = [LABEL_COLORS[label] for label in profile_df["investment_label"]]
        values = profile_df[column].tolist()
        ax.bar(profile_df["investment_label"], values, color=colors, edgecolor="white", linewidth=1.2)
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
        "Ce graphe montre directement les 67 articles classes. Good Investment doit idealement etre plus positif, avec un impact fondamental plus haut et une pression de risque plus basse que Do Not Invest.",
    )

    path = os.path.join(PLOTS_DIR, "5_final_class_profiles.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def plot_investment_decision_map(df):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C_BG)
    fig.suptitle("6. Carte de decision sur les 67 articles", fontsize=16, fontweight="bold", y=0.97)

    label_order = ["Good Investment", "Watchlist", "Do Not Invest"]
    for label in label_order:
        subset = df[df["investment_label"] == label]
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
        "Chaque point est un article. Plus a droite = les fondamentaux soutiennent la these d'investissement. Plus en haut = plus de risque. La taille du point represente la confiance dans la classe retenue.",
    )

    path = os.path.join(PLOTS_DIR, "6_decision_map.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  OK {path}")


def main():
    print("Generation des visualisations essentielles...")
    cleanup_existing_plots()

    df = load_article_level_data()
    investment_metrics = load_json(INVESTMENT_METRICS_PATH)
    fundamental_benchmarks = load_json(FUNDAMENTAL_BENCHMARKS_PATH)
    litigious_benchmarks = load_json(LITIGIOUS_BENCHMARKS_PATH)
    polarity_benchmarks = load_json(POLARITY_BENCHMARKS_PATH)
    uncertainty_benchmarks = load_json(UNCERTAINTY_BENCHMARKS_PATH)

    plot_classifier_benchmarks(polarity_benchmarks, investment_metrics)
    plot_score_benchmarks(uncertainty_benchmarks, fundamental_benchmarks, litigious_benchmarks)
    plot_benchmark_guide()
    plot_output_overview(df)
    plot_final_class_profiles(df)
    plot_investment_decision_map(df)
    plot_benchmark_coverage()
    print("Visualisations terminees.")


if __name__ == "__main__":
    main()
