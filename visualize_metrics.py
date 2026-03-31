"""
Visualisation des résultats d'analyse — Pipeline News Agents
=============================================================
Génère 6 graphiques clairs avec explications détaillées :

1. Répartition des Sentiments (Polarité)
2. Distribution des Scores d'Incertitude
3. Scatter Plot : Incertitude vs Confiance de Polarité
4. Top 10 Articles les plus Incertains vs les plus Factuels
5. Incertitude Moyenne par Sentiment
6. Matrice Sentiment × Incertitude (Heatmap)

Usage :
    python visualize_metrics.py
"""

import os
import sys
import sqlite3
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "metrics_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Couleurs
C_POS = '#2ecc71'      # Vert  — Positif
C_NEU = '#95a5a6'      # Gris  — Neutre
C_NEG = '#e74c3c'      # Rouge — Négatif
C_ACCENT = '#3498db'   # Bleu
C_ORANGE = '#f39c12'   # Orange
C_BG = '#fafbfc'       # Fond clair


def load_data():
    """Charge les données depuis la base SQLite."""
    db_path = os.path.join(OUTPUT_DIR, "news_database.db")
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT a.title, a.ticker, s.polarity, s.polarity_conf, s.uncertainty
        FROM article_scores s
        JOIN articles a ON s.url = a.url
    """
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    data = {
        'titles': [r[0] for r in rows],
        'tickers': [r[1] for r in rows],
        'polarity': [r[2] for r in rows],
        'confidence': [r[3] for r in rows],
        'uncertainty': [r[4] for r in rows],
    }
    return data


def add_explanation(fig, text, y_pos=0.02):
    """Ajoute une zone d'explication sous le graphique."""
    fig.text(0.5, y_pos, text, ha='center', va='bottom', fontsize=9.5,
             color='#444444', style='italic', wrap=True,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#eef2f7',
                      edgecolor='#bdc3c7', alpha=0.9),
             transform=fig.transFigure)


# ═══════════════════════════════════════
# GRAPHIQUE 1 : Répartition des Sentiments
# ═══════════════════════════════════════
def plot_sentiment_distribution(data):
    """Camembert + barres de la répartition Positif / Neutre / Négatif."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
    fig.suptitle('📊  Répartition des Sentiments (Polarité)', fontsize=16, fontweight='bold', y=0.97)

    pols = data['polarity']
    n_neg = pols.count(-1)
    n_neu = pols.count(0)
    n_pos = pols.count(1)
    total = len(pols)

    # Camembert
    sizes = [n_neg, n_neu, n_pos]
    labels = [f'Négatif\n({n_neg})', f'Neutre\n({n_neu})', f'Positif\n({n_pos})']
    colors = [C_NEG, C_NEU, C_POS]
    explode = (0.03, 0.03, 0.03)

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, explode=explode, colors=colors,
        autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12},
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontweight('bold')
        t.set_color('white')
    ax1.set_title(f'Répartition sur {total} articles', fontsize=12, pad=10)

    # Barres horizontales avec confiance moyenne
    categories = ['Négatif (-1)', 'Neutre (0)', 'Positif (+1)']
    counts = [n_neg, n_neu, n_pos]
    
    # Confiance moyenne par catégorie
    neg_conf = np.mean([c for p, c in zip(pols, data['confidence']) if p == -1]) if n_neg > 0 else 0
    neu_conf = np.mean([c for p, c in zip(pols, data['confidence']) if p == 0]) if n_neu > 0 else 0
    pos_conf = np.mean([c for p, c in zip(pols, data['confidence']) if p == 1]) if n_pos > 0 else 0
    confs = [neg_conf, neu_conf, pos_conf]
    
    bars = ax2.barh(categories, counts, color=colors, edgecolor='white', linewidth=1.5, height=0.6)
    for bar, count, conf in zip(bars, counts, confs):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}  (conf. moy: {conf:.0%})',
                va='center', fontsize=11, fontweight='bold', color='#333')
    
    ax2.set_xlabel('Nombre d\'articles', fontsize=11)
    ax2.set_title('Détail par catégorie', fontsize=12, pad=10)
    ax2.set_xlim(0, max(counts) * 1.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    add_explanation(fig,
        "💡 Ce graphique montre comment FinBERT a classé chaque article. Un bon modèle produit une répartition équilibrée,\n"
        "pas tout positif ou tout négatif. La confiance moyenne indique à quel point le modèle est sûr de ses classifications."
    )
    
    path = os.path.join(PLOTS_DIR, "1_sentiment_distribution.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ═══════════════════════════════════════
# GRAPHIQUE 2 : Distribution de l'Incertitude
# ═══════════════════════════════════════
def plot_uncertainty_distribution(data):
    """Histogramme de la distribution des scores d'incertitude."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C_BG)
    fig.suptitle('📈  Distribution des Scores d\'Incertitude', fontsize=16, fontweight='bold', y=0.97)

    scores = data['uncertainty']
    
    # Histogramme
    bins = np.linspace(0, max(scores) + 0.05, 20)
    n, bins_out, patches = ax.hist(scores, bins=bins, edgecolor='white', linewidth=1.2, alpha=0.85)
    
    # Colorer les barres selon le niveau
    for patch, left_edge in zip(patches, bins_out[:-1]):
        if left_edge < 0.10:
            patch.set_facecolor(C_POS)     # Vert = factuel
        elif left_edge < 0.20:
            patch.set_facecolor(C_ACCENT)  # Bleu = légèrement incertain
        elif left_edge < 0.30:
            patch.set_facecolor(C_ORANGE)  # Orange = modéré
        else:
            patch.set_facecolor(C_NEG)     # Rouge = incertain

    # Lignes de référence
    mean_s = np.mean(scores)
    median_s = np.median(scores)
    ax.axvline(mean_s, color='#2c3e50', linestyle='--', linewidth=2, label=f'Moyenne : {mean_s:.3f}')
    ax.axvline(median_s, color='#8e44ad', linestyle=':', linewidth=2, label=f'Médiane : {median_s:.3f}')

    # Légende des zones
    legend_patches = [
        mpatches.Patch(color=C_POS, label='🟢 Factuel (< 0.10)'),
        mpatches.Patch(color=C_ACCENT, label='🔵 Léger (0.10 - 0.20)'),
        mpatches.Patch(color=C_ORANGE, label='🟠 Modéré (0.20 - 0.30)'),
        mpatches.Patch(color=C_NEG, label='🔴 Incertain (> 0.30)'),
    ]
    
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0],
             loc='upper right', fontsize=9, framealpha=0.9)
    
    ax.set_xlabel('Score d\'incertitude', fontsize=12)
    ax.set_ylabel('Nombre d\'articles', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Stats en annotation
    stats_text = (f"n = {len(scores)} articles\n"
                  f"min = {min(scores):.4f}\n"
                  f"max = {max(scores):.4f}\n"
                  f"écart-type = {np.std(scores):.4f}")
    ax.text(0.98, 0.65, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    add_explanation(fig,
        "💡 Ce graphique montre la distribution des scores d'incertitude prédits par notre modèle fine-tuné.\n"
        "Un score proche de 0 = article factuel et sûr. Un score élevé = article spéculatif, plein de doutes.\n"
        "La plupart des articles ont un score bas → les news financières sont majoritairement factuelles."
    )

    path = os.path.join(PLOTS_DIR, "2_uncertainty_distribution.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ═══════════════════════════════════════
# GRAPHIQUE 3 : Scatter Incertitude vs Confiance
# ═══════════════════════════════════════
def plot_scatter_uncertainty_vs_confidence(data):
    """Nuage de points : chaque article positionné par (confiance, incertitude)."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C_BG)
    fig.suptitle('🔍  Incertitude vs Confiance de Polarité', fontsize=16, fontweight='bold', y=0.97)

    colors_map = {-1: C_NEG, 0: C_NEU, 1: C_POS}
    labels_map = {-1: 'Négatif', 0: 'Neutre', 1: 'Positif'}

    for pol_val in [-1, 0, 1]:
        mask = [i for i, p in enumerate(data['polarity']) if p == pol_val]
        x = [data['confidence'][i] for i in mask]
        y = [data['uncertainty'][i] for i in mask]
        ax.scatter(x, y, c=colors_map[pol_val], label=labels_map[pol_val],
                  s=80, alpha=0.7, edgecolors='white', linewidth=0.8, zorder=5)

    # Zones annotées
    ax.axhspan(0.25, max(data['uncertainty']) + 0.05, alpha=0.05, color=C_NEG)
    ax.axhspan(0, 0.10, alpha=0.05, color=C_POS)
    
    ax.text(0.52, max(data['uncertainty']) - 0.01, '⚠️ Zone d\'incertitude haute',
            fontsize=9, color=C_NEG, alpha=0.7)
    ax.text(0.52, 0.02, '✅ Zone factuelle',
            fontsize=9, color=C_POS, alpha=0.7)

    ax.set_xlabel('Confiance du modèle de polarité', fontsize=12)
    ax.set_ylabel('Score d\'incertitude', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.45, 1.02)

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    add_explanation(fig,
        "💡 Chaque point = un article. Axe X = confiance de FinBERT dans sa classification sentiment.\n"
        "Axe Y = score d'incertitude de notre modèle custom. Idéalement, un article factuel (incertitude basse)\n"
        "devrait aussi avoir une confiance élevée → les points en bas à droite sont les articles les plus \"sûrs\"."
    )

    path = os.path.join(PLOTS_DIR, "3_scatter_uncertainty_confidence.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ═══════════════════════════════════════
# GRAPHIQUE 4 : Top articles (plus/moins incertains)
# ═══════════════════════════════════════
def plot_top_articles(data):
    """Barres horizontales : Top 8 incertains vs Top 8 factuels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_BG)
    fig.suptitle('📰  Articles les Plus Incertains vs les Plus Factuels', fontsize=16, fontweight='bold', y=0.97)

    # Trier par incertitude
    indices = list(range(len(data['titles'])))
    indices_sorted = sorted(indices, key=lambda i: data['uncertainty'][i])

    def truncate(s, n=45):
        return s[:n] + '...' if len(s) > n else s

    # Top 8 les plus factuels (incertitude la plus basse)
    top_factual = indices_sorted[:8]
    titles_f = [truncate(data['titles'][i]) for i in top_factual]
    scores_f = [data['uncertainty'][i] for i in top_factual]
    
    bars1 = ax1.barh(range(len(titles_f)), scores_f, color=C_POS, edgecolor='white',
                     linewidth=1, height=0.65)
    ax1.set_yticks(range(len(titles_f)))
    ax1.set_yticklabels(titles_f, fontsize=9)
    ax1.invert_yaxis()
    for bar, score in zip(bars1, scores_f):
        ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10, fontweight='bold', color='#27ae60')
    ax1.set_xlabel('Score d\'incertitude', fontsize=11)
    ax1.set_title('🟢 Les 8 Plus Factuels', fontsize=13, fontweight='bold', pad=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, max(data['uncertainty']) * 1.3)

    # Top 8 les plus incertains
    top_uncertain = indices_sorted[-8:][::-1]
    titles_u = [truncate(data['titles'][i]) for i in top_uncertain]
    scores_u = [data['uncertainty'][i] for i in top_uncertain]
    
    bars2 = ax2.barh(range(len(titles_u)), scores_u, color=C_NEG, edgecolor='white',
                     linewidth=1, height=0.65)
    ax2.set_yticks(range(len(titles_u)))
    ax2.set_yticklabels(titles_u, fontsize=9)
    ax2.invert_yaxis()
    for bar, score in zip(bars2, scores_u):
        ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10, fontweight='bold', color='#c0392b')
    ax2.set_xlabel('Score d\'incertitude', fontsize=11)
    ax2.set_title('🔴 Les 8 Plus Incertains', fontsize=13, fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, max(data['uncertainty']) * 1.3)

    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    add_explanation(fig,
        "💡 À gauche : les articles les plus factuels (score bas = texte sûr, chiffré). "
        "À droite : les articles les plus incertains (spéculatifs, conditionnels).\n"
        "On voit que les annonces officielles (lancements, résultats) sont factuelles, "
        "tandis que les analyses d'opinion et prédictions sont incertaines."
    )

    path = os.path.join(PLOTS_DIR, "4_top_articles.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ═══════════════════════════════════════
# GRAPHIQUE 5 : Incertitude par Sentiment
# ═══════════════════════════════════════
def plot_uncertainty_by_sentiment(data):
    """Box plots + strip plots de l'incertitude par catégorie de sentiment."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C_BG)
    fig.suptitle('📦  Incertitude Moyenne par Catégorie de Sentiment', fontsize=16, fontweight='bold', y=0.97)

    # Grouper les scores par sentiment
    groups = {-1: [], 0: [], 1: []}
    for p, u in zip(data['polarity'], data['uncertainty']):
        groups[p].append(u)

    positions = [0, 1, 2]
    colors_list = [C_NEG, C_NEU, C_POS]
    labels = ['Négatif (-1)', 'Neutre (0)', 'Positif (+1)']
    group_data = [groups[-1], groups[0], groups[1]]

    # Box plots
    bp = ax.boxplot(group_data, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color='black', linewidth=2),
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#7f8c8d'),
                    capprops=dict(color='#7f8c8d'))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Points individuels (strip plot)
    for i, (grp, color) in enumerate(zip(group_data, colors_list)):
        jitter = np.random.normal(0, 0.05, len(grp))
        ax.scatter([i + j for j in jitter], grp, c=color, alpha=0.6, s=40,
                  edgecolors='white', linewidth=0.5, zorder=5)

    # Moyennes en texte
    for i, grp in enumerate(group_data):
        if grp:
            mean_val = np.mean(grp)
            ax.text(i, max(data['uncertainty']) + 0.015, f'μ = {mean_val:.3f}',
                   ha='center', fontsize=11, fontweight='bold', color='#2c3e50')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Score d\'incertitude', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    add_explanation(fig,
        "💡 Ce graphique montre si les articles négatifs sont plus incertains que les positifs.\n"
        "Chaque point = un article. La boîte montre la médiane et les quartiles.\n"
        "Si les articles négatifs ont un score d'incertitude plus élevé, cela signifie que les mauvaises nouvelles\n"
        "sont souvent plus spéculatives, tandis que les bonnes nouvelles sont plus factuelles."
    )

    path = os.path.join(PLOTS_DIR, "5_uncertainty_by_sentiment.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ═══════════════════════════════════════
# GRAPHIQUE 6 : Heatmap Sentiment × Incertitude
# ═══════════════════════════════════════
def plot_heatmap(data):
    """Matrice qui croise sentiment et niveau d'incertitude."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C_BG)
    fig.suptitle('🗺️  Matrice Sentiment × Niveau d\'Incertitude', fontsize=16, fontweight='bold', y=0.97)

    # Bins d'incertitude
    unc_bins = ['Factuel\n(0 - 0.15)', 'Faible\n(0.15 - 0.30)', 'Modéré\n(0.30 - 0.45)',
                'Notable\n(0.45 - 0.60)', 'Élevé\n(0.60 - 0.75)', 'Très élevé\n(> 0.75)']
    unc_thresholds = [0, 0.15, 0.30, 0.45, 0.60, 0.75, float('inf')]
    
    pol_labels = ['Négatif', 'Neutre', 'Positif']
    pol_values = [-1, 0, 1]

    # Compter
    matrix = np.zeros((3, len(unc_bins)), dtype=int)
    for p, u in zip(data['polarity'], data['uncertainty']):
        row = pol_values.index(p)
        for col in range(len(unc_thresholds) - 1):
            if unc_thresholds[col] <= u < unc_thresholds[col + 1]:
                matrix[row, col] += 1
                break

    # Heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Annotations dans chaque cellule
    for i in range(3):
        for j in range(len(unc_bins)):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else '#333'
            ax.text(j, i, str(val), ha='center', va='center',
                   fontsize=14, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(unc_bins)))
    ax.set_xticklabels(unc_bins, fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(pol_labels, fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Nombre d\'articles', fontsize=10)

    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    add_explanation(fig,
        "💡 Cette matrice croise les deux dimensions : sentiment (lignes) et niveau d'incertitude (colonnes).\n"
        "Chaque case indique le nombre d'articles dans cette combinaison.\n"
        "Les cases les plus foncées montrent où se concentrent la majorité des articles analysés."
    )

    path = os.path.join(PLOTS_DIR, "6_heatmap_sentiment_uncertainty.png")
    fig.savefig(path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return path


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    print("📊 Génération des graphiques de visualisation...")
    print()
    
    data = load_data()
    print(f"  📰 {len(data['titles'])} articles chargés depuis la base de données")
    print()

    paths = []
    paths.append(plot_sentiment_distribution(data))
    paths.append(plot_uncertainty_distribution(data))
    paths.append(plot_scatter_uncertainty_vs_confidence(data))
    paths.append(plot_top_articles(data))
    paths.append(plot_uncertainty_by_sentiment(data))
    paths.append(plot_heatmap(data))

    print()
    print(f"✅ {len(paths)} graphiques générés dans : {PLOTS_DIR}/")
    print("   Ouvre les fichiers PNG pour les visualiser !")
    return paths


if __name__ == "__main__":
    main()
