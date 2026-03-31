"""
Comparateur Modèle vs Humain
============================
Ce script lit la base de données SQLite pour récupérer les scores 
d'incertitude originaux du modèle et les comparer avec les scores
révisés (via l'interface HITL) par l'humain.

Il génère un graphique (scatter plot) pour visualiser l'alignement
et les corrections apportées.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = "news_database.db"
OUTPUT_DIR = "metrics_plots"

# Theme sombre pour correspondre aux autres
C_BG = "#0f0f1a"
C_TEXT = "#e8e8f0"
C_TEXT_DIM = "#9090a8"

def style_plot(fig, ax):
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_TEXT_DIM)
    for spine in ax.spines.values():
        spine.set_color(C_TEXT_DIM)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)

def generate_comparison_plot():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Connexion et extraction
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT s.uncertainty AS model_score, hr.human_score, hr.human_status
        FROM article_scores s
        JOIN human_reviews hr ON s.url = hr.url
        WHERE hr.human_status IN ('approved', 'modified')
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"⚠️ Erreur lors de la lecture des données (la table human_reviews est-elle vide ?) : {e}")
        conn.close()
        return
    conn.close()

    if len(df) == 0:
        print("ℹ️ Aucun article n'a encore été 'approuvé' ou 'modifié' via l'interface HITL.")
        print("ℹ️ Fais quelques révisions dans http://localhost:5555 d'abord !")
        return

    print(f"📊 Génération du graphique de comparaison pour {len(df)} articles révisés...")

    # 2. Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    style_plot(fig, ax)

    # Ligne idéale (Modèle == Humain)
    ax.plot([0, 1], [0, 1], color=C_TEXT_DIM, linestyle='--', alpha=0.5, zorder=1, label="Alignement parfait")

    # Points "Approuvés"
    approved = df[df['human_status'] == 'approved']
    if len(approved) > 0:
        ax.scatter(approved['model_score'], approved['human_score'], 
                   color='#00d4aa', s=80, alpha=0.7, zorder=3, label=f"Approuvés ({len(approved)})")

    # Points "Modifiés"
    modified = df[df['human_status'] == 'modified']
    if len(modified) > 0:
        ax.scatter(modified['model_score'], modified['human_score'], 
                   color='#ffa726', s=80, alpha=0.9, zorder=4, edgecolor='white', label=f"Modifiés ({len(modified)})")
        
        # Dessiner des flèches pour montrer la correction
        for _, row in modified.iterrows():
            ax.annotate('', xy=(row['model_score'], row['human_score']), 
                        xytext=(row['model_score'], row['model_score']),
                        arrowprops=dict(arrowstyle="->", color='#ffa726', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Score du Modèle d'Incertitude", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Score Humain (Ground Truth)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("🤖 Modèle vs 🧠 Humain : Évaluation des Scores", fontsize=16, fontweight='bold', pad=20)
    
    # Légende stylisée
    legend = ax.legend(facecolor=C_BG, edgecolor=C_TEXT_DIM, loc='upper left')
    for text in legend.get_texts():
        text.set_color(C_TEXT)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "7_human_vs_model_comparison.png")
    fig.savefig(out_path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {out_path}")

if __name__ == "__main__":
    generate_comparison_plot()
