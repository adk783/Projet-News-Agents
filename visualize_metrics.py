"""
Visualisation des métriques d'entraînement de l'agent d'incertitude.
Génère 4 graphiques :
1. Loss curve (train & eval)
2. Distribution des weak labels
3. Comparaison inférence haute vs basse incertitude
4. Heatmap des scores par catégorie
"""

import json
import os
import sys
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Couleurs premium ──
COLORS = {
    'bg': '#0f0f1a',
    'card': '#1a1a2e',
    'accent1': '#e94560',    # Rouge
    'accent2': '#0f3460',    # Bleu foncé
    'accent3': '#16c79a',    # Vert
    'accent4': '#f5a623',    # Orange
    'text': '#e8e8e8',
    'text_dim': '#888899',
    'grid': '#2a2a3e',
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "metrics_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def style_ax(ax, title, xlabel='', ylabel=''):
    """Style commun pour tous les axes."""
    ax.set_facecolor(COLORS['card'])
    ax.set_title(title, color=COLORS['text'], fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=COLORS['text_dim'], fontsize=10)
    ax.set_ylabel(ylabel, color=COLORS['text_dim'], fontsize=10)
    ax.tick_params(colors=COLORS['text_dim'], labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])


def load_trainer_state():
    """Charge les logs du Trainer depuis trainer_state.json."""
    state_paths = [
        "./uncertainty_model/checkpoint-9/trainer_state.json",
        "./uncertainty_model/checkpoint-6/trainer_state.json",
        "./uncertainty_model/trainer_state.json",
    ]
    for p in state_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    return None


def plot_loss_curves(fig, ax):
    """Graphique 1 : Courbes de loss train & eval."""
    state = load_trainer_state()

    if state and 'log_history' in state:
        logs = state['log_history']
        train_steps, train_losses = [], []
        eval_epochs, eval_losses = [], []

        for entry in logs:
            if 'loss' in entry and 'eval_loss' not in entry:
                train_steps.append(entry.get('step', 0))
                train_losses.append(entry['loss'])
            if 'eval_loss' in entry:
                eval_epochs.append(entry.get('epoch', 0))
                eval_losses.append(entry['eval_loss'])

        if train_steps:
            ax.plot(train_steps, train_losses, color=COLORS['accent1'],
                    linewidth=2.5, marker='o', markersize=6, label='Train Loss',
                    zorder=5)
            ax.fill_between(train_steps, train_losses, alpha=0.1, color=COLORS['accent1'])

        if eval_epochs:
            # Map eval epochs to steps for overlay
            ax2 = ax.twinx()
            ax2.plot(eval_epochs, eval_losses, color=COLORS['accent3'],
                     linewidth=2.5, marker='s', markersize=7, label='Eval Loss',
                     zorder=5)
            ax2.set_ylabel('Eval Loss', color=COLORS['accent3'], fontsize=10)
            ax2.tick_params(colors=COLORS['accent3'], labelsize=9)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_color(COLORS['accent3'])
            ax2.spines['bottom'].set_visible(False)

            # Annotation du meilleur eval
            best_idx = np.argmin(eval_losses)
            ax2.annotate(f'Best: {eval_losses[best_idx]:.4f}',
                        xy=(eval_epochs[best_idx], eval_losses[best_idx]),
                        xytext=(20, 20), textcoords='offset points',
                        fontsize=9, color=COLORS['accent3'],
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=COLORS['accent3'],
                                       lw=1.5))

        ax.legend(loc='upper left', facecolor=COLORS['card'],
                 edgecolor=COLORS['grid'], labelcolor=COLORS['text'], fontsize=9)
        if eval_epochs:
            ax2.legend(loc='upper right', facecolor=COLORS['card'],
                      edgecolor=COLORS['grid'], labelcolor=COLORS['text'], fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Pas de données trainer_state.json',
               transform=ax.transAxes, ha='center', va='center',
               color=COLORS['text_dim'], fontsize=12)

    style_ax(ax, 'Courbes de Loss (Train & Eval)', 'Steps / Epochs', 'Train Loss')


def plot_weak_label_distribution(fig, ax):
    """Graphique 2 : Distribution des scores de weak labeling."""
    # Recalculer les weak labels
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uncertainty_agent import download_lm_uncertainty_lexicon, compute_uncertainty_score, _generate_synthetic_data

    lexicon = download_lm_uncertainty_lexicon()
    df = _generate_synthetic_data(lexicon)
    scores = [compute_uncertainty_score(t, lexicon) for t in df['content']]

    # Catégoriser
    low = [s for s in scores if s <= 0.3]
    mid = [s for s in scores if 0.3 < s <= 0.6]
    high = [s for s in scores if s > 0.6]

    # Histogramme
    bins = np.linspace(0, 1, 15)
    ax.hist(scores, bins=bins, color=COLORS['accent4'], alpha=0.85,
            edgecolor=COLORS['card'], linewidth=1.2, zorder=5)

    # Zones colorées
    ax.axvspan(0, 0.3, alpha=0.08, color=COLORS['accent3'], label=f'Basse ({len(low)})')
    ax.axvspan(0.3, 0.6, alpha=0.08, color=COLORS['accent4'], label=f'Moyenne ({len(mid)})')
    ax.axvspan(0.6, 1.0, alpha=0.08, color=COLORS['accent1'], label=f'Haute ({len(high)})')

    # Lignes de séparation
    ax.axvline(x=0.3, color=COLORS['accent3'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=0.6, color=COLORS['accent1'], linestyle='--', alpha=0.6, linewidth=1.5)

    # Stats
    mean_s = np.mean(scores)
    ax.axvline(x=mean_s, color=COLORS['text'], linestyle='-', alpha=0.8, linewidth=2,
              label=f'Moyenne: {mean_s:.3f}')

    ax.legend(loc='upper left', facecolor=COLORS['card'],
             edgecolor=COLORS['grid'], labelcolor=COLORS['text'], fontsize=8)
    ax.set_xlim(0, 1)

    style_ax(ax, 'Distribution des Weak Labels', 'Score d\'incertitude', 'Nombre de textes')


def plot_inference_comparison(fig, ax):
    """Graphique 3 : Barres de scores d'inférence."""
    try:
        from uncertainty_agent import UncertaintyAgent
        agent = UncertaintyAgent()

        texts = {
            'Volatile + uncertain\nmarket outlook': 'Markets remain volatile amid trade tensions and uncertainty about future policy directions.',
            'Strong earnings\nreport': 'The company reported strong quarterly earnings, exceeding analyst expectations by 15 percent.',
            'Hesitant investors\nrumors': 'Investors are hesitant amid rumors of possible restructuring and unpredictable outcomes.',
            'Steady revenue\ngrowth': 'Revenue grew steadily at 8 percent year-over-year driven by robust demand across all segments.',
            'Unclear merger\nprobability': 'The probability of a merger remains unclear and preliminary estimates suggest variable outcomes.',
        }

        labels = list(texts.keys())
        scores = [agent.predict(t) for t in texts.values()]

        # Couleur par score
        colors = []
        for s in scores:
            if s > 0.6:
                colors.append(COLORS['accent1'])
            elif s > 0.3:
                colors.append(COLORS['accent4'])
            else:
                colors.append(COLORS['accent3'])

        bars = ax.barh(range(len(labels)), scores, color=colors, alpha=0.85,
                       edgecolor=[c for c in colors], linewidth=1.5, height=0.6,
                       zorder=5)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, color=COLORS['text'])
        ax.set_xlim(0, 1)

        # Annotations
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f'{score:.4f}',
                   va='center', fontsize=10, fontweight='bold',
                   color=colors[i])

        # Lignes de seuil
        ax.axvline(x=0.3, color=COLORS['accent3'], linestyle=':', alpha=0.5)
        ax.axvline(x=0.6, color=COLORS['accent1'], linestyle=':', alpha=0.5)

    except Exception as e:
        ax.text(0.5, 0.5, f'Modèle non disponible:\n{e}',
               transform=ax.transAxes, ha='center', va='center',
               color=COLORS['text_dim'], fontsize=10)

    style_ax(ax, 'Scores d\'Inférence par Texte', 'Score d\'incertitude [0, 1]', '')


def plot_model_summary(fig, ax):
    """Graphique 4 : Résumé visuel du modèle (paramètres, config)."""
    ax.set_facecolor(COLORS['card'])
    ax.axis('off')

    # Données du modèle
    total_params = 9_778_690
    trainable_params = 26_370
    frozen_params = total_params - trainable_params
    pct = trainable_params / total_params * 100

    # Donut chart des paramètres
    ax_inset = fig.add_axes([0.56, 0.06, 0.35, 0.35])
    ax_inset.set_facecolor(COLORS['card'])

    sizes = [trainable_params, frozen_params]
    colors_pie = [COLORS['accent1'], COLORS['accent2']]
    explode = (0.05, 0)

    wedges, texts, autotexts = ax_inset.pie(
        sizes, explode=explode, colors=colors_pie,
        autopct='%1.2f%%', startangle=90,
        pctdistance=0.75, textprops={'color': COLORS['text'], 'fontsize': 9}
    )

    # Cercle central pour donut
    centre = plt.Circle((0, 0), 0.55, fc=COLORS['card'])
    ax_inset.add_artist(centre)
    ax_inset.text(0, 0, 'LoRA', ha='center', va='center',
                 fontsize=14, fontweight='bold', color=COLORS['accent1'])

    ax_inset.legend(['Entraînables (LoRA)', 'Gelés (FinBERT)'],
                   loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   facecolor=COLORS['card'], edgecolor=COLORS['grid'],
                   labelcolor=COLORS['text'], fontsize=8)

    # Texte résumé à gauche
    summary_text = (
        f"┌─────────────────────────────────┐\n"
        f"│  CONFIGURATION DU MODÈLE        │\n"
        f"├─────────────────────────────────┤\n"
        f"│  Base        : ProsusAI/finbert │\n"
        f"│  Méthode     : LoRA (r=8, α=16) │\n"
        f"│  Tâche       : Régression [0,1] │\n"
        f"│  Max tokens  : 512              │\n"
        f"├─────────────────────────────────┤\n"
        f"│  Total params: {total_params:>12,}  │\n"
        f"│  Entraînables: {trainable_params:>12,}  │\n"
        f"│  Ratio       : {pct:>10.2f}%    │\n"
        f"│  Epochs      : 3                │\n"
        f"│  Batch size  : 4                │\n"
        f"│  LR          : 2e-4             │\n"
        f"└─────────────────────────────────┘"
    )

    ax.text(0.02, 0.95, summary_text, transform=ax.transAxes,
           fontfamily='monospace', fontsize=9, color=COLORS['text'],
           va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg'],
                    edgecolor=COLORS['grid'], alpha=0.8))

    ax.set_title('Configuration & Paramètres du Modèle',
                color=COLORS['text'], fontsize=14, fontweight='bold', pad=12)


def main():
    """Génère tous les graphiques."""
    print("Génération des graphiques de métriques...")

    fig = plt.figure(figsize=(18, 14), facecolor=COLORS['bg'])
    fig.suptitle('UNCERTAINTY AGENT — Métriques d\'Entraînement',
                fontsize=20, fontweight='bold', color=COLORS['text'],
                y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.96, top=0.92, bottom=0.05)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    plot_loss_curves(fig, ax1)

    # 2. Weak label distribution
    ax2 = fig.add_subplot(gs[0, 1])
    plot_weak_label_distribution(fig, ax2)

    # 3. Inference comparison
    ax3 = fig.add_subplot(gs[1, 0])
    plot_inference_comparison(fig, ax3)

    # 4. Model summary
    ax4 = fig.add_subplot(gs[1, 1])
    plot_model_summary(fig, ax4)

    # Sauvegarde
    output_path = os.path.join(PLOTS_DIR, "training_metrics.png")
    fig.savefig(output_path, dpi=150, facecolor=COLORS['bg'],
               edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"Graphiques sauvegardés dans : {output_path}")
    return output_path


if __name__ == "__main__":
    main()
