"""
config.py — Configuration centralisée du pipeline

OBJECTIF
--------
Regrouper au même endroit les constantes éparpillées dans le code afin de :
  1. Voir en un coup d'œil les seuils qui pilotent la stratégie
  2. Pouvoir A/B-tester des valeurs sans patcher 12 modules
  3. Documenter la justification de chaque valeur (références, dataset,
     intuition, ou "valeur empirique à calibrer")

CONVENTION
----------
  - Les constantes spécifiques à un domaine restent dans leur module si
    elles ne sont jamais réutilisées ailleurs.
  - Seules les constantes cross-module ou "interrupteurs de stratégie"
    sont promues ici.
  - Chaque constante porte un commentaire `# [source]` : référence biblio
    ou "empirique" si c'est un magic number assumé.

Chargement override via variables d'environnement (.env ou shell) :
  ABSA_CACHE_MAX, MACRO_CONTEXT_TTL_SEC, KELLY_UNCERTAINTY_KAPPA, etc.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/news_database.db")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
WEEKLY_AUDIT_DIR = os.path.join(REPORTS_DIR, "weekly_audit")
BACKTESTS_DIR = os.path.join(REPORTS_DIR, "backtests")

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
ABSA_CACHE_MAX = int(os.getenv("ABSA_CACHE_MAX", "256"))
MACRO_CONTEXT_TTL_SEC = int(os.getenv("MACRO_CONTEXT_TTL_SEC", str(15 * 60)))

# ---------------------------------------------------------------------------
# Sizer / Kelly
# ---------------------------------------------------------------------------
# κ (kappa) : multiplicateur de pénalité Bayesian sur Var(p_win)
# f* ← f / (1 + κ · Var). Plus κ est grand, plus la taille baisse quand
# l'incertitude paramétrique est forte. κ=4 : Medo & Pignatti (2013).
KELLY_UNCERTAINTY_KAPPA = float(os.getenv("KELLY_UNCERTAINTY_KAPPA", "4.0"))

# Bornes de clamp pour f*. 0.15 = on ne sort jamais si l'edge est maigre ;
# 0.85 = protection anti-ruine (même avec p_win≈1, on ne risque pas tout).
KELLY_F_CLAMP_MIN = float(os.getenv("KELLY_F_CLAMP_MIN", "0.15"))
KELLY_F_CLAMP_MAX = float(os.getenv("KELLY_F_CLAMP_MAX", "0.85"))

# ---------------------------------------------------------------------------
# Régimes de marché (Ang & Bekaert 2002)
# ---------------------------------------------------------------------------
# Seuils SPY 20-day sur retour + volatilité pour classifier SIDEWAYS/BULL/BEAR/HIGH_VOL
REGIME_SIDEWAYS_RETURN_ABS = float(os.getenv("REGIME_SIDEWAYS_RETURN_ABS", "0.02"))  # |r_20d| < 2%
REGIME_SIDEWAYS_VOL_MAX = float(os.getenv("REGIME_SIDEWAYS_VOL_MAX", "0.015"))  # σ_20d < 1.5%
REGIME_HIGH_VOL_THRESHOLD = float(os.getenv("REGIME_HIGH_VOL_THRESHOLD", "0.025"))  # σ_20d > 2.5%

# ---------------------------------------------------------------------------
# Counterfactual invariance — seuils d'alerte (offline only)
# ---------------------------------------------------------------------------
# Veitch et al. 2021 : CI < 0.75 = pipeline utilise des priors d'entraînement
CI_SCORE_ALERT_THRESHOLD = float(os.getenv("CI_SCORE_ALERT_THRESHOLD", "0.75"))
CI_SCORE_GOOD_THRESHOLD = float(os.getenv("CI_SCORE_GOOD_THRESHOLD", "0.90"))

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
# ECE cible post-calibration pour considérer la Platt/Isotonic OK
ECE_TARGET_POST_CALIBRATION = float(os.getenv("ECE_TARGET_POST_CALIBRATION", "0.05"))

# ---------------------------------------------------------------------------
# Exécution (coûts)
# ---------------------------------------------------------------------------
# Activer Almgren-Chriss uniquement si (quantity / ADV) dépasse ce seuil
ALMGREN_CHRISS_AUTO_TRIGGER_ADV_PCT = float(
    os.getenv("ALMGREN_CHRISS_AUTO_TRIGGER_ADV_PCT", "0.005")  # 0.5%
)

# ---------------------------------------------------------------------------
# LLMs — prix indicatifs (USD par 1M tokens, entrée+sortie combinées)
# ---------------------------------------------------------------------------
# Source : tarifs publics des providers (oct 2025). Mettre à jour avant chaque
# audit de coût. Utilisé uniquement pour l'estimation de budget, pas pour le
# routing live (le routing est dans agent_debat.py).
LLM_PRICING_USD_PER_1M_TOKENS = {
    "llama-3.1-8b-instant": 0.05,  # Cerebras free-tier → $0.05 en payant
    "llama-4-scout-17b": 0.11,
    "mistral-small-latest": 0.20,
    "llama-3.3-70b-versatile": 0.59,
}

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
# Permet d'activer/désactiver certaines features sans redéploiement
ENABLE_TEMPORAL_FENCE = os.getenv("ENABLE_TEMPORAL_FENCE", "1") == "1"
ENABLE_BAYESIAN_AGGREGATOR = os.getenv("ENABLE_BAYESIAN_AGGREGATOR", "1") == "1"
ENABLE_POSITION_SIZER = os.getenv("ENABLE_POSITION_SIZER", "1") == "1"
EVAL_ANONYMIZE = os.getenv("EVAL_ANONYMIZE", "0") == "1"

# ---------------------------------------------------------------------------
# DRY_RUN / Paper trading — ne persiste PAS les décisions sur le portefeuille
# ---------------------------------------------------------------------------
# En mode DRY_RUN=1, le pipeline :
#   - lit les news, déroule tout le débat, calcule sizing + risque
#   - logge les décisions dans logs/dry_run_trades.jsonl
#   - NE modifie PAS portfolio_state (cash, positions)
#   - NE persiste PAS signal_final/sizing dans articles (ou sur une colonne shadow)
# Sert à tourner "à blanc" pendant 2-4 semaines avant passage broker réel.
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
DRY_RUN_LOG_PATH = os.getenv("DRY_RUN_LOG_PATH", "logs/dry_run_trades.jsonl")

# ---------------------------------------------------------------------------
# Kill-switch systémique VIX (Whaley 2009, Bollerslev 2015)
# ---------------------------------------------------------------------------
# Si VIX > VIX_KILL_SWITCH_THRESHOLD, le pipeline refuse toute nouvelle position
# et force HOLD_SYSTEMIC. Benchmarks historiques : COVID mars 2020 (VIX 82),
# Lehman oct 2008 (VIX 80), Flash Crash mai 2010 (VIX 45).
# Seuil 45 = zone "stressed market" où les algos doivent s'éteindre d'eux-mêmes.
VIX_KILL_SWITCH_THRESHOLD = float(os.getenv("VIX_KILL_SWITCH_THRESHOLD", "45.0"))

# ---------------------------------------------------------------------------
# LLM cost monitoring
# ---------------------------------------------------------------------------
# Budget quotidien en USD. Si dépassé, le pipeline lève BudgetExceededError.
# Mis à 0 pour désactiver le garde-fou (dev).
LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0"))
LLM_COST_LOG_DIR = os.getenv("LLM_COST_LOG_DIR", "reports/llm_cost_daily")

# ---------------------------------------------------------------------------
# Portfolio constraints (concentration par secteur)
# ---------------------------------------------------------------------------
# Si une nouvelle position pousserait l'exposition d'un secteur au-delà de ce
# seuil (% du capital total), l'ordre est refusé.
MAX_SECTOR_EXPOSURE_PCT = float(os.getenv("MAX_SECTOR_EXPOSURE_PCT", "0.30"))

# ---------------------------------------------------------------------------
# Cap de corrélation cross-sectionnelle (Phase 4)
# ---------------------------------------------------------------------------
# Si la position proposée a ρ > MAX_PAIRWISE_CORRELATION avec au moins une
# position existante (Pearson sur daily log-returns, fenêtre glissante
# CORRELATION_LOOKBACK_DAYS), l'ordre est refusé → `HOLD_CORR_CAP`.
#
# Justification : deux positions ρ>0.8 dupliquent le même facteur de risque.
# Rankin-Jegadeesh 1993 et Ang-Chen 2002 montrent que la corrélation
# cross-sectionnelle explose en régime stressé (typiquement en krach), donc
# la diversification apparente disparaît quand on en a le plus besoin.
MAX_PAIRWISE_CORRELATION = float(os.getenv("MAX_PAIRWISE_CORRELATION", "0.80"))
CORRELATION_LOOKBACK_DAYS = int(os.getenv("CORRELATION_LOOKBACK_DAYS", "60"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def as_dict() -> dict:
    """Snapshot sérialisable (pour logs / reports d'audit)."""
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}


if __name__ == "__main__":
    # Smoke test : dump la config courante
    import json

    print(json.dumps(as_dict(), indent=2, default=str))
