"""
test_regime_filter.py — Tests unitaires du filtre de regime SIDEWAYS (audit L12)
Ref : Hamilton (1989), Lo (2004) — calibration sur 19 trades, 0% accuracy SIDEWAYS
"""

import sys
from pathlib import Path

from src.utils.yolo_classifier import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGH_VOL,
    REGIME_SIDEWAYS,
    REGIME_UNKNOWN,
    SPY_BEAR_THRESHOLD,
    SPY_BULL_THRESHOLD,
    VIX_HIGH_THRESHOLD,
    classify_market_regime,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


print("\n=== Filtre Regime de Marche (audit L12) ===")

# --- Classification de base ---
check("UNKNOWN si aucune donnee", classify_market_regime(None, None) == REGIME_UNKNOWN)

check("HIGH_VOL si VIX > 30", classify_market_regime(spy_20d_return=2.0, vix=35.0) == REGIME_HIGH_VOL)

check("HIGH_VOL prioritaire sur BULL", classify_market_regime(spy_20d_return=5.0, vix=32.0) == REGIME_HIGH_VOL)

check("BULL si SPY > 3% et VIX < 25", classify_market_regime(spy_20d_return=4.5, vix=18.0) == REGIME_BULL)

check("BULL sans VIX disponible", classify_market_regime(spy_20d_return=4.0, vix=None) == REGIME_BULL)

check("BEAR si SPY < -3% et VIX > 20", classify_market_regime(spy_20d_return=-4.0, vix=22.0) == REGIME_BEAR)

check("SIDEWAYS si SPY entre -3% et +3%", classify_market_regime(spy_20d_return=1.0, vix=18.0) == REGIME_SIDEWAYS)

check("SIDEWAYS si SPY = -1.5%", classify_market_regime(spy_20d_return=-1.5, vix=20.0) == REGIME_SIDEWAYS)

check("SIDEWAYS si SPY = 0.0%", classify_market_regime(spy_20d_return=0.0, vix=15.0) == REGIME_SIDEWAYS)

check(
    "UNKNOWN si seul VIX disponible et VIX <= 30",
    classify_market_regime(spy_20d_return=None, vix=20.0) == REGIME_UNKNOWN,
)

check("Seuil exact BULL : SPY = 3.01%", classify_market_regime(spy_20d_return=3.01, vix=24.0) == REGIME_BULL)

check(
    "Seuil exact SIDEWAYS : SPY = 3.0% (non BULL car exactement au seuil)",
    classify_market_regime(spy_20d_return=3.0, vix=24.0) == REGIME_SIDEWAYS,
)

check("Seuil exact BEAR : SPY = -3.01%", classify_market_regime(spy_20d_return=-3.01, vix=21.0) == REGIME_BEAR)

# --- Intégration classify_risk avec filtre regime ---
print("\n=== Integration classify_risk() avec filtre regime ===")
from src.utils.yolo_classifier import classify_risk

# Cas SIDEWAYS : signal Achat doit etre veto
yolo_sw = classify_risk(
    signal_final="Achat",
    consensus_rate=0.7,
    impact_strength=0.8,
    scratchpad_xml="",
    absa_result={"aspects": [{"sentiment": "positive"}, {"sentiment": "positive"}]},
    score_finbert=0.75,
    contexte_marche={"variation_5d": 0.5},
    argument_dominant="Strong earnings beat with 15% revenue growth YoY.",
    spy_20d_return=0.8,  # SIDEWAYS
    vix=18.0,
)
check("SIDEWAYS prior : regime_veto = True (indicatif)", yolo_sw.regime_veto)
check("SIDEWAYS prior : market_regime correct", yolo_sw.market_regime == REGIME_SIDEWAYS)
check("SIDEWAYS prior : risque escalade > 0.40", yolo_sw.risk_score >= 0.40)
check("SIDEWAYS prior : mention bayesienne", any("Prior Bayesien" in r for r in yolo_sw.reasons))

# Cas BULL : signal Achat conserve
yolo_bull = classify_risk(
    signal_final="Achat",
    consensus_rate=0.8,
    impact_strength=0.85,
    scratchpad_xml="",
    absa_result={"aspects": [{"sentiment": "positive"}, {"sentiment": "positive"}]},
    score_finbert=0.80,
    contexte_marche={"variation_5d": 1.5},
    argument_dominant="Revenue grew 20% YoY, P/E at 18x, below sector average.",
    spy_20d_return=5.2,  # BULL
    vix=16.0,
)
check("BULL : pas de veto regime", not yolo_bull.regime_veto)
check("BULL : market_regime correct", yolo_bull.market_regime == REGIME_BULL)
check("BULL : features contiennent spy_20d", yolo_bull.features.get("spy_20d_return") == 5.2)
check("BULL : features contiennent vix", yolo_bull.features.get("vix_at_decision") == 16.0)

# Cas HIGH_VOL : escalade de risque
yolo_hv = classify_risk(
    signal_final="Vente",
    consensus_rate=0.6,
    impact_strength=0.7,
    scratchpad_xml="",
    absa_result={"aspects": [{"sentiment": "negative"}]},
    score_finbert=0.3,
    contexte_marche={"variation_5d": -3.0},
    argument_dominant="Debt/EBITDA at 8x with covenant breach risk.",
    spy_20d_return=-2.0,
    vix=38.0,  # HIGH_VOL
)
check("HIGH_VOL : market_regime correct", yolo_hv.market_regime == REGIME_HIGH_VOL)
check("HIGH_VOL : pas de veto (signal Vente)", not yolo_hv.regime_veto)  # HIGH_VOL n'override pas le signal
check("HIGH_VOL : risque escalade", any("[Regime HIGH_VOL]" in r for r in yolo_hv.reasons))

# Cas Latency (Critique E)
yolo_late = classify_risk(
    signal_final="Achat",
    consensus_rate=0.9,
    impact_strength=0.9,
    scratchpad_xml="",
    absa_result={"aspects": [{"sentiment": "positive"}]},
    score_finbert=0.9,
    contexte_marche={"variation_5d": 1.0},
    argument_dominant="Great news.",
    processing_time_ms=12000.0,  # 12 secondes
)
check("LATENCY : penalite appliquee", any("Latence" in r for r in yolo_late.reasons))


print("\n" + "=" * 50)
print(f"  RESULTAT : {PASS} PASS  /  {FAIL} FAIL")
print("=" * 50)
if FAIL > 0:
    sys.exit(1)
