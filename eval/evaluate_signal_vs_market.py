"""
evaluate_signal_vs_market.py — Couche 9 : Comparaison Signal Pipeline vs Réalité Marché
=========================================================================================
OBJECTIF CENTRAL :
  Comparer systématiquement les prédictions du pipeline (Achat / Vente / Neutre)
  avec ce qui s'est RÉELLEMENT passé sur les marchés financiers, sur 3 horizons
  temporels distincts.

  Répondre à la question : "Mon système prédit-il la bonne direction, et combien
  de temps faut-il pour que l'effet se matérialise ?"

LES 3 HORIZONS TEMPORELS :
  ┌─────────────────┬──────────────────────────────────────────────────────────┐
  │ Horizon         │ Définition        │ Logique financière                   │
  ├─────────────────┼───────────────────┼──────────────────────────────────────┤
  │ COURT TERME     │ +1 à +5 jours     │ Réaction immédiate à la news (drift) │
  │ MOYEN TERME     │ +10 à +30 jours   │ Intégration par les institutionnels   │
  │ LONG TERME      │ +60 à +120 jours  │ Tendance fondamentale confirmée       │
  └─────────────────┴───────────────────┴──────────────────────────────────────┘

MÉTRIQUES PAR HORIZON :
  1. Accuracy directionnelle
     → % de fois où le signal prédit correctement Up/Down/Flat.

  2. Hit Rate conditionnel par signal
     → P(Return > 0 | signal = Achat), P(Return < 0 | signal = Vente), etc.

  3. Return moyen conditionnel
     → Le return moyen observé quand le pipeline dit "Achat", quand il dit "Vente", etc.

  4. Information Coefficient (IC) de Spearman
     → Corrélation de rang entre le signal numérique (+1 Achat, 0 Neutre, -1 Vente)
        et le return réel. Un IC > 0.1 est jugé significatif en finance quantitative.

  5. p-value du IC (test bilatéral de Spearman)
     → H0: IC = 0 (pas d'information dans le signal)

  6. Confusion Matrix signal vs direction réelle
     → Comparer les classes signal (Achat/Vente/Neutre) vs direction observée (Up/Flat/Down)

  7. P&L simulé par horizon vs benchmark SPY
     → Simuler un portefeuille qui suit tous les signaux Achat/Vente
        et comparer au Buy-and-Hold SPY sur la même période.

SEUILS DE DIRECTION (pour classifier le return réel) :
  - Up   : return ≥ +1.0%   (mouvement haussier significatif)
  - Down : return ≤ -1.0%   (mouvement baissier significatif)
  - Flat : -1.0% < return < +1.0% (bruit de marché)

SOURCES DE DONNÉES :
  - SQLite (`data/news_database.db`) pour les décisions du pipeline live
  - Résultats de L2 (benchmark annoté) pour les prédictions avec ground truth
  - Prix via yfinance pour calculer les returns réels

Lancé via : python eval/run_eval.py --layer 9
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import yfinance as yf

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger("SignalVsMarket")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

DATABASE_PATH = "data/news_database.db"
EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"

# ---------------------------------------------------------------------------
# Définition des 3 horizons
# ---------------------------------------------------------------------------

HORIZONS = {
    "COURT_TERME": {
        "label": "Court terme (1–5j)",
        "days_mid": 3,  # Milieu de la fenêtre → point de mesure principal
        "days_start": 1,
        "days_end": 5,
        "description": "Réaction immédiate à la news (momentum informationnel)",
    },
    "MOYEN_TERME": {
        "label": "Moyen terme (10–30j)",
        "days_mid": 20,
        "days_start": 10,
        "days_end": 30,
        "description": "Intégration progressive par les investisseurs institutionnels",
    },
    "LONG_TERME": {
        "label": "Long terme (60–120j)",
        "days_mid": 90,
        "days_start": 60,
        "days_end": 120,
        "description": "Confirmation de la tendance fondamentale",
    },
}

# Seuil pour Up / Down / Flat (en %)
DIRECTION_THRESHOLD = 1.0  # ±1%

# Encodage numérique des signaux pour le calcul de l'IC
SIGNAL_TO_NUM = {"Achat": 1, "Neutre": 0, "Vente": -1}


# ---------------------------------------------------------------------------
# Utilitaires prix
# ---------------------------------------------------------------------------

_price_cache: dict = {}


def _get_price(ticker: str, date_str: str, offset_days: int = 0) -> Optional[float]:
    """Cours de clôture à date_str + offset_days jours ouvrés (via yfinance)."""
    key = (ticker, date_str, offset_days)
    if key in _price_cache:
        return _price_cache[key]
    try:
        base = datetime.strptime(date_str[:10], "%Y-%m-%d")
        target = base + timedelta(days=offset_days)
        end = target + timedelta(days=7)  # Marge weekends / jours fériés
        hist = yf.Ticker(ticker).history(
            start=target.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if hist.empty:
            _price_cache[key] = None
            return None
        price = round(float(hist["Close"].iloc[0]), 4)
        _price_cache[key] = price
        return price
    except Exception as e:
        logger.debug("Prix %s @ %s+%dd : %s", ticker, date_str, offset_days, e)
        _price_cache[key] = None
        return None


def _compute_return(ticker: str, date_str: str, offset_days: int) -> Optional[float]:
    """Return en % entre J0 (date_str) et J+offset_days."""
    p0 = _get_price(ticker, date_str, 0)
    pn = _get_price(ticker, date_str, offset_days)
    if p0 is None or pn is None or p0 == 0:
        return None
    return round((pn - p0) / p0 * 100, 4)


def _direction(return_pct: float) -> str:
    """Classifie un return en Up / Flat / Down."""
    if return_pct >= DIRECTION_THRESHOLD:
        return "Up"
    if return_pct <= -DIRECTION_THRESHOLD:
        return "Down"
    return "Flat"


def _spy_return(date_str: str, offset_days: int) -> Optional[float]:
    """Return SPY buy-and-hold sur la même période (benchmark passif)."""
    return _compute_return("SPY", date_str, offset_days)


# ---------------------------------------------------------------------------
# Chargement des prédictions
# ---------------------------------------------------------------------------


def _load_all_predictions() -> list[dict]:
    """
    Charge toutes les prédictions disponibles depuis :
      1. SQLite (pipeline live)
      2. Dernier run L2 (benchmark annoté)
    """
    predictions = []

    # 1. SQLite
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker, date_utc, title, signal_final
            FROM articles
            WHERE signal_final IN ('Achat', 'Vente', 'Neutre')
              AND date_utc IS NOT NULL
              AND date_utc != ''
            ORDER BY date_utc DESC
        """)
        for row in cur.fetchall():
            predictions.append(
                {
                    "source": "sqlite_live",
                    "id": f"{row['ticker']}_{row['date_utc'][:10]}",
                    "ticker": row["ticker"],
                    "date": row["date_utc"][:10],
                    "signal": row["signal_final"],
                    "title": (row["title"] or "")[:60],
                }
            )
        conn.close()
    except Exception as e:
        logger.warning("SQLite: %s", e)

    # 2. L2 — Dernier run benchmark
    if EVAL_RESULTS_DIR.exists():
        l2_runs = sorted(
            [d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir() and "pipeline_layer2" in d.name],
            reverse=True,
        )
        if l2_runs:
            details_file = l2_runs[0] / "details.json"
            if details_file.exists():
                try:
                    with open(details_file, encoding="utf-8") as f:
                        l2_data = json.load(f)
                    for r in l2_data:
                        if (
                            r.get("filtrage_pred") == "pertinent"
                            and r.get("signal_pred") in ("Achat", "Vente", "Neutre")
                            and r.get("ground_truth", {}).get("date")
                        ):
                            predictions.append(
                                {
                                    "source": "benchmark_L2",
                                    "id": r["id"],
                                    "ticker": r["ticker"],
                                    "date": r["ground_truth"]["date"][:10],
                                    "signal": r["signal_pred"],
                                    "title": r.get("title", "")[:60],
                                }
                            )
                except Exception as e:
                    logger.warning("L2 load: %s", e)

    # Déduplique sur (ticker, date, signal)
    seen = set()
    unique = []
    for p in predictions:
        key = (p["ticker"], p["date"], p["signal"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ---------------------------------------------------------------------------
# Métriques statistiques
# ---------------------------------------------------------------------------


def _spearman_ic(signals_num: list[float], returns: list[float]) -> dict:
    """
    Information Coefficient (IC) de Spearman.
    Mesure la corrélation de rang entre le signal et le return réel.
    → IC > 0 : le signal prédit la direction avec succès
    → IC = 0 : pas d'information
    → IC < 0 : le signal est inversé (catastrophique)

    Interprétation pratique en finance quantitative :
      IC > 0.10 → signal informatif
      IC > 0.30 → signal fort
    """
    if len(signals_num) < 4:
        return {"ic": None, "p_value": None, "n": len(signals_num)}

    arr_s = np.array(signals_num)
    arr_r = np.array(returns)

    if HAS_SCIPY:
        ic, p_val = scipy_stats.spearmanr(arr_s, arr_r)
        ic = round(float(ic), 4)
        p_val = round(float(p_val), 4)
    else:
        # Calcul manuel du rho de Spearman
        n = len(arr_s)
        rank_s = np.argsort(np.argsort(arr_s)).astype(float)
        rank_r = np.argsort(np.argsort(arr_r)).astype(float)
        d2 = np.sum((rank_s - rank_r) ** 2)
        ic = round(float(1 - 6 * d2 / (n * (n**2 - 1))), 4) if n > 2 else 0.0
        # p-value approximée via t-distribution
        if n > 2 and abs(ic) < 1:
            t = ic * math.sqrt((n - 2) / (1 - ic**2))
            p_val = round(float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))), 4)
        else:
            p_val = 1.0

    return {
        "ic": ic,
        "p_value": p_val,
        "n": len(signals_num),
        "significant": p_val < 0.05 if p_val is not None else False,
        "interpretation": (
            "Fort et significatif"
            if (ic or 0) > 0.30 and p_val < 0.05
            else "Bon et significatif"
            if (ic or 0) > 0.10 and p_val < 0.05
            else "Faible mais positif"
            if (ic or 0) > 0
            else "Inversé (problème)"
            if (ic or 0) < -0.05
            else "Non informatif"
        ),
    }


def _confusion_matrix(signals: list[str], directions: list[str]) -> dict:
    """
    Construit la matrice de confusion : Signal prédit × Direction réelle.
    Lignes = signal prédit (Achat / Neutre / Vente)
    Colonnes = direction observée (Up / Flat / Down)
    """
    matrix = {s: {"Up": 0, "Flat": 0, "Down": 0} for s in ["Achat", "Neutre", "Vente"]}
    for sig, dir_ in zip(signals, directions):
        if sig in matrix and dir_ in ("Up", "Flat", "Down"):
            matrix[sig][dir_] += 1
    return matrix


def _print_confusion_matrix(matrix: dict, horizon_label: str) -> None:
    """Affiche la matrice de confusion en ASCII."""
    print(f"\n  Matrice de Confusion — {horizon_label}")
    print(f"  {'Signal↓ / Marché→':<20} {'Up':>8} {'Flat':>8} {'Down':>8} {'Total':>8}")
    print(f"  {'-' * 52}")
    for sig in ["Achat", "Neutre", "Vente"]:
        row = matrix.get(sig, {"Up": 0, "Flat": 0, "Down": 0})
        total = sum(row.values())
        up_pct = f"{row['Up'] / total:.0%}" if total else "N/A"
        flat_pct = f"{row['Flat'] / total:.0%}" if total else "N/A"
        down_pct = f"{row['Down'] / total:.0%}" if total else "N/A"
        print(
            f"  {sig:<20} {row['Up']:>4}({up_pct:>3}) {row['Flat']:>4}({flat_pct:>3}) "
            f"{row['Down']:>4}({down_pct:>3}) {total:>8}"
        )


def _hit_rate_by_signal(signals: list[str], returns: list[float]) -> dict:
    """
    Hit rate conditionnel par signal :
    - P(return > 0 | signal = Achat) → devrait être > 50%
    - P(return < 0 | signal = Vente) → devrait être > 50%
    - P(|return| < threshold | signal = Neutre) → devrait être élevé
    """
    by_signal: dict[str, list[float]] = {"Achat": [], "Vente": [], "Neutre": []}
    for sig, ret in zip(signals, returns):
        if sig in by_signal:
            by_signal[sig].append(ret)

    hit_rates = {}
    for sig, rets in by_signal.items():
        if not rets:
            hit_rates[sig] = {"n": 0, "hit_rate": None, "mean_return": None}
            continue

        if sig == "Achat":
            hits = sum(1 for r in rets if r > 0)
            definition = "P(return > 0 | Achat)"
        elif sig == "Vente":
            hits = sum(1 for r in rets if r < 0)
            definition = "P(return < 0 | Vente)"
        else:  # Neutre
            hits = sum(1 for r in rets if abs(r) < DIRECTION_THRESHOLD)
            definition = f"P(|return| < {DIRECTION_THRESHOLD}% | Neutre)"

        hit_rates[sig] = {
            "n": len(rets),
            "hit_rate": round(hits / len(rets), 3) if rets else None,
            "mean_return": round(sum(rets) / len(rets), 3),
            "std_return": round(float(np.std(rets)), 3) if len(rets) > 1 else 0.0,
            "definition": definition,
        }
    return hit_rates


# ---------------------------------------------------------------------------
# Analyse par horizon
# ---------------------------------------------------------------------------


def _analyse_horizon(
    predictions: list[dict],
    horizon_key: str,
    offset_days: int,
) -> dict:
    """
    Pour un horizon donné (offset_days), calcule toutes les métriques
    en récupérant les returns réels via yfinance.
    """
    h = HORIZONS[horizon_key]
    print(f"\n  Récupération des prix ({offset_days}j)...", end=" ", flush=True)

    valid_data: list[dict] = []
    skipped = 0

    for pred in predictions:
        ret = _compute_return(pred["ticker"], pred["date"], offset_days)
        spy = _spy_return(pred["date"], offset_days)
        if ret is None:
            skipped += 1
            continue
        valid_data.append(
            {
                **pred,
                "return_pct": ret,
                "spy_return": spy,
                "direction": _direction(ret),
                "signal_num": SIGNAL_TO_NUM.get(pred["signal"], 0),
            }
        )

    print(f"{len(valid_data)} trades valides ({skipped} ignorés).")

    if not valid_data:
        return {"horizon": horizon_key, "n": 0, "error": "Pas de données de prix"}

    signals = [d["signal"] for d in valid_data]
    returns = [d["return_pct"] for d in valid_data]
    directions = [d["direction"] for d in valid_data]
    sigs_num = [d["signal_num"] for d in valid_data]

    # Accuracy directionnelle
    correct_direction = sum(
        1
        for d in valid_data
        if (d["signal"] == "Achat" and d["direction"] == "Up")
        or (d["signal"] == "Vente" and d["direction"] == "Down")
        or (d["signal"] == "Neutre" and d["direction"] == "Flat")
    )
    accuracy = round(correct_direction / len(valid_data), 3)

    # IC de Spearman
    ic_result = _spearman_ic(sigs_num, returns)

    # Hit rates par signal
    hit_rates = _hit_rate_by_signal(signals, returns)

    # Confusion matrix
    confusion = _confusion_matrix(signals, directions)

    # P&L simulé (Achat/Vente uniquement) avec Slippage (0.05% par trade)
    SLIPPAGE = 0.05
    active_rets = [
        (d["return_pct"] - SLIPPAGE)
        if d["signal"] == "Achat"
        else (-d["return_pct"] - SLIPPAGE)
        if d["signal"] == "Vente"
        else 0.0
        for d in valid_data
        if d["signal"] != "Neutre"
    ]
    portfolio_ret = round(sum(active_rets), 2) if active_rets else None
    portfolio_mean = round(sum(active_rets) / len(active_rets), 3) if active_rets else None

    # Benchmark SPY
    spy_rets = [d["spy_return"] for d in valid_data if d.get("spy_return") is not None]
    spy_mean = round(sum(spy_rets) / len(spy_rets), 3) if spy_rets else None
    alpha = round(portfolio_mean - spy_mean, 3) if portfolio_mean is not None and spy_mean is not None else None

    # Return moyen par signal
    mean_ret_by_signal = {}
    for sig in ["Achat", "Vente", "Neutre"]:
        rets_sig = [d["return_pct"] for d in valid_data if d["signal"] == sig]
        mean_ret_by_signal[sig] = round(sum(rets_sig) / len(rets_sig), 3) if rets_sig else None

    return {
        "horizon": horizon_key,
        "label": h["label"],
        "description": h["description"],
        "offset_days": offset_days,
        "n_total": len(predictions),
        "n_valid": len(valid_data),
        "n_skipped": skipped,
        "accuracy_directionnelle": accuracy,
        "ic": ic_result,
        "hit_rates": hit_rates,
        "confusion": confusion,
        "mean_return_by_signal": mean_ret_by_signal,
        "portfolio": {
            "n_active_trades": len(active_rets) if active_rets else 0,
            "total_return": portfolio_ret,
            "mean_return": portfolio_mean,
            "spy_mean": spy_mean,
            "alpha": alpha,
        },
        "details": valid_data,
    }


# ---------------------------------------------------------------------------
# Affichage des résultats
# ---------------------------------------------------------------------------


def _print_horizon_report(h: dict) -> None:
    """Affiche le rapport complet pour un horizon."""
    hl = h.get("label", h.get("horizon", ""))
    print(f"\n  {'═' * 65}")
    print(f"  {hl.upper()} — {h.get('description', '')}")
    print(f"  {'─' * 65}")

    n = h.get("n_valid", 0)
    if n == 0:
        print("  Aucune donnée de prix disponible pour cet horizon.")
        return

    acc = h.get("accuracy_directionnelle", 0)
    ic = h.get("ic", {})
    port = h.get("portfolio", {})

    print(f"\n  [1] Accuracy directionnelle : {acc:.1%}  ({n} décisions évaluées)")
    if acc >= 0.60:
        print("      ✓ Bon signal — prédit la bonne direction plus de 60% du temps.")
    elif acc >= 0.50:
        print("      ~ Médiocre — légèrement au-dessus du hasard.")
    else:
        print("      ✗ Sous le hasard — le signal est inversé ou non informatif.")

    print("\n  [2] Information Coefficient (IC Spearman) :")
    ic_val = ic.get("ic")
    p_val = ic.get("p_value")
    if ic_val is not None:
        sig_star = "***" if (p_val or 1) < 0.01 else "**" if (p_val or 1) < 0.05 else "*" if (p_val or 1) < 0.10 else ""
        print(f"      IC = {ic_val:+.4f}{sig_star}  (p = {p_val:.4f}, n = {ic.get('n', 0)})")
        print(f"      Interprétation : {ic.get('interpretation', 'N/A')}")
        if ic.get("significant"):
            print("      ✓ Signal statistiquement significatif (p < 0.05)")
        else:
            print("      ~ Signal non significatif (p ≥ 0.05) — attention à la taille d'échantillon")
    else:
        print(f"      N/A (échantillon insuffisant : {ic.get('n', 0)} obs.)")

    print("\n  [3] Return moyen observé par signal :")
    for sig, mean_r in h.get("mean_return_by_signal", {}).items():
        if mean_r is not None:
            n_sig = h.get("hit_rates", {}).get(sig, {}).get("n", 0)
            mark = "▲" if (sig == "Achat" and mean_r > 0) or (sig == "Vente" and mean_r < 0) else "▼"
            print(f"      {sig:<8} : {mean_r:+.3f}%  ({mark}, n={n_sig})")
        else:
            print(f"      {sig:<8} : N/A")

    print("\n  [4] Hit Rate conditionnel par signal :")
    hr = h.get("hit_rates", {})
    for sig in ["Achat", "Vente", "Neutre"]:
        data = hr.get(sig, {})
        hit = data.get("hit_rate")
        n_sig = data.get("n", 0)
        defn = data.get("definition", "")
        if hit is not None:
            quality = "✓" if hit >= 0.55 else "~" if hit >= 0.45 else "✗"
            print(f"      {quality} {defn:<40} : {hit:.1%}  (n={n_sig})")
        else:
            print(f"      {sig:<8}: N/A")

    print("\n  [5] P&L simulé vs SPY :")
    n_trades = port.get("n_active_trades", 0)
    if n_trades > 0:
        total_ret = port.get("total_return")
        mean_ret = port.get("mean_return")
        spy_m = port.get("spy_mean")
        alpha_val = port.get("alpha")
        print(f"      Trades actifs  : {n_trades}")
        print(f"      Return cumulé  : {total_ret:+.2f}%" if total_ret is not None else "      Return cumulé  : N/A")
        print(
            f"      Return moyen   : {mean_ret:+.3f}% / trade" if mean_ret is not None else "      Return moyen   : N/A"
        )
        print(f"      SPY moyen      : {spy_m:+.3f}% / trade" if spy_m is not None else "      SPY moyen      : N/A")
        if alpha_val is not None:
            alpha_mark = "✓ Surperformance" if alpha_val > 0 else "✗ Sous-performance"
            print(f"      Alpha          : {alpha_val:+.3f}% → {alpha_mark}")
    else:
        print("      Aucun trade actif (que des Neutres ou données manquantes)")

    _print_confusion_matrix(h.get("confusion", {}), hl)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_signal_vs_market_analysis() -> dict:
    """
    Couche 9 : Comparaison complète signal pipeline vs réalité marché.
    Lance l'analyse sur les 3 horizons temporels.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 9 : Comparaison Signal Pipeline vs Réalité Marché")
    print("          Court terme (1–5j) | Moyen terme (10–30j) | Long terme (60–120j)")
    print(f"{'=' * 70}")

    predictions = _load_all_predictions()
    if not predictions:
        print("\n  [INFO] Aucune prédiction disponible.")
        print("  → Lancez d'abord : python eval/run_eval.py --layer 2 --limit 10")
        print("  → Ou utilisez le pipeline sur des news réelles.")
        return {}

    print(f"\n  {len(predictions)} prédictions chargées (SQLite + L2).")
    print("  Récupération des prix yfinance sur 3 horizons...")
    print(f"  Seuil direction : Up ≥ +{DIRECTION_THRESHOLD}%, Down ≤ -{DIRECTION_THRESHOLD}%\n")

    results = {}
    horizon_map = {
        "COURT_TERME": HORIZONS["COURT_TERME"]["days_mid"],
        "MOYEN_TERME": HORIZONS["MOYEN_TERME"]["days_mid"],
        "LONG_TERME": HORIZONS["LONG_TERME"]["days_mid"],
    }

    for h_key, days in horizon_map.items():
        hr = _analyse_horizon(predictions, h_key, days)
        results[h_key] = hr

    # Affichage détaillé par horizon
    for h_key in ["COURT_TERME", "MOYEN_TERME", "LONG_TERME"]:
        _print_horizon_report(results[h_key])

    # Résumé comparatif
    print(f"\n  {'=' * 70}")
    print("  RÉSUMÉ COMPARATIF — SIGNAL VS MARCHÉ")
    print(f"  {'=' * 70}")
    print(f"\n  {'Horizon':<22} {'Accuracy':>10} {'IC':>8} {'p-val':>8} {'Alpha':>8}")
    print(f"  {'-' * 58}")

    for h_key in ["COURT_TERME", "MOYEN_TERME", "LONG_TERME"]:
        hr = results[h_key]
        label = HORIZONS[h_key]["label"][:20]
        acc = hr.get("accuracy_directionnelle")
        ic_d = hr.get("ic", {})
        ic_v = ic_d.get("ic")
        p_v = ic_d.get("p_value")
        alpha = hr.get("portfolio", {}).get("alpha")

        acc_s = f"{acc:.1%}" if acc is not None else "N/A"
        ic_s = f"{ic_v:+.3f}" if ic_v is not None else "N/A"
        p_s = f"{p_v:.3f}" if p_v is not None else "N/A"
        alpha_s = f"{alpha:+.3f}%" if alpha is not None else "N/A"

        sig_mark = "***" if (p_v or 1) < 0.01 else "**" if (p_v or 1) < 0.05 else " "
        print(f"  {label:<22} {acc_s:>10} {ic_s:>8}{sig_mark:<2} {p_s:>8} {alpha_s:>8}")

    # Verdict global
    print(f"\n  {'=' * 70}")
    print("  ANALYSE GLOBALE :")

    ics = [results[h].get("ic", {}).get("ic") for h in results if results[h].get("ic", {}).get("ic") is not None]
    if ics:
        best_h = max(results, key=lambda h: results[h].get("ic", {}).get("ic") or -999)
        best_ic = results[best_h].get("ic", {}).get("ic", 0)
        print(f"\n  Meilleur horizon pour le signal : {HORIZONS[best_h]['label']}")
        print(f"  (IC = {best_ic:+.4f})")

        if best_ic > 0.30:
            print("  → Le signal est FORT. Concentrer les décisions sur cet horizon.")
        elif best_ic > 0.10:
            print("  → Le signal est BON mais pas exceptionnel. Filtrer avec le niveau de risque YOLO.")
        elif best_ic > 0:
            print("  → Le signal est FAIBLE. Il faut plus d'historique pour confirmer.")
        else:
            print("  → Le signal ne prédit pas la direction du marché sur ces horizons.")
            print("  → Vérifier les seuils de décision ou l'horizon de trading visé.")

    print("=" * 70)

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = EVAL_RESULTS_DIR / f"{timestamp}_signal_vs_market"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nettoyer les details pour la sérialisation (les arrays numpy posent problème)
    save_results = {}
    for h_key, h_data in results.items():
        save_data = {k: v for k, v in h_data.items() if k != "details"}
        # Convertir les éventuels numpy types
        save_data["details_sample"] = h_data.get("details", [])[:5]
        save_results[h_key] = save_data

    with open(out_dir / "signal_vs_market_stats.json", "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Résultats sauvegardés : {out_dir}/signal_vs_market_stats.json")

    return {
        "layer": 9,
        "n_predictions": len(predictions),
        "horizons": {
            h: {
                "n_valid": results[h].get("n_valid", 0),
                "accuracy": results[h].get("accuracy_directionnelle"),
                "ic": results[h].get("ic", {}).get("ic"),
                "p_value": results[h].get("ic", {}).get("p_value"),
                "alpha": results[h].get("portfolio", {}).get("alpha"),
            }
            for h in results
        },
    }


if __name__ == "__main__":
    run_signal_vs_market_analysis()
