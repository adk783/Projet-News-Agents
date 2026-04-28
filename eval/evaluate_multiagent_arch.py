"""
evaluate_multiagent_arch.py — Couche 8 : Architecture multi-agents & Mise à l'échelle
=======================================================================================
OBJECTIF :
  Répondre à la question : "L'architecture multi-agents apporte-t-elle vraiment
  de la valeur par rapport à un agent seul ?"

  Et aussi : "Quel agent contribue réellement à la décision finale ?"

DEUX SOUS-MODULES :

  8a. BENCHMARK COMPARATIF (FinGAIA-inspired)
      Teste les mêmes articles sous 3 configurations d'agents :
        - SOLO       : Un seul agent (Haussier) décide seul
        - DEBATE_2   : Haussier + Baissier (sans modérateur Neutre)
        - DEBATE_3   : Architecture complète (Haussier + Baissier + Neutre)
      Compare les performances vs ground truth sur le benchmark annoté.
      -> Mesure si la collaboration améliore réellement la précision.
      -> Calcule le Facteur d'Amplification des Erreurs (FAE) et la Pénalité Séquentielle.

  8b. ÉVALUATION CONTREFACTUELLE
      Pour chaque article du benchmark, retire un agent et compare le signal obtenu
      avec le signal complet. Calcule la contribution marginale de chaque agent.
      -> Si retirer l'Agent Neutre ne change jamais le signal : il n'apporte rien.
      -> Si retirer l'Agent Baissier fait exploser les erreurs : il est crucial.

MÉTRIQUES :
  - Accuracy par configuration (SOLO / DEBATE_2 / DEBATE_3)
  - F1-Score pondéré par configuration
  - Facteur d'amplification des erreurs (FAE) : erreurs L3 qui n'existaient pas en L2
  - Pénalité séquentielle : dégradation progressive de la précision quand les
    composants s'enchaînent
  - Contribution marginale de chaque agent (contrefactuel)
  - FinGAIA Score composite : agrégation des 4 dimensions

NOTE SUR FinGAIA :
  FinGAIA (Yin et al., 2024) est un benchmark propriétaire non public.
  Ce module implémente une méthodologie inspirée, adaptée à notre architecture :
  comparaison multi-configuration sur le même corpus annoté.

Lancé via : python eval/run_eval.py --layer 8 [--limit 5]
"""

import asyncio
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("MultiAgentArch")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

BENCHMARK_PATH = Path(__file__).parent / "benchmark_dataset.json"
EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"

SIGNALS = ["Achat", "Vente", "Neutre"]


# ---------------------------------------------------------------------------
# Helpers métriques
# ---------------------------------------------------------------------------


def _accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 3)


def _f1_weighted(y_true: list, y_pred: list) -> float:
    """F1-Score pondéré macro sur les 3 classes de signal."""
    total = len(y_true)
    if total == 0:
        return 0.0
    weighted_f1 = 0.0
    for cls in SIGNALS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        weighted_f1 += f1 * (support / total)
    return round(weighted_f1, 3)


def _error_amplification_factor(
    errors_stage1: list[bool],
    errors_stage2: list[bool],
) -> float:
    """
    Facteur d'Amplification des Erreurs (FAE).

    Mesure combien d'erreurs de l'étape 2 n'existaient pas en étape 1.
    FAE = 1.0 -> les deux étapes font exactement les mêmes erreurs.
    FAE > 1.0 -> l'étape 2 AMPLIFIE les erreurs (chaque erreur S1 entraîne plus d'erreurs S2).
    FAE < 1.0 -> l'étape 2 CORRIGE certaines erreurs de S1.

    Args:
        errors_stage1 : liste de bool (True = erreur) pour la config S1
        errors_stage2 : liste de bool (True = erreur) pour la config S2

    Returns:
        FAE arrondi à 3 décimales.
    """
    n = len(errors_stage1)
    if n == 0:
        return 1.0

    n_err_s1 = sum(errors_stage1)
    n_err_s2 = sum(errors_stage2)

    if n_err_s1 == 0:
        # S1 parfait — si S2 fait des erreurs, c'est de l'amplification pure
        return round(n_err_s2 / (n + 1e-9), 3)

    # Erreurs en S2 QUI N'ÉTAIENT PAS des erreurs en S1
    new_errors = sum(1 for e1, e2 in zip(errors_stage1, errors_stage2) if not e1 and e2)
    fae = 1.0 + (new_errors / (n_err_s1 + 1e-9))

    return round(fae, 3)


def _sequential_penalty(accuracy_list: list[float]) -> float:
    """
    Pénalité séquentielle : mesure à quel point chaque étape supplémentaire
    dégrade les performances par rapport à la première.

    sequential_penalty = mean(max(0, acc[0] - acc[i]) for i in 1..N)
    -> 0 = pas de dégradation, 1 = dégradation totale.
    """
    if len(accuracy_list) <= 1:
        return 0.0
    baseline = accuracy_list[0]
    drops = [max(0.0, baseline - acc) for acc in accuracy_list[1:]]
    return round(sum(drops) / len(drops), 3)


# ---------------------------------------------------------------------------
# Chargement du benchmark
# ---------------------------------------------------------------------------


def _load_benchmark(limit: int = 0) -> list[dict]:
    """Charge les articles pertinents avec ground truth signal."""
    if not BENCHMARK_PATH.exists():
        logger.error("[MultiAgent] benchmark_dataset.json introuvable")
        return []
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        data = json.load(f)
    articles = [
        a
        for a in data["articles"]
        if a["ground_truth"]["filtrage"] == "pertinent" and a["ground_truth"].get("signal") in SIGNALS
    ]
    if limit > 0:
        articles = articles[:limit]
    return articles


# ---------------------------------------------------------------------------
# 8a. Benchmark comparatif multi-configurations
# ---------------------------------------------------------------------------


def _run_solo_agent(article_text: str, ticker: str) -> str:
    """
    Configuration SOLO : un seul agent LLM (Haussier) décide.
    Pas de débat, pas de consensus — réponse directe du modèle haussier.
    """
    try:
        from autogen_core.models import SystemMessage, UserMessage

        from src.agents.agent_debat import _get_model_client

        client, _ = _get_model_client("cerebras")  # L'agent Haussier

        prompt = f"""Tu es un analyste financier senior spécialisé dans l'analyse de sentiment.
Analyse cet article financier et donne ton signal d'investissement.
Réponds UNIQUEMENT avec un mot : 'Achat', 'Vente' ou 'Neutre'.

ARTICLE :
{article_text[:2000]}

SIGNAL :"""

        response = asyncio.run(
            client.create(
                messages=[
                    SystemMessage(content="Tu es un analyste financier expert. Réponds avec UN seul mot."),
                    UserMessage(content=prompt, source="user"),
                ]
            )
        )
        content = response.content.strip()
        for signal in SIGNALS:
            if signal.lower() in content.lower():
                return signal
        return "Neutre"
    except Exception as e:
        logger.warning("[SOLO] Erreur : %s", e)
        return "Neutre"


def _run_debate_2_agents(article_text: str, ticker: str) -> str:
    """
    Configuration DEBATE_2 : Haussier + Baissier, sans Agent Neutre.
    Le consensus est leur vote majoritaire direct (pas de modérateur).
    """
    try:
        from autogen_core.models import SystemMessage, UserMessage

        from src.agents.agent_debat import _get_model_client

        prompt_template = """Tu es {role}.
Analyse cet article et donne ton signal : 'Achat', 'Vente' ou 'Neutre'.
Réponds avec UN seul mot.

ARTICLE :
{article}

SIGNAL :"""

        signals = []
        for role, provider in [("Analyste Haussier convaincu", "cerebras"), ("Analyste Baissier sceptique", "groq")]:
            client, _ = _get_model_client(provider)
            prompt = prompt_template.format(role=role, article=article_text[:1500])
            try:
                response = asyncio.run(
                    client.create(
                        messages=[
                            SystemMessage(content="Analyse financière. UN seul mot de réponse."),
                            UserMessage(content=prompt, source="user"),
                        ]
                    )
                )
                content = response.content.strip()
                for s in SIGNALS:
                    if s.lower() in content.lower():
                        signals.append(s)
                        break
                else:
                    signals.append("Neutre")
            except Exception as e:
                logger.debug("[DEBATE_2] Erreur agent %s: %s", role[:10], e)
                signals.append("Neutre")

        # Majorité simple sur 2 agents (en cas d'égalité -> Neutre)
        if len(signals) == 2 and signals[0] == signals[1]:
            return signals[0]
        return "Neutre"  # Désaccord -> Neutre par défaut (pas de modérateur)

    except Exception as e:
        logger.warning("[DEBATE_2] Erreur : %s", e)
        return "Neutre"


def _run_debate_3_agents(article_text: str, ticker: str) -> str:
    """
    Configuration DEBATE_3 : Architecture complète (Haussier + Baissier + Neutre).
    Utilise le workflow de débat complet existant.
    """
    try:
        from src.agents.agent_debat import workflow_debat_actualite

        result = workflow_debat_actualite.invoke(
            {
                "texte_article": article_text,
                "ticker_symbol": ticker,
                "contexte_marche": {},
                "absa_result": {"aspects": []},
            }
        )
        signal = result.get("signal", "Neutre")
        return signal if signal in SIGNALS else "Neutre"
    except Exception as e:
        logger.warning("[DEBATE_3] Erreur : %s", e)
        return "Neutre"


# ---------------------------------------------------------------------------
# Helpers Financiers
# ---------------------------------------------------------------------------
from eval.evaluate_abnormal_returns import run_event_study
from eval.evaluate_market import _compute_trade_result, _get_close_price, _sharpe


def run_fingaia_benchmark(limit: int = 0) -> dict:
    """
    8a — Benchmark comparatif des configurations d'agents.
    Teste 3 architectures sur le même corpus et compare leurs performances.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 8a : Benchmark Comparatif (FinGAIA-inspired)")
    print(f"{'=' * 70}")
    print("  Configurations : SOLO | DEBATE_2 (2 agents) | DEBATE_3 (3 agents)")

    articles = _load_benchmark(limit)
    if not articles:
        print("  [INFO] Aucun article dans le benchmark. Vérifiez benchmark_dataset.json.")
        return {}

    print(f"\n  {len(articles)} articles à évaluer sur 3 configurations.\n")
    print("  Note : 3 appels LLM consécutifs par article — soyez patient.\n")

    configs = {
        "SOLO": {"fn": _run_solo_agent, "results": [], "y_pred": [], "returns": [], "cars": []},
        "DEBATE_2": {"fn": _run_debate_2_agents, "results": [], "y_pred": [], "returns": [], "cars": []},
        "DEBATE_3": {"fn": _run_debate_3_agents, "results": [], "y_pred": [], "returns": [], "cars": []},
    }
    y_true = []

    print(f"  {'ID':<18} {'GT':<8} {'SOLO':<10} {'DEBATE_2':<12} {'DEBATE_3':<10}")
    print(f"  {'-' * 60}")

    for art in articles:
        gt_signal = art["ground_truth"]["signal"]
        article_text = f"{art.get('title', '')}\n\n{art['content']}"
        ticker = art.get("ticker", "UNKNOWN")
        date_str = art.get("date", "2024-01-01")
        y_true.append(gt_signal)

        # Prix réels pour P&L
        p0 = _get_close_price(ticker, date_str, 0)
        p5 = _get_close_price(ticker, date_str, 5)

        row_preds = {}
        for cfg_name, cfg in configs.items():
            pred = cfg["fn"](article_text, ticker)
            cfg["y_pred"].append(pred)
            row_preds[cfg_name] = pred

            # Calcul Performance (P&L avec slippage)
            if p0 is not None and p5 is not None:
                tr = _compute_trade_result(pred, p0, p5)
                cfg["returns"].append(tr["return_pct"])

            # Calcul CAR (si le signal != Neutre)
            if pred != "Neutre":
                car_res = run_event_study(ticker, date_str, pred)
                if car_res.get("car") is not None:
                    # Si pred == Vente, le CAR doit être inversé pour la strat
                    val = car_res["car"] if pred == "Achat" else -car_res["car"]
                    cfg["cars"].append(val)

        def mark(pred, gt):
            return f"{pred}[OK]" if pred == gt else f"{pred}[KO]"

        print(
            f"  {art['id']:<18} {gt_signal:<8} "
            f"{mark(row_preds['SOLO'], gt_signal):<10} "
            f"{mark(row_preds['DEBATE_2'], gt_signal):<12} "
            f"{mark(row_preds['DEBATE_3'], gt_signal):<10}"
        )

    # Métriques par configuration
    print(f"\n  {'=' * 70}")
    print("  RÉSULTATS PAR CONFIGURATION (IA & FINANCE)")
    print(f"  {'=' * 70}")
    print(f"  {'Config':<12} {'Acc':>6} {'F1':>6} {'P&L(%)':>8} {'Sharpe':>7} {'CAR(%)':>8}")
    print(f"  {'-' * 55}")

    config_metrics = {}
    for cfg_name, cfg in configs.items():
        acc = _accuracy(y_true, cfg["y_pred"])
        f1 = _f1_weighted(y_true, cfg["y_pred"])
        n_err = sum(1 for t, p in zip(y_true, cfg["y_pred"]) if t != p)

        # Finance
        pl = sum(cfg["returns"]) if cfg["returns"] else 0.0
        shp = _sharpe(cfg["returns"]) if cfg["returns"] else 0.0
        avg_car = (sum(cfg["cars"]) / len(cfg["cars"]) * 100) if cfg["cars"] else 0.0

        config_metrics[cfg_name] = {
            "accuracy": acc,
            "f1": f1,
            "n_errors": n_err,
            "errors": [t != p for t, p in zip(y_true, cfg["y_pred"])],
            "pl": pl,
            "sharpe": shp,
            "avg_car": avg_car,
        }
        print(f"  {cfg_name:<12} {acc:>6.0%} {f1:>6.2f} {pl:>+8.2f} {shp:>7.2f} {avg_car:>+8.2f}")

    # Différentiels (Delta)
    delta_pl = config_metrics["DEBATE_3"]["pl"] - config_metrics["SOLO"]["pl"]
    delta_shp = config_metrics["DEBATE_3"]["sharpe"] - config_metrics["SOLO"]["sharpe"]
    delta_car = config_metrics["DEBATE_3"]["avg_car"] - config_metrics["SOLO"]["avg_car"]

    # Facteur d'amplification des erreurs (SOLO -> DEBATE_3)
    fae_solo_to_d3 = _error_amplification_factor(
        config_metrics["SOLO"]["errors"],
        config_metrics["DEBATE_3"]["errors"],
    )
    fae_d2_to_d3 = _error_amplification_factor(
        config_metrics["DEBATE_2"]["errors"],
        config_metrics["DEBATE_3"]["errors"],
    )

    # Pénalité séquentielle
    accuracy_sequence = [
        config_metrics["DEBATE_3"]["accuracy"],  # Architecture cible
        config_metrics["DEBATE_2"]["accuracy"],  # Version allégée
        config_metrics["SOLO"]["accuracy"],  # Version minimale
    ]
    seq_penalty = _sequential_penalty(accuracy_sequence)

    # FinGAIA Score composite (inspiré de l'approche originale)
    # 4 dimensions : Accuracy (40%), F1 (30%), pas d'amplification (15%), pas de pénalité (15%)
    d3_acc = config_metrics["DEBATE_3"]["accuracy"]
    d3_f1 = config_metrics["DEBATE_3"]["f1"]
    fae_score = max(0, 1.0 - abs(fae_solo_to_d3 - 1.0))  # 1 = neutre, <1 = amplifie
    pen_score = max(0, 1.0 - seq_penalty)
    fingaia_score = round(0.40 * d3_acc + 0.30 * d3_f1 + 0.15 * fae_score + 0.15 * pen_score, 3)

    print("\n  GAINS MULTI-AGENT (DEBATE_3 vs SOLO) :")
    print(f"  Delta Performance (P&L) : {delta_pl:>+6.2f}%")
    print(f"  Delta Ratio de Sharpe   : {delta_shp:>+6.2f}")
    print(f"  Delta CAR Moyen         : {delta_car:>+6.2f}%")

    print("\n  MÉTRIQUES D'ARCHITECTURE :")
    print(
        f"  FAE (SOLO -> DEBATE_3)    : {fae_solo_to_d3:.3f}  "
        f"({'Correction d erreurs' if fae_solo_to_d3 < 1 else 'Amplification' if fae_solo_to_d3 > 1.2 else 'Neutre'})"
    )
    print(
        f"  FAE (DEBATE_2 -> DEBATE_3): {fae_d2_to_d3:.3f}  "
        f"({'Le 3e agent corrige' if fae_d2_to_d3 < 1 else 'Le 3e agent amplifie' if fae_d2_to_d3 > 1.2 else 'Neutre'})"
    )
    print(
        f"  Pénalité séquentielle    : {seq_penalty:.3f}  "
        f"({'Faible' if seq_penalty < 0.10 else 'Modérée' if seq_penalty < 0.20 else 'Élevée'})"
    )
    print(f"\n  FinGAIA Score composite  : {fingaia_score:.3f}/1.000")

    # Verdict
    best_cfg = max(config_metrics, key=lambda c: config_metrics[c]["accuracy"])
    if best_cfg == "DEBATE_3":
        verdict = "L'architecture complète 3 agents surpasse les configurations réduites."
    elif best_cfg == "DEBATE_2":
        verdict = "DEBATE_2 surpasse DEBATE_3 — le 3e agent (Neutre) introduit du bruit."
    else:
        verdict = "SOLO surpasse les architectures de débat — les agents se contredisent inutilement."

    print(f"\n  Verdict : {verdict}")
    print("=" * 70)

    return {
        "sub": "fingaia_benchmark",
        "n_articles": len(articles),
        "configs": {
            k: {
                "accuracy": v["accuracy"],
                "f1": v["f1"],
                "n_errors": v["n_errors"],
                "pl": v["pl"],
                "sharpe": v["sharpe"],
                "avg_car": v["avg_car"],
            }
            for k, v in config_metrics.items()
        },
        "deltas": {"pl": delta_pl, "sharpe": delta_shp, "car": delta_car},
        "fae_solo_to_d3": fae_solo_to_d3,
        "fae_d2_to_d3": fae_d2_to_d3,
        "sequential_penalty": seq_penalty,
        "fingaia_score": fingaia_score,
        "best_config": best_cfg,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 8b. Évaluation contrefactuelle — Contribution marginale de chaque agent
# ---------------------------------------------------------------------------


def _run_without_agent(article_text: str, ticker: str, agent_to_remove: str) -> str:
    """
    Lance un débat simplifié sans l'agent spécifié.
    Utilise une version allégée du workflow pour simuler l'absence d'un agent.
    """
    remaining_providers = {
        "Neutre": [("cerebras", "Haussier"), ("groq", "Baissier")],
        "Baissier": [("cerebras", "Haussier"), ("mistral", "Neutre")],
        "Haussier": [("groq", "Baissier"), ("mistral", "Neutre")],
    }.get(agent_to_remove, [])

    if not remaining_providers:
        return "Neutre"

    try:
        from autogen_core.models import SystemMessage, UserMessage

        from src.agents.agent_debat import _get_model_client

        signals = []
        for provider, role in remaining_providers:
            client, _ = _get_model_client(provider)
            prompt = (
                f"Tu es un analyste financier ({role}). "
                f"Donne ton signal pour cet article : 'Achat', 'Vente' ou 'Neutre'.\n\n"
                f"ARTICLE :\n{article_text[:1500]}\n\nSIGNAL :"
            )
            try:
                response = asyncio.run(
                    client.create(
                        messages=[
                            SystemMessage(content="Analyste financier. UN seul mot."),
                            UserMessage(content=prompt, source="user"),
                        ]
                    )
                )
                content = response.content.strip()
                for s in SIGNALS:
                    if s.lower() in content.lower():
                        signals.append(s)
                        break
                else:
                    signals.append("Neutre")
            except Exception:
                signals.append("Neutre")

        # Vote majoritaire simplifié
        if signals.count("Achat") > len(signals) // 2:
            return "Achat"
        if signals.count("Vente") > len(signals) // 2:
            return "Vente"
        return "Neutre"

    except Exception as e:
        logger.warning("[Contref.] Erreur sans %s: %s", agent_to_remove, e)
        return "Neutre"


def run_counterfactual_evaluation(limit: int = 0) -> dict:
    """
    8b — Évaluation contrefactuelle.
    Pour chaque article, retire un agent et mesure l'impact sur le signal.
    Calcule la contribution marginale de chaque agent.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 8b : Évaluation Contrefactuelle (Contribution Marginale des Agents)")
    print(f"{'=' * 70}")

    articles = _load_benchmark(limit=min(limit or 5, 5))  # Max 5 pour limiter les appels API
    if not articles:
        print("  [INFO] Aucun article dans le benchmark.")
        return {}

    print(f"\n  {len(articles)} articles × 4 configurations = {len(articles) * 4} appels LLM.")
    print("  Configurations : Full | Sans Neutre | Sans Baissier | Sans Haussier\n")

    agents_to_test = ["Neutre", "Baissier", "Haussier"]
    results = []

    # Dictionnaires pour stocker les rendements et CARs pour chaque config
    financials = {
        "Full": {"returns": [], "cars": []},
        "Neutre": {"returns": [], "cars": []},
        "Baissier": {"returns": [], "cars": []},
        "Haussier": {"returns": [], "cars": []},
    }

    print(f"  {'ID':<18} {'GT':<8} {'Full':<8} {'¬Neutre':<10} {'¬Baissier':<12} {'¬Haussier':<10}")
    print(f"  {'-' * 66}")

    for art in articles:
        gt = art["ground_truth"]["signal"]
        text = f"{art.get('title', '')}\n\n{art['content']}"
        ticker = art.get("ticker", "UNKNOWN")
        date_str = art.get("date", "2024-01-01")

        # Prix pour P&L
        p0 = _get_close_price(ticker, date_str, 0)
        p5 = _get_close_price(ticker, date_str, 5)

        # Signal complet (DEBATE_3)
        full_signal = _run_debate_3_agents(text, ticker)

        # Signaux contrefactuels
        cf_signals = {}
        for agent in agents_to_test:
            cf_signals[agent] = _run_without_agent(text, ticker, agent)

        def record_finance(cfg_key, pred):
            if p0 is not None and p5 is not None:
                tr = _compute_trade_result(pred, p0, p5)
                financials[cfg_key]["returns"].append(tr["return_pct"])
            if pred != "Neutre":
                car_res = run_event_study(ticker, date_str, pred)
                if car_res.get("car") is not None:
                    val = car_res["car"] if pred == "Achat" else -car_res["car"]
                    financials[cfg_key]["cars"].append(val)

        record_finance("Full", full_signal)
        for agent in agents_to_test:
            record_finance(agent, cf_signals[agent])

        def mark(pred, gt):
            return f"{pred}[OK]" if pred == gt else f"{pred}[KO]"

        print(
            f"  {art['id']:<18} {gt:<8} "
            f"{mark(full_signal, gt):<8} "
            f"{mark(cf_signals['Neutre'], gt):<10} "
            f"{mark(cf_signals['Baissier'], gt):<12} "
            f"{mark(cf_signals['Haussier'], gt):<10}"
        )

        results.append(
            {
                "id": art["id"],
                "gt": gt,
                "full": full_signal,
                "sans_neutre": cf_signals["Neutre"],
                "sans_baissier": cf_signals["Baissier"],
                "sans_haussier": cf_signals["Haussier"],
                "full_correct": full_signal == gt,
                "cf_neutre_correct": cf_signals["Neutre"] == gt,
                "cf_baissier_correct": cf_signals["Baissier"] == gt,
                "cf_haussier_correct": cf_signals["Haussier"] == gt,
            }
        )

    if not results:
        return {}

    n = len(results)
    full_acc = sum(r["full_correct"] for r in results) / n
    cf_accs = {
        "Neutre": sum(r["cf_neutre_correct"] for r in results) / n,
        "Baissier": sum(r["cf_baissier_correct"] for r in results) / n,
        "Haussier": sum(r["cf_haussier_correct"] for r in results) / n,
    }

    # Agrégation financière
    def agg_finance(cfg_key):
        pl = sum(financials[cfg_key]["returns"]) if financials[cfg_key]["returns"] else 0.0
        shp = _sharpe(financials[cfg_key]["returns"]) if financials[cfg_key]["returns"] else 0.0
        car = (
            (sum(financials[cfg_key]["cars"]) / len(financials[cfg_key]["cars"]) * 100)
            if financials[cfg_key]["cars"]
            else 0.0
        )
        return pl, shp, car

    full_pl, full_shp, full_car = agg_finance("Full")

    # Contribution marginale IA (Acc)
    marginal = {agent: round(full_acc - cf_accs[agent], 3) for agent in agents_to_test}

    # Contribution marginale Finance (Delta = Full - Sans Agent)
    # Ex: Si Full fait +10% et Sans Baissier fait +2%, la diff est +8% (le Baissier a rapporté +8%)
    marginal_finance = {}
    for agent in agents_to_test:
        pl, shp, car = agg_finance(agent)
        marginal_finance[agent] = {"pl": full_pl - pl, "sharpe": full_shp - shp, "car": full_car - car}

    # Taux de consensus (signal identique entre Full et Sans-Agent)
    consensus_rates = {
        "Neutre": sum(r["full"] == r["sans_neutre"] for r in results) / n,
        "Baissier": sum(r["full"] == r["sans_baissier"] for r in results) / n,
        "Haussier": sum(r["full"] == r["sans_haussier"] for r in results) / n,
    }

    print(f"\n  {'=' * 70}")
    print("  CONTRIBUTION MARGINALE PAR AGENT (ABLATION STUDY)")
    print(f"  {'=' * 70}")
    print(
        f"  {'Agent Retiré':<15} | {'Delta Acc':>8} | {'Delta P&L(%)':>10} | {'Delta Sharpe':>10} | {'Delta CAR(%)':>10}"
    )
    print(f"  {'-' * 70}")

    for agent in agents_to_test:
        delta_acc = marginal[agent]
        d_pl = marginal_finance[agent]["pl"]
        d_shp = marginal_finance[agent]["sharpe"]
        d_car = marginal_finance[agent]["car"]

        print(f"  {f'Sans {agent}':<15} | {delta_acc:>+8.1%} | {d_pl:>+9.2f}% | {d_shp:>+10.2f} | {d_car:>+9.2f}%")

    # Agent le plus critique
    most_critical = max(marginal, key=marginal.get)
    most_critical_finance = max(marginal_finance.keys(), key=lambda k: marginal_finance[k]["pl"])

    print(
        f"\n  Agent le plus critique (Précision) : {most_critical} (retrait -> {-marginal[most_critical]:.1%} d'accuracy)"
    )
    print(
        f"  Agent le plus rentable (Finance)   : {most_critical_finance} (retrait -> {-marginal_finance[most_critical_finance]['pl']:+.2f}% de P&L)"
    )
    print("=" * 70)

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = EVAL_RESULTS_DIR / f"{timestamp}_multiagent_arch"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "counterfactual.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "marginal_contributions": marginal,
                "marginal_finance": marginal_finance,
                "consensus_rates": {k: round(v, 3) for k, v in consensus_rates.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "sub": "counterfactual",
        "n_articles": n,
        "full_accuracy": round(full_acc, 3),
        "config_accuracies": {agent: round(v, 3) for agent, v in cf_accs.items()},
        "marginal_contributions": marginal,
        "marginal_finance": marginal_finance,
        "consensus_rates": {k: round(v, 3) for k, v in consensus_rates.items()},
        "most_critical_agent": most_critical,
    }


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_multiagent_arch_analysis(limit: int = 0) -> dict:
    """
    Lance l'analyse complète de l'architecture multi-agents.
    Combine le benchmark FinGAIA-inspired et l'évaluation contrefactuelle.

    Args:
        limit : Nombre d'articles du benchmark à évaluer (0 = tous)
    """
    print(f"\n{'#' * 70}")
    print("COUCHE 8 : Architecture Multi-Agents & Mise à l'Échelle (IA + FINANCE)")
    print(f"{'#' * 70}")

    result = {"layer": 8}

    # 8a — Benchmark comparatif
    try:
        result["fingaia"] = run_fingaia_benchmark(limit=limit)
    except Exception as e:
        print(f"\n  [ERREUR 8a] {e}")
        result["fingaia"] = {"error": str(e)}

    # 8b — Évaluation contrefactuelle
    try:
        result["counterfactual"] = run_counterfactual_evaluation(limit=limit)
    except Exception as e:
        print(f"\n  [ERREUR 8b] {e}")
        result["counterfactual"] = {"error": str(e)}

    # Résumé synthétique
    print(f"\n{'=' * 70}")
    print("RÉSUMÉ COUCHE 8 (PERFORMANCE ARCHITECTURE MULTI-AGENTS)")
    print(f"{'=' * 70}")
    if "fingaia" in result and "fingaia_score" in result.get("fingaia", {}):
        fg = result["fingaia"]
        print(f"  FinGAIA Score         : {fg['fingaia_score']:.3f}/1.000")
        print(f"  Meilleure config      : {fg.get('best_config', 'N/A')}")
        deltas = fg.get("deltas", {})
        print(f"  Gain Multi-Agent P&L  : {deltas.get('pl', 0.0):+.2f}%")
        print(f"  Gain Multi-Agent CAR  : {deltas.get('car', 0.0):+.2f}%")
        print(f"  FAE (Solo->3 agents)   : {fg['fae_solo_to_d3']:.3f}")
    if "counterfactual" in result and "most_critical_agent" in result.get("counterfactual", {}):
        cf = result["counterfactual"]
        print(f"  Agent le + critique   : {cf['most_critical_agent']}")
        mf = cf.get("marginal_finance", {})
        for agent, metrics in mf.items():
            print(f"    Delta P&L si {agent} est retiré : {-metrics['pl']:+.2f}%")
    print("=" * 70)

    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=3, help="Nombre d'articles du benchmark (défaut: 3)")
    a = p.parse_args()
    run_multiagent_arch_analysis(limit=a.limit)
