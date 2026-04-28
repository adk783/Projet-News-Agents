"""
evaluate_unit_llm.py — Couche 7a : Benchmarks unitaires des agents LLM
=======================================================================
Vérifie qu'un agent individuel comprend correctement le langage financier
AVANT de travailler en équipe, en le testant sur des benchmarks (FinancialPhraseBank, FiQA, etc.)

Métriques calculées :
  - F1-Score pondéré (macro) par signal (Achat / Vente / Neutre)
  - Indice Kappa de Cohen (κ) : accord au-delà du hasard, anti-biais
  - Divergence de Kullback-Leibler (KL) : détecte les biais systématiques
    de distribution (ex: agent trop optimiste = sur-représentation "Achat")

Approche :
  Interroger directement chaque agent LLM sur l'ensemble des JSONL du dossier data/.

Lancé via : python eval/run_eval.py --layer 7 --sub7 unit
"""

import asyncio
import json
import logging
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("EvalUnitLLM")
logger.setLevel(logging.WARNING)

DATA_DIR = Path(__file__).parent / "lm_tasks" / "data"
RESULTS_DIR = Path(__file__).parent / "eval_results"
SIGNAL_CLASSES = ["positive", "negative", "neutral"]

agents_to_test = [
    {"name": "Haussier", "provider": "cerebras"},
    {"name": "Baissier", "provider": "groq"},
    {"name": "Neutre", "provider": "mistral"},
    {"name": "Consensus", "provider": "consensus"},
]

UNIT_CLASSIFICATION_PROMPT = """Tu es un analyste financier expert.
Classifie le sentiment de la phrase suivante.
Réponds UNIQUEMENT par un mot strict : 'positive', 'negative' ou 'neutral'.

PHRASE : {sentence}
SENTIMENT :"""


async def _classify_single(client, sentence: str) -> str:
    from autogen_core.models import SystemMessage, UserMessage

    prompt = UNIT_CLASSIFICATION_PROMPT.format(sentence=sentence)
    try:
        response = await client.create(
            messages=[
                SystemMessage(content="Tu es un classificateur financier expert. Réponds avec un seul mot."),
                UserMessage(content=prompt, source="user"),
            ]
        )
        content = response.content.strip().lower()
        if "positive" in content:
            return "positive"
        if "negative" in content:
            return "negative"
        if "neutral" in content:
            return "neutral"
        return "neutral"
    except Exception as e:
        logger.warning(f"Classify error: {e}")
        return "neutral"


def _f1_weighted(y_true: list, y_pred: list) -> dict:
    total = len(y_true)
    results = {}
    macro_f1 = 0.0

    for cls in SIGNAL_CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        weight = support / total if total > 0 else 0.0

        results[cls] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": support}
        macro_f1 += f1 * weight

    results["weighted_f1"] = round(macro_f1, 3)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    results["accuracy"] = round(correct / total, 3) if total > 0 else 0.0
    return results


def _cohen_kappa(y_true: list, y_pred: list) -> float:
    classes = list(set(y_true + y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    count_true = Counter(y_true)
    count_pred = Counter(y_pred)
    pe = sum((count_true.get(c, 0) / n) * (count_pred.get(c, 0) / n) for c in classes)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return round(kappa, 3)


def _kl_divergence(y_true: list, y_pred: list) -> dict:
    eps = 1e-9
    n = len(y_true)
    if n == 0:
        return {"kl_divergence": 0.0, "bias_class": "Aucun"}

    p_dist = {c: (y_true.count(c) / n + eps) for c in SIGNAL_CLASSES}
    q_dist = {c: (y_pred.count(c) / n + eps) for c in SIGNAL_CLASSES}
    kl = sum(p_dist[c] * math.log(p_dist[c] / q_dist[c]) for c in SIGNAL_CLASSES)
    bias_class = max(SIGNAL_CLASSES, key=lambda c: q_dist[c] - p_dist[c])
    bias_dir = q_dist[bias_class] - p_dist[bias_class]

    return {
        "kl_divergence": round(kl, 4),
        "bias_class": bias_class if abs(bias_dir) > 0.05 else "Aucun",
        "bias_direction": "sur-représenté" if bias_dir > 0 else "sous-représenté",
        "p_distribution": {c: round(p_dist[c], 3) for c in SIGNAL_CLASSES},
        "q_distribution": {c: round(q_dist[c], 3) for c in SIGNAL_CLASSES},
    }


def run_unit_llm_benchmark(limit: int = 0) -> dict:
    print(f"\n{'=' * 70}")
    print("COUCHE 7a : Benchmark Unitaire LLM — Multi-Agent & Multi-Dataset")
    print(f"{'=' * 70}")

    datasets = list(DATA_DIR.glob("*.jsonl"))
    if not datasets:
        print("Aucun dataset .jsonl trouvé. Lancez d'abord setup_benchmarks.py.")
        return {"error": "no_dataset"}

    from src.agents.agent_debat import _get_model_client

    all_results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RESULTS_DIR / f"{timestamp}_unit_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_acc_sum = 0
    total_f1_sum = 0
    total_kappa_sum = 0
    total_kl_sum = 0
    eval_count = 0

    for agent_config in agents_to_test:
        agent_name = agent_config["name"]
        provider = agent_config["provider"]
        client, model_id = _get_model_client(provider)

        print(f"\n>> Évaluation de l'Agent: {agent_name} [{model_id}]")
        agent_metrics = {}

        for dataset_path in datasets:
            print(f"  - Dataset: {dataset_path.name}")
            data = []
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))

            if limit > 0:
                data = data[:limit]

            y_true = []
            y_pred = []

            for item in tqdm(data, desc=f"    {dataset_path.name[:15]}...", ncols=80, leave=False):
                gt = item["label"].lower()
                sentence = item["sentence"]
                pred = asyncio.run(_classify_single(client, sentence))

                y_true.append(gt)
                y_pred.append(pred)

            f1_mets = _f1_weighted(y_true, y_pred)
            kappa = _cohen_kappa(y_true, y_pred)
            kl_info = _kl_divergence(y_true, y_pred)

            agent_metrics[dataset_path.name] = {
                "accuracy": f1_mets["accuracy"],
                "weighted_f1": f1_mets["weighted_f1"],
                "cohen_kappa": kappa,
                "kl_divergence": kl_info["kl_divergence"],
                "bias": kl_info["bias_class"],
            }

            total_acc_sum += f1_mets["accuracy"]
            total_f1_sum += f1_mets["weighted_f1"]
            total_kappa_sum += kappa
            total_kl_sum += kl_info["kl_divergence"]
            eval_count += 1

            print(
                f"    -> Acc: {f1_mets['accuracy']:.0%} | F1: {f1_mets['weighted_f1']:.3f} | k: {kappa:.3f} | KL: {kl_info['kl_divergence']:.4f}"
            )

        all_results[agent_name] = {"model_id": model_id, "datasets": agent_metrics}

    with open(out_dir / "unit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats unitaires détaillés sauvegardés dans : {out_dir}/unit_metrics.json")

    return {
        "sub": "unit_llm",
        "weighted_f1": total_f1_sum / eval_count if eval_count else 0,
        "cohen_kappa": total_kappa_sum / eval_count if eval_count else 0,
        "accuracy": total_acc_sum / eval_count if eval_count else 0,
        "kl_divergence": total_kl_sum / eval_count if eval_count else 0,
    }


if __name__ == "__main__":
    run_unit_llm_benchmark()
