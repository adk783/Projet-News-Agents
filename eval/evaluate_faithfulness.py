"""
evaluate_faithfulness.py -- Couche 6a : Faithfulness, NLI & Answer Relevancy
=============================================================================
Deux métriques complémentaires sur le module Synthèse (Agent Consensus) :

  1. FAITHFULNESS (NLI) — L'argument invente-t-il des faits ?
     Le juge décompose l'argument en prémisses atomiques et vérifie
     que chacune est dérivable (Entailment) de l'article source.
     Score : 0 (hallucination) / 1 (fidèle)

  2. ANSWER RELEVANCY — L'argument répond-il à la bonne question ?
     Vérifie que l'argument adresse la question financière implicite
     de l'article (signal d'investissement, impact sectoriel, risque
     macro…) et ne se contente pas d'être vrai mais hors-sujet.
     Score continu : 0.0 (hors-sujet) → 1.0 (parfaitement pertinent)
     Dimensions évaluées :
       - Complétude  : couvre-t-on le point clé de l'article ?
       - Actionnabilité : le signal Achat/Vente/Neutre est-il justifié ?
       - Concision   : l'argument est-il focalisé sans dilution ?

Lancé via : python eval/run_eval.py --layer 6 --sub faithfulness
"""

import json
import logging
import sqlite3
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()
from src.agents.agent_debat import _get_model_client

logger = logging.getLogger("EvalFaithfulness")
logger.setLevel(logging.WARNING)

DATABASE_PATH = "data/news_database.db"

# ---------------------------------------------------------------------------
# Setup LLM Evaluateur
# ---------------------------------------------------------------------------


def _get_judge_client():
    client, _ = _get_model_client("consensus")
    return client


JUDGE_SYSTEM_PROMPT = """Tu es un expert en vérification d'informations financières et en Inférence en Langage Naturel (NLI).
Ta mission est STRICTE : vérifier si l'Argument du Consensus dérive EXACTEMENT de l'Article d'origine, ou si des faits ont été inventés (hallucinations).

Règles de NLI :
1. Découpe l'Argument du Consensus en prémisses atomiques (ex: 'L'entreprise a gagné 10M', 'Ils ont signé avec Apple').
2. Pour CHAQUE prémisse, cherche la phrase PREUVE dans l'article.
3. Si une seule prémisse ajoute un fait NON MENTIONNÉ (même s'il est vrai dans le monde réel), c'est une HALLUCINATION. Les conclusions logiques évidentes sont permises, mais PAS l'invention de chiffres, partenariats, ou variations de pourcentages.

Réponds TOUJOURS avec un objet JSON strictement formaté comme suit :
{
    "premises": [
        {"claim": "La phrase atomique", "status": "Entailment|Contradiction|Neutral"}
    ],
    "is_faithful": true / false,
    "hallucination_reason": "Si false, explique brièvement quelle information a été inventée."
}
"""


# ---------------------------------------------------------------------------
# Answer Relevancy — Pertinence de l'argument par rapport à la question
# ---------------------------------------------------------------------------

RELEVANCY_SYSTEM_PROMPT = """Tu es un expert en analyse financière et en évaluation de la pertinence des réponses.
Ta mission : évaluer si l'ARGUMENT DU CONSENSUS répond correctement à la QUESTION FINANCIÈRE IMPLICITE de l'article.

Une bonne réponse doit :
  1. Adresser le POINT CLE financier de l'article (résultats, acquisition, macro, guidances...)
  2. Justifier le signal d'investissement (Achat / Vente / Neutre) avec des raisons concrètes
  3. Être FOCALISÉ — ne pas dériver vers des généralités hors-sujet

Tu dois répondre UNIQUEMENT avec ce JSON :
{
    "financial_question": "La question implicite que l'article pose à un investisseur (1 phrase)",
    "completeness": 0.0,
    "actionability": 0.0,
    "conciseness": 0.0,
    "relevancy_score": 0.0,
    "missing_dimensions": ["liste des aspects financiers non adressés"],
    "verdict": "PERTINENT|PARTIEL|HORS_SUJET"
}

Où chaque score est entre 0.0 et 1.0 :
  - completeness  : L'argument couvre-t-il le (les) point(s) clés de l'article ?
  - actionability : Le signal (Achat/Vente/Neutre) est-il clairement justifié par des éléments factuels ?
  - conciseness   : L'argument est-il focalisé (pas de dilution générique) ?
  - relevancy_score: Moyenne pondérée (0.4 * completeness + 0.4 * actionability + 0.2 * conciseness)

Verdict :
  - PERTINENT  : relevancy_score >= 0.70
  - PARTIEL    : 0.40 <= relevancy_score < 0.70
  - HORS_SUJET : relevancy_score < 0.40
"""


async def evaluate_relevancy_single(
    article_content: str,
    consensus_argument: str,
    signal: str = "",
) -> dict:
    """
    Évalue la pertinence de l'argument par rapport à la question
    financière implicite de l'article.

    Args:
        article_content    : Titre + corps de l'article
        consensus_argument : Argument produit par l'Agent Consensus
        signal             : Signal final (Achat/Vente/Neutre) pour contexte

    Returns:
        Dict avec relevancy_score [0.0, 1.0], dimensions, verdict
    """
    if not consensus_argument or consensus_argument.strip() == "Parsing impossible":
        return {
            "relevancy_score": 0.0,
            "verdict": "HORS_SUJET",
            "missing_dimensions": ["Argument absent"],
            "financial_question": "N/A",
        }

    client = _get_judge_client()
    signal_ctx = f" (signal prédit : {signal})" if signal else ""
    task = (
        f"=== ARTICLE SOURCE ===\n{article_content[:3000]}\n\n"
        f"=== ARGUMENT DU CONSENSUS{signal_ctx} ===\n{consensus_argument}\n\n"
        f"Évalue la pertinence en JSON."
    )

    import re

    from autogen_core.models import SystemMessage, UserMessage

    try:
        response = await client.create(
            messages=[SystemMessage(content=RELEVANCY_SYSTEM_PROMPT), UserMessage(content=task, source="user")]
        )
        content = response.content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group().strip())
            # Recalcul défensif du score si le LLM ne l'a pas bien pondéré
            comp = float(parsed.get("completeness", 0.0))
            acti = float(parsed.get("actionability", 0.0))
            conc = float(parsed.get("conciseness", 0.0))
            parsed["relevancy_score"] = round(0.4 * comp + 0.4 * acti + 0.2 * conc, 3)
            # Recalcul du verdict selon le score
            score = parsed["relevancy_score"]
            parsed["verdict"] = "PERTINENT" if score >= 0.70 else "PARTIEL" if score >= 0.40 else "HORS_SUJET"
            return parsed
        else:
            return {
                "relevancy_score": 0.0,
                "verdict": "HORS_SUJET",
                "error": "Format JSON non respecté par l'évaluateur",
                "raw_response": content[:200],
            }
    except Exception as e:
        return {
            "relevancy_score": 0.0,
            "verdict": "HORS_SUJET",
            "error": str(e),
        }


async def evaluate_faithfulness_single(article_content: str, consensus_argument: str) -> dict:
    """Utilise l'Evaluateur pour vérifier un seul article."""
    if not consensus_argument or consensus_argument.strip() == "Parsing impossible":
        return {"is_faithful": False, "premises": [], "hallucination_reason": "Aucun argument à évaluer."}

    client = _get_judge_client()
    task = (
        f"=== ARTICLE SOURCE ===\n{article_content}\n\n"
        f"=== ARGUMENT DU CONSENSUS A VERIFIER ===\n{consensus_argument}\n\n"
        f"Génère ton analyse en JSON."
    )

    from autogen_core.models import SystemMessage, UserMessage

    try:
        response = await client.create(
            messages=[SystemMessage(content=JUDGE_SYSTEM_PROMPT), UserMessage(content=task, source="user")]
        )
        content = response.content.strip()
        import re

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            clean_content = json_match.group().strip()
            return json.loads(clean_content)
        else:
            return {
                "is_faithful": False,
                "hallucination_reason": "Format JSON non respecte par l'evaluateur.",
                "raw_response": content,
            }
    except Exception as e:
        return {"error": str(e), "is_faithful": False, "hallucination_reason": f"Erreur système: {e}"}


# ---------------------------------------------------------------------------
# Point d'Entrée
# ---------------------------------------------------------------------------


def run_faithfulness_analysis(limit: int = 20) -> dict:
    """
    Analyse complète du module Synthèse :
      - Faithfulness (NLI) : l'argument invente-t-il des faits ?
      - Answer Relevancy   : l'argument répond-il à la bonne question financière ?
    """
    import asyncio

    print(f"\n{'=' * 70}")
    print("COUCHE 6a : Faithfulness (NLI) & Answer Relevancy")
    print(f"{'=' * 70}")

    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT url, ticker, title, content, argument_dominant, signal_final
            FROM articles
            WHERE argument_dominant IS NOT NULL
              AND argument_dominant != ''
              AND argument_dominant != 'Parsing impossible'
            ORDER BY date_utc DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"\n  [ERREUR] Lecture SQLite: {e}")
        return {}

    if not rows:
        print("  Aucun article éligible trouvé en base.")
        return {}

    print(f"\n  {len(rows)} arguments à auditer (2 appels LLM / article).")
    print(f"  {'Ticker':<8} {'Faithful':<12} {'Relevancy':<12} {'Verdict':<12} {'Hallucination / Manques'}")
    print(f"  {'-' * 68}")

    results = []

    for row in tqdm(rows, desc="  Audit Synthèse", ncols=80):
        article_text = f"{row['title']}\n\n{row['content']}"
        argument = row["argument_dominant"]
        signal = row["signal_final"] or ""

        # ---- 1. Faithfulness (NLI) ----
        faith_res = asyncio.run(evaluate_faithfulness_single(article_text, argument))

        # ---- 2. Answer Relevancy ----
        rel_res = asyncio.run(evaluate_relevancy_single(article_text, argument, signal))

        combined = {
            "url": row["url"],
            "ticker": row["ticker"],
            "signal": signal,
            # Faithfulness
            "is_faithful": faith_res.get("is_faithful", False),
            "hallucination_reason": faith_res.get("hallucination_reason", ""),
            "premises": faith_res.get("premises", []),
            # Answer Relevancy
            "relevancy_score": rel_res.get("relevancy_score", 0.0),
            "completeness": rel_res.get("completeness", 0.0),
            "actionability": rel_res.get("actionability", 0.0),
            "conciseness": rel_res.get("conciseness", 0.0),
            "verdict": rel_res.get("verdict", "HORS_SUJET"),
            "financial_question": rel_res.get("financial_question", ""),
            "missing_dimensions": rel_res.get("missing_dimensions", []),
        }
        results.append(combined)

        ticker_str = combined["ticker"]
        faith_str = "OUI" if combined["is_faithful"] else "NON"
        rel_str = f"{combined['relevancy_score']:.2f}"
        verdict_str = combined["verdict"]
        issue_str = (
            combined["hallucination_reason"][:25]
            if not combined["is_faithful"]
            else ", ".join(combined["missing_dimensions"])[:25]
            if combined["missing_dimensions"]
            else "—"
        )
        tqdm.write(f"  {ticker_str:<8} {faith_str:<12} {rel_str:<12} {verdict_str:<12} {issue_str}")

    # ---------------------------------------------------------------------------
    # Agrégation
    # ---------------------------------------------------------------------------
    total = len(results)
    faithful_count = sum(1 for r in results if r.get("is_faithful") is True)
    rel_scores = [r["relevancy_score"] for r in results]
    avg_relevancy = round(sum(rel_scores) / total, 3) if total else 0.0
    n_pertinent = sum(1 for r in results if r["verdict"] == "PERTINENT")
    n_partiel = sum(1 for r in results if r["verdict"] == "PARTIEL")
    n_hors_sujet = sum(1 for r in results if r["verdict"] == "HORS_SUJET")

    # Score composite Synthèse = 60% Faithfulness + 40% Relevancy
    faithfulness_rate = faithful_count / total if total else 0.0
    synthesis_score = round(0.60 * faithfulness_rate + 0.40 * avg_relevancy, 3)

    print(f"\n{'=' * 70}")
    print(f"RESUME — {total} articles audités")
    print(f"{'=' * 70}")
    print("\n  [Faithfulness NLI]")
    print(f"  Arguments fidèles     : {faithful_count}/{total} ({faithfulness_rate:.0%})")
    print(f"  Taux d'hallucination  : {total - faithful_count}/{total} ({1 - faithfulness_rate:.0%})")

    if total > faithful_count:
        fails = [r for r in results if not r["is_faithful"] and r["hallucination_reason"]]
        print("  Exemples :")
        for fail in fails[:3]:
            print(f"    [{fail['ticker']}] {fail['hallucination_reason'][:80]}")

    print("\n  [Answer Relevancy]")
    print(f"  Score moyen           : {avg_relevancy:.3f} / 1.00")
    print(f"  PERTINENT (>= 0.70)   : {n_pertinent}/{total} ({n_pertinent / total:.0%})")
    print(f"  PARTIEL   (0.40-0.70) : {n_partiel}/{total} ({n_partiel / total:.0%})")
    print(f"  HORS_SUJET (< 0.40)   : {n_hors_sujet}/{total} ({n_hors_sujet / total:.0%})")

    # Dimensions manquantes les plus fréquentes
    all_missing = []
    for r in results:
        all_missing.extend(r.get("missing_dimensions", []))
    if all_missing:
        from collections import Counter

        top_missing = Counter(all_missing).most_common(5)
        print("  Dimensions manquantes les plus fréquentes :")
        for dim, n in top_missing:
            print(f"    - {dim} ({n}x)")

    print("\n  [Score Composite Synthèse]")
    print("  synthesis_score = 0.60×Faithfulness + 0.40×Relevancy")
    print(f"  = 0.60×{faithfulness_rate:.2f} + 0.40×{avg_relevancy:.2f} = {synthesis_score:.3f}")
    verdict_global = (
        "EXCELLENT"
        if synthesis_score >= 0.80
        else "BON"
        if synthesis_score >= 0.65
        else "MOYEN"
        if synthesis_score >= 0.50
        else "INSUFFISANT"
    )
    print(f"  Verdict global        : {verdict_global}")

    print("=" * 70)

    return {
        "sub": "faithfulness",
        "total_audited": total,
        "faithful_count": faithful_count,
        "faithfulness_score": round(faithfulness_rate, 3),
        "avg_relevancy": avg_relevancy,
        "n_pertinent": n_pertinent,
        "n_partiel": n_partiel,
        "n_hors_sujet": n_hors_sujet,
        "synthesis_score": synthesis_score,
        "verdict_global": verdict_global,
        "details": results,
    }


if __name__ == "__main__":
    run_faithfulness_analysis(limit=5)
