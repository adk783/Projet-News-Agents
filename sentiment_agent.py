import requests
import json
import logging
from datetime import datetime, timezone

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("AgentSentiment")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

file_handler = logging.FileHandler("orchestrateur.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "phi4-mini"
MAX_CHARS   = 1500   # On tronque le contenu pour ne pas surcharger le modèle


# ─── FORMATTAGE DES FEATURES (agents amont) ───────────────────────────────────
def _format_features_block(features):
    """
    Transforme le dict de features produit par les 4 agents amont en bloc texte
    annoté (échelle + interprétation) à injecter dans le prompt phi4.
    Retourne une chaîne vide si features est None (rétrocompat / fallback).
    """
    if not features:
        return ""

    polarity = features.get("polarity")
    polarity_conf = features.get("polarity_conf")
    uncertainty = features.get("uncertainty")
    legal_risk = features.get("legal_risk")
    fundamental = features.get("fundamental_strength")

    # Lecture qualitative pour aider le LLM
    def _label_polarity(p, c):
        if p is None or c is None:
            return "unavailable"
        name = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}.get(int(p), "UNKNOWN")
        conf = "high" if c >= 0.7 else ("medium" if c >= 0.5 else "low")
        return f"{name} (confidence {conf})"

    def _label_uncertainty(u):
        if u is None:
            return "unavailable"
        return "HIGH (noisy signal)" if u > 0.5 else ("MODERATE" if u > 0.3 else "LOW")

    def _label_legal(l):
        if l is None:
            return "unavailable"
        return "MATERIAL legal exposure" if l > 0.5 else ("SOME legal context" if l > 0.2 else "none")

    def _label_fund(f):
        if f is None:
            return "unavailable"
        if f > 0.3:
            return "POSITIVE fundamentals"
        if f < -0.3:
            return "WEAK fundamentals"
        return "neutral fundamentals"

    def _fmt(v, prec=2):
        return "n/a" if v is None else f"{v:.{prec}f}"

    return f"""
UPSTREAM AGENT SIGNALS (use as priors to inform your decision):

- FinBERT polarity: {polarity if polarity is not None else "n/a"} (confidence {_fmt(polarity_conf)})
  Scale: -1 bearish / 0 neutral / +1 bullish. Confidence > 0.7 = trustworthy.
  Reading: {_label_polarity(polarity, polarity_conf)}.

- Uncertainty (Loughran-McDonald lexicon): {_fmt(uncertainty)}
  Scale: 0 = confident text, 1 = highly uncertain. Above 0.4 = noisy signal.
  Reading: {_label_uncertainty(uncertainty)}.

- Legal/Litigious risk (L&M lexicon): {_fmt(legal_risk)}
  Scale: 0 = no legal context, 1 = heavy litigation vocabulary. Above 0.3 = material.
  Reading: {_label_legal(legal_risk)}.

- Fundamental strength (lexicon): {_fmt(fundamental)}
  Scale: -1 weak / +1 strong fundamentals language. |score| > 0.3 = meaningful.
  Reading: {_label_fund(fundamental)}.

RECONCILIATION RULES:
- If your reading of the article AGREES with FinBERT (high confidence) and
  fundamentals are aligned → keep your score, possibly raise it slightly.
- If FinBERT (confidence > 0.7) STRONGLY CONTRADICTS your initial reading
  → lean toward neutral or lower your conviction score.
- If legal_risk > 0.5 → cap any bullish score at 0.5; consider neutral if
  the legal exposure dominates the narrative.
- If uncertainty > 0.5 → cap conviction; prefer neutral when text is vague.
- Your `reasoning` MUST cite at least one upstream agent signal explicitly
  (e.g. "FinBERT confirms...", "low uncertainty supports...", "legal risk caps...").
"""


# ─── PROMPT ────────────────────────────────────────────────────────────────────
def build_prompt(ticker, title, content, features=None):
    """
    Construit le prompt envoyé à phi4-mini.
    Si `features` est fourni (dict produit par les 4 agents amont), il est
    injecté dans le prompt avec annotation (échelle + interprétation), de
    sorte que phi4-mini joue le rôle d'arbitre final informé par l'archi.
    """

    features_block = _format_features_block(features)

    prompt = f"""You are a financial news sentiment analyst working as the FINAL ARBITER in a multi-agent pipeline. Four specialized models have already analyzed this article. Your job is to reconcile their signals with your own reading of the text and produce the final structured sentiment score.

RULES:
- You MUST respond ONLY with a valid JSON object, nothing else.
- No explanation, no introduction, no markdown, no backticks.

If the article has a clear financial impact on the ticker, use bullish or bearish:
{{"sentiment": "bullish", "score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
{{"sentiment": "bearish", "score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

If the article has NO direct financial impact (event announcements, general news, unrelated topics), or if upstream signals are too contradictory to call, use neutral:
{{"sentiment": "neutral", "score": null, "reasoning": "<one sentence>"}}

SCORING APPROACH:
- The TITLE carries 50% of the signal. If the title says "Gains", "Surges", "Beats", "Win", "Approval" → lean bullish. If it says "Falls", "Drops", "Misses", "Problem", "Dumping" → lean bearish.
- The CONTENT carries the other 50% — use it to confirm and calibrate the score.
- If title and content contradict each other, use neutral.
- If the article is mainly about another company and the ticker is just mentioned as a partner or context → use neutral.

SCORE GUIDELINES (bullish and bearish only, never null):
- 0.90 - 1.00 : very strong signal — record earnings, major acquisition, catastrophic news
- 0.70 - 0.89 : strong signal — solid earnings beat, significant product launch, regulatory approval
- 0.50 - 0.69 : mild signal — moderate beat/miss, minor news, indirect impact, analyst opinion
- Max 0.70 if the signal comes from a third party (analyst, fund manager, partner company)

EXAMPLES (no upstream signals shown for brevity):

Article: "Apple reports record quarterly revenue, iPhone sales surge 15%"
Response: {{"sentiment": "bullish", "score": 0.93, "reasoning": "Record revenue and strong iPhone sales indicate excellent financial performance; FinBERT confirms bullish with high confidence."}}

Article: "Apple slightly beats earnings estimates but guidance disappoints"
Response: {{"sentiment": "bullish", "score": 0.58, "reasoning": "Modest earnings beat offset by weak guidance creates a mixed but slightly positive signal."}}

Article: "Can Strong iPhone and Mac Portfolio Help Apple Stock Recover in FY26?"
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Article discusses recovery potential but presents no clear directional financial signal."}}

Article: "Microsoft Gains Despite OpenAI Partnership Tensions"
Response: {{"sentiment": "bullish", "score": 0.60, "reasoning": "Stock price gains despite tensions signals resilient investor confidence; low uncertainty supports the call."}}

Article: "Tesla faces massive recall of 2 million vehicles over safety defect"
Response: {{"sentiment": "bearish", "score": 0.88, "reasoning": "Large-scale recall creates significant financial liability; legal risk reading reinforces bearish stance."}}

Article: "Tesla FSD gets Dutch regulatory approval, sets stage for EU rollout"
Response: {{"sentiment": "bullish", "score": 0.75, "reasoning": "First European regulatory approval for FSD opens new market; FinBERT polarity aligned bullish."}}

RECONCILIATION EXAMPLES (with conflicting upstream signals):

Article: "Tesla's record deliveries beat all estimates" — Upstream: FinBERT +1 (0.88), legal_risk 0.65 (pending recall lawsuit)
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Strong delivery beat is offset by material legal exposure flagged by L&M legal risk score 0.65; net signal unclear."}}

Article: "Apple announces strategic restructuring" — Upstream: FinBERT 0 (0.62), uncertainty 0.71
Response: {{"sentiment": "neutral", "score": null, "reasoning": "FinBERT polarity is flat and uncertainty is high (0.71), text too vague to assert a direction."}}

Article: "Microsoft cloud revenue beats expectations slightly" — Upstream: FinBERT +1 (0.81), uncertainty 0.18, fundamental_strength +0.55
Response: {{"sentiment": "bullish", "score": 0.72, "reasoning": "FinBERT bullish high-conf, low uncertainty and strong fundamentals (+0.55) all align with the headline beat."}}

Article: "Nvidia chip pricing under SEC review, settlement rumored" — Upstream: FinBERT -1 (0.79), legal_risk 0.82
Response: {{"sentiment": "bearish", "score": 0.85, "reasoning": "FinBERT bearish high-conf reinforced by very high legal risk (0.82) — clear regulatory and litigation overhang."}}
{features_block}
---

Now analyze this article:

Ticker: {ticker}
Title: {title}
Content: {content[:MAX_CHARS]}

Response:"""

    return prompt


# ─── APPEL OLLAMA ──────────────────────────────────────────────────────────────
class OllamaUnavailableError(RuntimeError):
    """Levée quand Ollama est injoignable ou que le modèle de sentiment manque."""


def appeler_ollama(prompt):
    """
    Envoie le prompt à Ollama et retourne la réponse texte brute.
    Lève OllamaUnavailableError avec un message explicite si Ollama est down
    ou si le modèle requis est manquant.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 150,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    except requests.exceptions.ConnectionError as e:
        raise OllamaUnavailableError(
            f"Ollama injoignable sur {OLLAMA_URL} ({e}). "
            f"Lance `ollama serve` dans un autre terminal."
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaUnavailableError(
            f"Ollama timeout sur {OLLAMA_URL} ({e})."
        ) from e

    if response.status_code == 404:
        raise OllamaUnavailableError(
            f"Modèle '{MODEL_NAME}' non trouvé dans Ollama. "
            f"Lance `ollama pull {MODEL_NAME}`."
        )

    response.raise_for_status()
    return response.json()["response"].strip()


# ─── PARSING DE LA RÉPONSE ─────────────────────────────────────────────────────
def parser_reponse(reponse_brute):
    """
    Parse le JSON retourné par le modèle.
    Gère les cas où le modèle ajoute du texte avant/après le JSON.
    """
    # Nettoyage au cas où le modèle met des backticks ou du texte autour
    reponse_nettoyee = reponse_brute.strip()
    if "```" in reponse_nettoyee:
        reponse_nettoyee = reponse_nettoyee.split("```")[1]
        if reponse_nettoyee.startswith("json"):
            reponse_nettoyee = reponse_nettoyee[4:]

    # Extraction du JSON entre { et }
    debut = reponse_nettoyee.find("{")
    fin   = reponse_nettoyee.rfind("}") + 1
    if debut == -1 or fin == 0:
        raise ValueError(f"Aucun JSON trouvé dans la réponse : {reponse_brute}")

    json_str = reponse_nettoyee[debut:fin]
    data = json.loads(json_str)

    # Validation des champs obligatoires
    sentiment = data.get("sentiment", "").lower()
    score_raw = data.get("score", None)
    reasoning = data.get("reasoning", "")

    if sentiment not in ("bullish", "bearish", "neutral"):
        raise ValueError(f"Sentiment invalide : {sentiment}")
    if not reasoning:
        raise ValueError("Reasoning vide")

    if sentiment == "neutral":
        score = None
    else:
        if score_raw is None:
            logger.warning(f"Score null reçu pour sentiment {sentiment}, fallback 0.70")
            score = 0.70
        else:
            score = float(score_raw)
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score hors limites : {score}")

    return sentiment, score, reasoning


# ─── FONCTION PRINCIPALE ───────────────────────────────────────────────────────
def analyser_article(url, ticker, title, content, features=None):
    """
    Analyse un article et retourne un dictionnaire avec le résultat.
    C'est cette fonction que l'orchestrateur appelle.

    `features` : dict optionnel produit par les 4 agents amont
        {polarity, polarity_conf, uncertainty, legal_risk, fundamental_strength}
    Si fourni, ces signaux sont injectés dans le prompt phi4 avec annotation
    (échelle, interprétation, règles de réconciliation) — phi4 joue alors le
    rôle d'arbitre final cohérent avec l'architecture multi-agents.
    Si None, fallback sur le prompt classique (rétrocompat).
    """
    logger.debug(f"Analyse de : {title}")

    prompt        = build_prompt(ticker, title, content, features=features)
    reponse_brute = appeler_ollama(prompt)

    logger.debug(f"Réponse brute Ollama : {reponse_brute}")

    sentiment, score, reasoning = parser_reponse(reponse_brute)

    resultat = {
        "url":         url,
        "ticker":      ticker,
        "sentiment":   sentiment,
        "score":       score,  # None si neutral
        "reasoning":   reasoning,
        "analyzed_at": datetime.now(timezone.utc).isoformat()
    }

    score_disp = f"{score:.2f}" if score is not None else "null"
    logger.info(f"Résultat → {sentiment} ({score_disp}) | {reasoning}")
    return resultat
