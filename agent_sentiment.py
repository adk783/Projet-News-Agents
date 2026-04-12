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


# ─── PROMPT ────────────────────────────────────────────────────────────────────
def build_prompt(ticker, title, content):
    """
    Construit le prompt envoyé à phi4-mini.
    On lui donne le contexte, des exemples (few-shot), et le format JSON attendu.
    """

    prompt = f"""You are a financial news sentiment analyst. Your job is to analyze a financial news article and return a structured sentiment score.

RULES:
- You MUST respond ONLY with a valid JSON object, nothing else.
- No explanation, no introduction, no markdown, no backticks.

If the article has a clear financial impact on the ticker, use bullish or bearish:
{{"sentiment": "bullish", "score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
{{"sentiment": "bearish", "score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

If the article has NO direct financial impact (event announcements, general news, unrelated topics), use neutral:
{{"sentiment": "neutral", "score": null, "reasoning": "<one sentence>"}}

SCORE GUIDELINES (bullish and bearish only, never null):
- 0.90 - 1.00 : very strong signal — record earnings, major acquisition, catastrophic news
- 0.70 - 0.89 : strong signal — solid earnings beat, significant product launch, regulatory risk
- 0.50 - 0.69 : mild signal — moderate beat/miss, minor news, indirect impact

EXAMPLES:

Article: "Apple reports record quarterly revenue, iPhone sales surge 15%"
Response: {{"sentiment": "bullish", "score": 0.93, "reasoning": "Record revenue and strong iPhone sales indicate excellent financial performance."}}

Article: "Apple slightly beats earnings estimates but guidance disappoints"
Response: {{"sentiment": "bullish", "score": 0.58, "reasoning": "Modest earnings beat offset by weak guidance creates a mixed but slightly positive signal."}}

Article: "Tesla faces massive recall of 2 million vehicles over safety defect"
Response: {{"sentiment": "bearish", "score": 0.88, "reasoning": "Large-scale recall creates significant financial liability and reputational damage."}}

Article: "Microsoft misses cloud revenue target by a small margin"
Response: {{"sentiment": "bearish", "score": 0.61, "reasoning": "Minor cloud miss signals slight deceleration in growth but no major structural issue."}}

Article: "Google announces date for annual developer conference"
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Event announcement with no direct financial implications."}}

---

Now analyze this article:

Ticker: {ticker}
Title: {title}
Content: {content[:MAX_CHARS]}

Response:"""

    return prompt


# ─── APPEL OLLAMA ──────────────────────────────────────────────────────────────
def appeler_ollama(prompt):
    """
    Envoie le prompt à Ollama et retourne la réponse texte brute.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   
            "top_p": 0.9,
            "num_predict": 150   
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
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
def analyser_article(url, ticker, title, content):
    """
    Analyse un article et retourne un dictionnaire avec le résultat.
    C'est cette fonction que l'orchestrateur appelle.
    """
    logger.debug(f"Analyse de : {title}")

    prompt        = build_prompt(ticker, title, content)
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
