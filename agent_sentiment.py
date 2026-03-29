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
- The JSON must follow EXACTLY this structure:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining your choice in English>"
}}

SCORE GUIDELINES:
- score close to 1.0 = very strong signal (very bullish or very bearish)
- score close to 0.5 = weak or mixed signal
- sentiment "neutral" always has score between 0.45 and 0.55

EXAMPLES:

Article: "Apple reports record quarterly revenue, iPhone sales surge 15%"
Response: {{"sentiment": "bullish", "score": 0.91, "reasoning": "Record revenue and strong iPhone sales indicate excellent financial performance."}}

Article: "Microsoft faces antitrust investigation over cloud practices"
Response: {{"sentiment": "bearish", "score": 0.76, "reasoning": "Antitrust investigation creates regulatory risk and uncertainty for the company."}}

Article: "Google announces date for annual developer conference"
Response: {{"sentiment": "neutral", "score": 0.50, "reasoning": "Event announcement with no financial implications mentioned."}}

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
    score     = float(data.get("score", 0.0))
    reasoning = data.get("reasoning", "")

    if sentiment not in ("bullish", "bearish", "neutral"):
        raise ValueError(f"Sentiment invalide : {sentiment}")
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Score hors limites : {score}")
    if not reasoning:
        raise ValueError("Reasoning vide")

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
        "score":       score,
        "reasoning":   reasoning,
        "analyzed_at": datetime.now(timezone.utc).isoformat()
    }

    logger.info(f"Résultat → {sentiment} ({score}) | {reasoning}")
    return resultat
