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

EXAMPLES:

Article: "Apple reports record quarterly revenue, iPhone sales surge 15%"
Response: {{"sentiment": "bullish", "score": 0.93, "reasoning": "Record revenue and strong iPhone sales indicate excellent financial performance."}}

Article: "Apple slightly beats earnings estimates but guidance disappoints"
Response: {{"sentiment": "bullish", "score": 0.58, "reasoning": "Modest earnings beat offset by weak guidance creates a mixed but slightly positive signal."}}

Article: "Can Strong iPhone and Mac Portfolio Help Apple Stock Recover in FY26?"
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Article discusses recovery potential but presents no clear directional financial signal."}}

Article: "Microsoft Gains Despite OpenAI Partnership Tensions"
Response: {{"sentiment": "bullish", "score": 0.60, "reasoning": "Stock price gains despite partnership tensions signals resilient investor confidence."}}

Article: "Microsoft stock down 21% this year — options strategy to buy at discount"
Response: {{"sentiment": "bearish", "score": 0.70, "reasoning": "Significant YTD decline driven by slowing Azure growth and AI spending concerns."}}

Article: "Tesla faces massive recall of 2 million vehicles over safety defect"
Response: {{"sentiment": "bearish", "score": 0.88, "reasoning": "Large-scale recall creates significant financial liability and reputational damage."}}

Article: "Tesla FSD gets Dutch regulatory approval, sets stage for EU rollout"
Response: {{"sentiment": "bullish", "score": 0.75, "reasoning": "First European regulatory approval for FSD opens new market and software monetization opportunities."}}

Article: "Wall Street analyst rode in Tesla robotaxi, revealed 1 problem Elon Musk must overcome"
Response: {{"sentiment": "bearish", "score": 0.60, "reasoning": "Identified technological challenges in robotaxi efficiency signal execution risk."}}

Article: "Gary Black says rumored Tesla Model Q could have huge upside but warns about marketing"
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Speculative future opportunity with identified weakness — no direct financial signal yet."}}

Article: "Alphabet is among the high-flying AI stocks to buy — new Broadcom TPU deal"
Response: {{"sentiment": "bullish", "score": 0.80, "reasoning": "Concrete long-term AI infrastructure deal with Broadcom signals strong growth trajectory."}}

Article: "Microsoft misses cloud revenue target by a small margin"
Response: {{"sentiment": "bearish", "score": 0.61, "reasoning": "Minor cloud miss signals slight deceleration in growth but no major structural issue."}}

Article: "I'm not buying Nvidia right now — 2 other growth stocks are smarter plays"
Response: {{"sentiment": "bearish", "score": 0.55, "reasoning": "Analyst recommends alternatives over Nvidia, suggesting limited near-term upside."}}

Article: "Nvidia denies acquisition rumor that sparked Dell and HP rally"
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Denial of unconfirmed rumor with no direct financial impact on Nvidia."}}

Article: "Intel climbs 5% on landmark Google partnership"
Ticker: GOOGL
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Article is primarily about Intel's stock performance; Google is mentioned as a partner only."}}

Article: "Apple stock outperforms HP and Microsoft year to date despite overall decline"
Response: {{"sentiment": "neutral", "score": null, "reasoning": "Relative outperformance without a clear directional catalyst is not a strong financial signal."}}

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
