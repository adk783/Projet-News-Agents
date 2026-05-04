"""
Agent de filtrage en cascade (2 étages).

Étage 2a : pré-filtre déterministe par keywords extraits de yfinance.Ticker.info
           (shortName, longName, ticker, dirigeants, domaine du site web).
           Coût ~0, élimine les articles trivialement pertinents/non pertinents.

Étage 2b : confirmation LLM (Ollama llama3.2:3b) avec prompt few-shot YES/NO.
           Appelé uniquement si le pré-filtre keywords ne tranche pas.

Provenance :
- build_keywords / is_relevant_by_keywords : branche `filtrage-keywords`
- prompt YES/NO + appel Ollama                : branche `Antoinev2`
"""

import re
import logging

import requests


# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("AgentFiltrage")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ─── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
MAX_CHARS = 500


# ─── ÉTAGE 2a : PRÉ-FILTRE KEYWORDS (filtrage-keywords) ───────────────────────
def build_keywords(stock_info):
    """
    Construit l'ensemble des mots-clés associés à un ticker à partir de
    yfinance.Ticker(t).info : noms de la société, ticker, dirigeants, domaine web.
    """
    keywords = set()

    for field in ["shortName", "longName"]:
        name = stock_info.get(field, "")
        if name:
            keywords.add(name.lower())
            cleaned = re.sub(r"[^a-zA-Z\s-]", " ", name)
            for word in cleaned.split():
                if len(word) > 3:
                    keywords.add(word.lower())

    symbol = stock_info.get("symbol", "")
    if symbol:
        keywords.add(symbol.lower())

    for officer in stock_info.get("companyOfficers", []) or []:
        name = officer.get("name", "")
        if name:
            keywords.add(name.lower())
            parts = name.split()
            if len(parts) >= 2:
                keywords.add(parts[-1].lower())

    website = stock_info.get("website", "")
    if website:
        domain = (
            website.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
        )
        keywords.add(domain.lower())
        if "." in domain:
            pure = domain.rsplit(".", 1)[0]
            if len(pure) > 1:
                keywords.add(pure.lower())

    return keywords


def is_relevant_by_keywords(title, keywords):
    """Cherche un keyword (frontière de mot) dans le titre."""
    if not title:
        return False
    title_lower = title.lower()
    for kw in keywords:
        if not kw:
            continue
        if re.search(r"\b" + re.escape(kw) + r"\b", title_lower):
            return True
    return False


# ─── ÉTAGE 2b : CONFIRMATION LLM (Antoinev2) ──────────────────────────────────
def build_prompt(ticker, company_name, title, content):
    return f"""You are a financial news filter. Decide if a news article is relevant to a specific company's stock.

Answer only YES or NO. Nothing else.

Rules:
- YES if the company name or ticker appears in the TITLE — this is the strongest signal, always prioritize it
- YES if the company is the main subject OR one of the main subjects of the article
- YES if the article discusses the company's products, strategy, earnings, stock, or executives
- YES if the article compares the company directly to a competitor
- NO only if the company is briefly mentioned in passing in an article about something else entirely
- NO if the article is about general market indexes, unrelated sectors, or other companies with no link to this one
- Do NOT be confused by broad topics like AI, markets, or macro trends in the content — if the company name is in the title, answer YES

Examples:
Ticker: TSLA | Title: "Tesla reports record deliveries in Q1" -> YES
Ticker: TSLA | Title: "Is Rivian a better bet than Tesla?" -> NO
Ticker: TSLA | Title: "Sector Update: Consumer Stocks Mixed Monday" -> NO
Ticker: AAPL | Title: "Apple launches new iPhone model" -> YES
Ticker: AAPL | Title: "Apple faces regulatory scrutiny over App Store practices" -> YES
Ticker: AAPL | Title: "Netflix Before Q1 Earnings: Should Investors Buy?" -> NO
Ticker: AAPL | Title: "3 AI Stocks That Are Way Cheaper Than Apple Right Now" -> NO
Ticker: MSFT | Title: "Microsoft Gains Despite OpenAI Partnership Tensions" -> YES
Ticker: MSFT | Title: "Stock market today: Dow rises as Oracle surges" -> NO
Ticker: GOOGL | Title: "Google expands AI features in Search" -> YES
Ticker: GOOGL | Title: "Intel climbs on Google partnership news" -> NO
Ticker: NVDA | Title: "Nvidia reports record data center revenue" -> YES
Ticker: META | Title: "Meta launches new AI assistant for WhatsApp" -> YES
Ticker: META | Title: "CoreWeave stock soars after Meta AI deal" -> NO

Now decide:
Ticker: {ticker} ({company_name})
Title: {title}
Content preview: {content[:MAX_CHARS]}

Answer (YES or NO):"""


def _appeler_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["response"].strip().upper()


def est_pertinent_llm(ticker, company_name, title, content):
    """
    Demande à llama3.2:3b si l'article est principalement sur le ticker.
    En cas d'erreur Ollama → laisse passer (failsafe, ne bloque pas le pipeline).
    """
    try:
        prompt = build_prompt(ticker, company_name, title, content)
        reponse = _appeler_ollama(prompt)

        if reponse.startswith("YES"):
            logger.debug(f"[Filtrage IA] PERTINENT : {title}")
            return True, "ia_pertinent"
        if reponse.startswith("NO"):
            logger.info(f"[Filtrage IA] HORS SUJET : {title}")
            return False, "ia_hors_sujet"
        logger.warning(
            f"[Filtrage IA] Reponse inattendue '{reponse}' pour : {title} -> laisse passer"
        )
        return True, "ia_reponse_inattendue"
    except Exception as e:
        logger.warning(f"[Filtrage IA] Erreur ({e}) -> laisse passer : {title}")
        return True, "ia_erreur"


# ─── ENTRÉE PRINCIPALE — CASCADE ───────────────────────────────────────────────
def est_pertinent(ticker, company_name, title, content, keywords=None):
    """
    Filtrage en cascade :
      1) Pré-filtre keywords sur le titre (coût ~0).
         - match → on bypass le LLM, l'article est gardé.
      2) Sinon → confirmation LLM YES/NO.

    Args:
        ticker        : symbole du ticker (ex: AAPL)
        company_name  : nom de la société (pour le prompt LLM)
        title         : titre de l'article
        content       : contenu de l'article
        keywords      : set de keywords (optionnel ; si fourni, étage 2a actif)

    Returns:
        (bool, str)  -> (gardé ?, motif)
    """
    if keywords and is_relevant_by_keywords(title, keywords):
        logger.debug(f"[Filtrage KW] PERTINENT (titre match) : {title}")
        return True, "kw_titre_match"

    return est_pertinent_llm(ticker, company_name, title, content)
