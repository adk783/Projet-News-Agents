import requests
import logging

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("AgentFiltrage")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

file_handler = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3.2:3b"
MAX_CHARS   = 500  # on passe juste le début de l'article, pas besoin de plus


# ─── PROMPT ────────────────────────────────────────────────────────────────────
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
Ticker: TSLA | Title: "Tesla FSD Win In Europe And Compact SUV Revival" -> YES
Ticker: TSLA | Title: "Gary Black Says Rumored Tesla Model Q Could Have Huge Upside" -> YES
Ticker: TSLA | Title: "Is Rivian a better bet than Tesla?" -> NO
Ticker: TSLA | Title: "Sector Update: Consumer Stocks Mixed Monday" -> NO
Ticker: AAPL | Title: "Apple launches new iPhone model" -> YES
Ticker: AAPL | Title: "Apple AI Glasses Are Taking Shape Slowly" -> YES
Ticker: AAPL | Title: "Warren Buffett reveals why Berkshire is dumping Apple stock" -> YES
Ticker: AAPL | Title: "Can Strong iPhone and Mac Portfolio Help Apple Stock Recover?" -> YES
Ticker: AAPL | Title: "The More AI Models Come Out the More I'm Convinced Apple Has the Right Strategy" | Content: "As more AI models emerge from various companies, Apple's unique approach..." -> YES
Ticker: AAPL | Title: "Apple's AI Glasses Are Taking Shape Slowly" | Content: "Apple is quietly developing AI-powered glasses to compete with Meta Ray-Ban..." -> YES
Ticker: AAPL | Title: "Warren Buffett Reveals the Real Reason Berkshire Has Been Dumping Apple Stock" | Content: "Berkshire Hathaway has significantly reduced its Apple position over recent quarters..." -> YES
Ticker: AAPL | Title: "Apple faces regulatory scrutiny over App Store practices" -> YES
Ticker: AAPL | Title: "Netflix Before Q1 Earnings: Should Investors Buy?" -> NO
Ticker: AAPL | Title: "Why Millions Are Buying the Wrong NASDAQ ETF" -> NO
Ticker: AAPL | Title: "3 AI Stocks That Are Way Cheaper Than Apple Right Now" -> NO
Ticker: AAPL | Title: "Amazon stock leads Magnificent 7 performance" -> NO
Ticker: MSFT | Title: "Microsoft Gains Despite OpenAI Partnership Tensions" -> YES
Ticker: MSFT | Title: "Microsoft reports strong cloud revenue growth" -> YES
Ticker: MSFT | Title: "Stock market today: Dow rises as Oracle surges" -> NO
Ticker: MSFT | Title: "US Equity Indexes Rise Amid Tech Gains" -> NO
Ticker: MSFT | Title: "CoreWeave Stock Surges on AI Deals" -> NO
Ticker: GOOGL | Title: "Google expands AI features in Search" -> YES
Ticker: GOOGL | Title: "Intel climbs on Google partnership news" -> NO
Ticker: NVDA | Title: "Nvidia reports record data center revenue" -> YES
Ticker: NVDA | Title: "Gladstone Capital dividend coverage analysis" -> NO
Ticker: META | Title: "Meta launches new AI assistant for WhatsApp" -> YES
Ticker: META | Title: "CoreWeave stock soars after Meta AI deal" -> NO
Ticker: NFLX | Title: "Netflix reports record subscriber growth in Q1" -> YES
Ticker: NFLX | Title: "Netflix Before Q1 Earnings: Should Investors Buy?" -> YES
Ticker: NFLX | Title: "3 Big Reasons Netflix Will Continue to Soar" -> YES
Ticker: NFLX | Title: "Netflix expands into new markets with Buenos Aires push" -> YES
Ticker: NFLX | Title: "Pre-Markets in Red on Renewed Middle East Geopolitical Conflicts" -> NO
Ticker: NFLX | Title: "Stock Futures Down as US Plans Blockade of Iranian Ports" -> NO

Now decide:
Ticker: {ticker} ({company_name})
Title: {title}
Content preview: {content[:MAX_CHARS]}

Answer (YES or NO):"""


# ─── APPEL OLLAMA ──────────────────────────────────────────────────────────────
def _appeler_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 5
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["response"].strip().upper()


# ─── FONCTION PRINCIPALE ───────────────────────────────────────────────────────
def est_pertinent(ticker, company_name, title, content):
    """
    Demande à llama3.2:3b si l'article est principalement sur le ticker.
    Retourne (True, raison) si pertinent, (False, raison) sinon.
    En cas d'erreur, retourne (True, ...) pour ne pas bloquer le pipeline.
    """
    try:
        prompt   = build_prompt(ticker, company_name, title, content)
        reponse  = _appeler_ollama(prompt)

        if reponse.startswith("YES"):
            logger.debug(f"[Filtrage IA] PERTINENT : {title}")
            return True, "ia_pertinent"
        elif reponse.startswith("NO"):
            logger.info(f"[Filtrage IA] HORS SUJET : {title}")
            return False, "ia_hors_sujet"
        else:
            # Réponse inattendue → on laisse passer par sécurité
            logger.warning(f"[Filtrage IA] Reponse inattendue '{reponse}' pour : {title} -> laisse passer")
            return True, "ia_reponse_inattendue"

    except Exception as e:
        # Si Ollama est down ou timeout → on laisse passer sans bloquer
        logger.warning(f"[Filtrage IA] Erreur ({e}) -> laisse passer : {title}")
        return True, "ia_erreur"
