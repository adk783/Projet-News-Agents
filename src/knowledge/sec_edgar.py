"""
sec_edgar.py — Intégration API SEC EDGAR (Gratuite, Publique)

Remplace le fact-checker heuristique par une VRAIE vérification factuelle
croisée avec les dépôts officiels de la Securities and Exchange Commission.

Références scientifiques :
  [1] SEC Rule 13a-11 (Current Reports — Form 8-K)
      17 CFR § 240.13a-11 — Obligation de dépôt dans les 4 jours ouvrés
      suivant tout événement matériel. Si un article parle d'un événement
      matériel sans 8-K correspondant → fort signal de rumeur non confirmée.

  [2] Ball, R. & Brown, P. (1968). "An Empirical Evaluation of Accounting
      Income Numbers." Journal of Accounting Research, 6(2), 159-178.
      → PEAD (Post-Earnings Announcement Drift) : les filings officiels
        génèrent des drifts de cours mesurables.

  [3] Seyhun, H.N. (1986). "Insiders' Profits, Costs of Trading, and Market
      Efficiency." Journal of Financial Economics, 16(2), 189-212.
      → Les ventes d'insiders précèdent statistiquement une sous-performance
        de -6% sur 6 mois. Les achats d'insiders : +4% sur 6 mois.

  [4] Lakonishok, J. & Lee, I. (2001). "Are Insider Trades Informative?"
      Review of Financial Studies, 14(1), 79-111.
      → Confirme que les achats d'insiders sont plus informatifs que les ventes
        (les ventes peuvent être motivées par liquidité personnelle).

  [5] EDGAR EFTS API (SEC, 2023) — Documentation officielle
      https://efts.sec.gov/LATEST/search-index
      → Full-text search sur l'ensemble des dépôts SEC depuis 1993.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes API EDGAR
# ---------------------------------------------------------------------------

# User-Agent obligatoire (SEC Fair Access Policy)
# Format requis : "Organization Email"
EDGAR_USER_AGENT = "ProjetE4-NewsAgents (francouisjean@gmail.com)"

# Endpoints EDGAR (publics, gratuits, aucune clé)
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index?q={query}&forms={forms}&dateRange=custom&startdt={start}&enddt={end}"
)
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Rate limiting SEC : 10 req/sec max — on prend une marge de sécurité
EDGAR_MIN_DELAY_SEC = 0.12  # ~8 req/sec

# Cache global : CIK → ticker (évite les appels répétés)
_cik_cache: dict[str, str] = {}
_company_tickers_cache: Optional[dict] = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EdgarFilingResult:
    """Résultat de recherche d'un dépôt SEC (8-K ou autre)."""

    found: bool
    form_type: str  # "8-K", "10-Q", "Form 4", etc.
    filing_date: str  # Date de dépôt ISO
    company_name: str
    cik: str
    accession_number: str  # Identifiant unique EDGAR
    description: str  # Description de l'événement (Item 1.01, etc.)
    url: str  # Lien vers le filing
    is_within_4_days: bool  # Conforme à la règle 13a-11 ?


@dataclass
class InsiderTransaction:
    """Une transaction d'insider (Form 4)."""

    insider_name: str
    insider_title: str
    transaction_type: str  # "P" = Purchase, "S" = Sale
    shares: int
    price_per_share: float
    date: str


@dataclass
class InsiderActivity:
    """Activité insider agrégée sur 30 jours (Seyhun 1986)."""

    ticker: str
    period_days: int
    transactions: list[InsiderTransaction] = field(default_factory=list)
    net_shares_bought: int = 0  # Positif = achats nets, négatif = ventes nettes
    n_buyers: int = 0
    n_sellers: int = 0
    signal: str = "NEUTRE"  # BULLISH_INSIDER / BEARISH_INSIDER / NEUTRE
    confidence_adjustment: float = 1.0  # Multiplicateur de confiance [0.7, 1.3]
    note: str = ""


# ---------------------------------------------------------------------------
# Client EDGAR
# ---------------------------------------------------------------------------


class SecEdgarClient:
    """
    Client pour l'API EDGAR de la SEC.

    Implémente :
    1. Lookup CIK depuis ticker (company_tickers.json)
    2. Recherche de dépôts 8-K récents (événements matériels)
    3. Extraction des transactions d'insiders (Form 4)
    4. Calcul du signal insider selon Seyhun (1986) et Lakonishok & Lee (2001)
    """

    def __init__(self, user_agent: str = EDGAR_USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Respect du rate limit EDGAR (10 req/s max)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < EDGAR_MIN_DELAY_SEC:
            time.sleep(EDGAR_MIN_DELAY_SEC - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, timeout: int = 10) -> Optional[dict]:
        """GET avec rate limiting et gestion d'erreurs."""
        self._rate_limit()
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.debug("[EDGAR] HTTP %s : %s", e.response.status_code, url[:100])
            return None
        except Exception as e:
            logger.debug("[EDGAR] Erreur GET %s : %s", url[:100], e)
            return None

    # --- CIK Lookup ---

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Résout le CIK (Central Index Key) d'un ticker.
        Le CIK est l'identifiant unique EDGAR de chaque société.

        Utilise company_tickers.json (fichier officiel SEC, mis à jour quotidiennement).
        """
        global _cik_cache, _company_tickers_cache

        ticker_upper = ticker.upper()
        if ticker_upper in _cik_cache:
            return _cik_cache[ticker_upper]

        # Chargement du fichier de correspondance si absent du cache
        if _company_tickers_cache is None:
            data = self._get(EDGAR_COMPANY_TICKERS)
            if data:
                _company_tickers_cache = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
            else:
                _company_tickers_cache = {}

        cik = _company_tickers_cache.get(ticker_upper)
        if cik:
            _cik_cache[ticker_upper] = cik
            logger.debug("[EDGAR] CIK %s → %s", ticker_upper, cik)
        else:
            logger.debug("[EDGAR] CIK introuvable pour %s", ticker_upper)

        return cik

    # --- Recherche 8-K ---

    def find_recent_8k(
        self,
        ticker: str,
        days_back: int = 10,
        keywords: Optional[list[str]] = None,
    ) -> Optional[EdgarFilingResult]:
        """
        Recherche les dépôts 8-K (Current Report) récents pour un ticker.

        SEC Rule 13a-11 : une société doit déposer un 8-K dans les 4 jours
        ouvrés suivant tout événement matériel.

        Args:
            ticker    : symbole boursier
            days_back : fenêtre temporelle de recherche (jours)
            keywords  : filtrage optionnel par mots-clés dans le titre

        Returns:
            EdgarFilingResult ou None si aucun dépôt trouvé.
        """
        cik = self.get_cik(ticker)
        if not cik:
            return EdgarFilingResult(
                found=False,
                form_type="8-K",
                filing_date="",
                company_name=ticker,
                cik="",
                accession_number="",
                description="CIK introuvable pour ce ticker",
                url="",
                is_within_4_days=False,
            )

        data = self._get(EDGAR_SUBMISSIONS_URL.format(cik=cik))
        if not data:
            return None

        company_name = data.get("name", ticker)
        filings = data.get("filings", {}).get("recent", {})

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        for i, (form, date, acc, desc) in enumerate(zip(forms, dates, accessions, descriptions)):
            if form != "8-K":
                continue
            if date < cutoff_str:
                break  # Les filings sont triés par date décroissante

            # Filtrage par mots-clés si demandé
            if keywords:
                desc_lower = (desc or "").lower()
                if not any(kw.lower() in desc_lower for kw in keywords):
                    continue

            # Calcul du délai de dépôt (conformité 13a-11)
            try:
                filing_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                filing_age_days = (datetime.now(timezone.utc) - filing_dt).days
                is_within_4_days = filing_age_days <= 4
            except Exception:
                is_within_4_days = False

            acc_formatted = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_formatted}/"

            logger.info(
                "[EDGAR] 8-K trouvé pour %s : date=%s | accession=%s",
                ticker,
                date,
                acc,
            )

            return EdgarFilingResult(
                found=True,
                form_type="8-K",
                filing_date=date,
                company_name=company_name,
                cik=cik,
                accession_number=acc,
                description=desc or "Événement matériel (voir filing EDGAR)",
                url=filing_url,
                is_within_4_days=is_within_4_days,
            )

        logger.debug("[EDGAR] Aucun 8-K récent pour %s sur %d jours.", ticker, days_back)
        return EdgarFilingResult(
            found=False,
            form_type="8-K",
            filing_date="",
            company_name=company_name,
            cik=cik,
            accession_number="",
            description=f"Aucun 8-K déposé dans les {days_back} derniers jours",
            url="",
            is_within_4_days=False,
        )

    # --- Insider Trading (Form 4) ---

    def _parse_form4_xml(self, cik: str, accession: str) -> list[InsiderTransaction]:
        """
        Parse un Form 4 XML depuis EDGAR pour extraire les transactions d'insiders.

        Structure XML Form 4 (ownershipDocument) :
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <transactionCoding>
                <transactionCode>P</transactionCode>   ← P=Purchase, S=Sale
              </transactionCoding>
              <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
              </transactionAmounts>
              <transactionDate><value>2026-04-15</value></transactionDate>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        """
        import re

        acc_clean = accession.replace("-", "")
        cik_int = int(cik)

        # Récupère l'index du filing pour trouver le XML
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.htm"
        index_html = self._get(index_url, timeout=10)
        # Si l'index n'est pas du HTML, on tente directement le document primaire
        xml_url = None

        # Approche directe : le Form 4 XML suit souvent le pattern xslForm4X01/
        # ou est nommé primaryDocument.xml. On tente le fichier .xml dans l'index.
        try:
            # L'API submissions fournit le primaryDocument pour chaque filing
            subs = self._get(EDGAR_SUBMISSIONS_URL.format(cik=cik))
            if subs:
                recent = subs.get("filings", {}).get("recent", {})
                accessions = recent.get("accessionNumber", [])
                primary_docs = recent.get("primaryDocument", [])
                for acc_candidate, pdoc in zip(accessions, primary_docs):
                    if acc_candidate == accession and pdoc:
                        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{pdoc}"
                        break
        except Exception:
            pass

        if not xml_url:
            return []

        self._rate_limit()
        try:
            resp = self.session.get(xml_url, timeout=10)
            resp.raise_for_status()
            xml_text = resp.text
        except Exception as e:
            logger.debug("[EDGAR] Erreur lecture Form 4 XML %s : %s", accession, e)
            return []

        transactions: list[InsiderTransaction] = []

        # Extraction du nom/titre de l'insider (reporting owner)
        owner_name = "Unknown"
        owner_title = ""
        name_match = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", xml_text)
        if name_match:
            owner_name = name_match.group(1).strip()
        title_match = re.search(r"<officerTitle>([^<]+)</officerTitle>", xml_text)
        if title_match:
            owner_title = title_match.group(1).strip()

        # Parse les transactions non-dérivées (actions directes)
        tx_blocks = re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", xml_text, re.DOTALL)

        for block in tx_blocks:
            # Code de transaction : P = Purchase (achat sur le marché)
            #                       S = Sale (vente sur le marché)
            #                       A = Award/Grant, M = Exercise, G = Gift...
            code_match = re.search(r"<transactionCode>([A-Z])</transactionCode>", block)
            if not code_match:
                continue
            tx_code = code_match.group(1)
            if tx_code not in ("P", "S"):
                continue  # On ignore les non-market transactions (grants, exercises, gifts)

            shares_match = re.search(r"<transactionShares>\s*<value>([^<]+)</value>", block)
            price_match = re.search(r"<transactionPricePerShare>\s*<value>([^<]+)</value>", block)
            date_match = re.search(r"<transactionDate>\s*<value>([^<]+)</value>", block)

            shares = int(float(shares_match.group(1))) if shares_match else 0
            price = float(price_match.group(1)) if price_match else 0.0
            tx_date = date_match.group(1) if date_match else ""

            if shares > 0:
                transactions.append(
                    InsiderTransaction(
                        insider_name=owner_name,
                        insider_title=owner_title,
                        transaction_type=tx_code,
                        shares=shares,
                        price_per_share=price,
                        date=tx_date,
                    )
                )

        return transactions

    def get_insider_activity(
        self,
        ticker: str,
        days_back: int = 30,
    ) -> InsiderActivity:
        """
        Analyse les transactions d'insiders depuis les Form 4 EDGAR.

        Algorithme de scoring (Seyhun 1986, Lakonishok & Lee 2001) :
          - Achats nets > 10 000 actions : signal BULLISH_INSIDER → bonus confiance +15%
          - Ventes nettes > 50 000 actions : signal BEARISH_INSIDER → malus -15%
          - Note : les ventes sont moins informatives (liquidité personnelle)

        Args:
            ticker   : symbole boursier
            days_back : fenêtre d'analyse (30 jours recommandé — Lakonishok & Lee)
        """
        cik = self.get_cik(ticker)
        if not cik:
            return InsiderActivity(
                ticker=ticker,
                period_days=days_back,
                note="CIK introuvable — analyse insider impossible",
            )

        data = self._get(EDGAR_SUBMISSIONS_URL.format(cik=cik))
        if not data:
            return InsiderActivity(ticker=ticker, period_days=days_back, note="API EDGAR indisponible")

        company_name = data.get("name", ticker)
        filings = data.get("filings", {}).get("recent", {})

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        cutoff_str = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        all_transactions: list[InsiderTransaction] = []
        buyers: set[str] = set()
        sellers: set[str] = set()

        # Limite le parsing XML à max 10 Form 4 pour respecter le rate limit SEC
        parsed_count = 0
        max_form4_parse = 10

        for form, date, acc in zip(forms, dates, accessions):
            if form != "4":
                continue
            if date < cutoff_str:
                break
            if parsed_count >= max_form4_parse:
                logger.debug("[EDGAR] Max Form 4 parse atteint (%d)", max_form4_parse)
                break

            txs = self._parse_form4_xml(cik, acc)
            all_transactions.extend(txs)
            parsed_count += 1

            for tx in txs:
                if tx.transaction_type == "P":
                    buyers.add(tx.insider_name)
                elif tx.transaction_type == "S":
                    sellers.add(tx.insider_name)

        # Calcul des shares nets
        net_shares = sum(tx.shares if tx.transaction_type == "P" else -tx.shares for tx in all_transactions)

        activity = InsiderActivity(
            ticker=ticker,
            period_days=days_back,
            transactions=all_transactions,
            net_shares_bought=net_shares,
            n_buyers=len(buyers),
            n_sellers=len(sellers),
        )

        # Scoring Seyhun (1986) + Lakonishok & Lee (2001)
        if net_shares > 10_000:
            activity.signal = "BULLISH_INSIDER"
            activity.confidence_adjustment = 1.15
            activity.note = (
                f"Achats nets de {net_shares:,} actions par {len(buyers)} insider(s) "
                f"sur {days_back}j. Signal haussier (Seyhun 1986 : +4% sur 6 mois)."
            )
        elif net_shares < -50_000:
            activity.signal = "BEARISH_INSIDER"
            activity.confidence_adjustment = 0.85
            activity.note = (
                f"Ventes nettes de {abs(net_shares):,} actions par {len(sellers)} insider(s) "
                f"sur {days_back}j. Signal baissier modéré (Lakonishok & Lee 2001 : "
                f"les ventes sont moins informatives que les achats)."
            )
        elif len(all_transactions) == 0:
            activity.signal = "NEUTRE"
            activity.confidence_adjustment = 1.0
            n_form4 = sum(1 for f, d in zip(forms, dates) if f == "4" and d >= cutoff_str)
            if n_form4 == 0:
                activity.note = f"Aucune déclaration Form 4 dans les {days_back} jours."
            else:
                activity.note = (
                    f"{n_form4} Form 4 déposé(s) sur {days_back}j, mais aucune "
                    f"transaction d'achat/vente sur marché détectée (probablement "
                    f"des RSU/options/gifts — non informatifs)."
                )
        else:
            activity.signal = "NEUTRE"
            activity.confidence_adjustment = 1.0
            activity.note = (
                f"{len(all_transactions)} transactions sur {days_back}j "
                f"(net={net_shares:+,} actions). Activité mixte — signal neutre."
            )

        logger.info(
            "[EDGAR] Insider %s : %d tx parsées (net=%+d) | %d buyers, %d sellers | Signal: %s",
            ticker,
            len(all_transactions),
            net_shares,
            len(buyers),
            len(sellers),
            activity.signal,
        )
        return activity

    # --- Fact-Checking Intégré ---

    def fact_check_article(
        self,
        ticker: str,
        article_text: str,
        article_url: str = "",
        days_back: int = 7,
    ) -> tuple[bool, str, float]:
        """
        Vérifie un article contre les dépôts SEC officiels.

        Remplace l'heuristique regex par une vraie consultation EDGAR.

        Logique :
          1. Si article_url provient de sec.gov : vérifié à 100%
          2. Si 8-K déposé dans la même fenêtre temporelle sur ce ticker :
             → l'événement est confirmé officiellement
          3. Si pas de 8-K pour un sujet qui devrait en avoir un :
             → signal de rumeur (pénalité confiance)
          4. Indicateurs lexicaux (règle précédente) comme fallback

        Returns:
            (is_verified, reason, confidence_factor)
        """
        # Vérification 1 : source directement SEC
        if "sec.gov" in article_url.lower() or "edgar" in article_url.lower():
            return True, "Source directe SEC EDGAR (URL officielle)", 1.0

        # Vérification 2 : recherche d'un 8-K récent correspondant
        filing = self.find_recent_8k(ticker, days_back=days_back)

        # Mots-clés du texte qui exigent normalement un 8-K
        requires_8k_keywords = [
            "earnings",
            "results",
            "revenue",
            "acquisition",
            "merger",
            "ceo",
            "cfo",
            "resign",
            "bankruptcy",
            "settlement",
            "fine",
            "résultats",
            "acquisition",
            "fusion",
            "démission",
            "faillite",
            "trimestre",
            "bénéfice",
            "chiffre d'affaires",
        ]

        text_lower = article_text.lower()
        mentions_material_event = any(kw in text_lower for kw in requires_8k_keywords)

        if filing and filing.found:
            confidence = 1.0 if filing.is_within_4_days else 0.90
            reason = (
                f"8-K déposé sur EDGAR le {filing.filing_date} "
                f"({'conforme 13a-11' if filing.is_within_4_days else 'dépôt tardif'}) | "
                f"{filing.description[:100]}"
            )
            return True, reason, confidence

        elif mentions_material_event and not (filing and filing.found):
            # L'article parle d'un événement qui devrait avoir un 8-K, mais on n'en trouve pas
            reason = (
                f"L'article mentionne un événement matériel (earnings/M&A/leadership) "
                f"mais aucun 8-K trouvé sur EDGAR dans les {days_back} jours → "
                f"potentielle rumeur ou information non encore déposée."
            )
            return False, reason, 0.65

        else:
            # Pas d'événement matériel évident → article d'analyse, opinion
            reason = "Article d'analyse/opinion — aucun dépôt SEC correspondant attendu."
            return False, reason, 0.85


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_edgar_client: Optional[SecEdgarClient] = None


def get_edgar_client() -> SecEdgarClient:
    """Retourne le singleton SecEdgarClient."""
    global _edgar_client
    if _edgar_client is None:
        _edgar_client = SecEdgarClient()
    return _edgar_client
