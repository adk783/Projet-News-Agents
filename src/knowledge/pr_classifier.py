"""
pr_classifier.py — Detecte les communiques de presse vs l'information editoriale.

POURQUOI
--------
Un communique de presse (corporate press release) est une source AUTO-EDITEE
par l'emetteur. Il faut le traiter differemment d'un article Reuters :
  - Contenu factuel (officiel) mais partisan (spin pro-emetteur)
  - Pas de verification editoriale independante
  - Souvent publie pendant heures de faible volume (flux-stuffing)
  - Utilise fortement pour "pump" les micro-caps

Un signal doit :
  - Reduire la confiance sur un PR par rapport a un wire
  - Rejeter tout trade sur micro-cap+PR (red flag pump-and-dump)
  - Ne pas rejeter les PR de grands groupes (AAPL, MSFT publient aussi)

DETECTION
---------
Deux signaux combines :
  1. Domaine : businesswire, prnewswire, globenewswire, accesswire (cf. source_tier)
  2. Contenu : patterns textuels classiques d'un PR
     - Boilerplate "About [Company]"
     - "Forward-looking statements"
     - "For more information, contact:"
     - "[Company] is a leading provider of..."
     - Mentions excessives du nom de la boite en peu de mots

REFERENCES
----------
- Bushee, B. J., Matsumoto, D. A., Miller, G. S. (2004). "Managerial and
  Investor Responses to Disclosure Regulation: The Case of Reg FD and
  Conference Calls."
- Bonsall, S. B., Leone, A. J., Miller, B. P. (2017). "A Plain English
  Measure of Financial Reporting Readability."
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Import absolu (cf. ADR-008). Le fallback None est conserve pour les rares
# environnements ou source_tier ne serait pas installable.
try:
    from src.utils.source_tier import is_press_release as _domain_is_pr
except ImportError:
    _domain_is_pr = None  # type: ignore


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    (re.compile(r"\babout\s+[A-Z][A-Za-z0-9&\.\- ]+\b.*?(\n|$)", re.I), "about_section"),
    (re.compile(r"\bforward[- ]looking\s+statements?\b", re.I), "forward_looking"),
    (re.compile(r"\bsafe\s+harbor\s+(statement|provisions?)\b", re.I), "safe_harbor"),
    (re.compile(r"\bfor\s+more\s+information,?\s+(please\s+)?(contact|visit)\b", re.I), "contact"),
    (re.compile(r"\b(investor|media)\s+(relations|contact(s)?)\s*:", re.I), "ir_contact"),
    (re.compile(r"\bpress\s+release\b", re.I), "press_release_literal"),
    (
        re.compile(
            r"\bis\s+a\s+(leading|global|premier|world[- ]class)\s+(provider|manufacturer|developer|supplier|company)\b",
            re.I,
        ),
        "leading_provider",
    ),
    (re.compile(r"\b(nyse|nasdaq|otc)\s*:\s*[A-Z]{2,5}\b"), "ticker_prefix"),
    (re.compile(r"\btraded\s+on\s+the\s+(nyse|nasdaq|otc|euronext|lse)\b", re.I), "listing_mention"),
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRDetection:
    is_press_release: bool
    confidence: float  # in [0, 1]
    reasons: Tuple[str, ...]
    domain_match: bool
    content_score: float

    def summary(self) -> str:
        tag = "PR" if self.is_press_release else "news"
        return f"{tag} conf={self.confidence:.2f} reasons={','.join(self.reasons[:3])}"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

_COMPANY_MENTION_RE = re.compile(r"\b[A-Z][A-Za-z][A-Za-z0-9\.]{2,}\b")


def _content_score(text: str, company_name: Optional[str]) -> Tuple[float, list]:
    """
    Score in [0, 1] base sur des patterns de communique de presse.
    Retourne aussi la liste des patterns matches.
    """
    if not text:
        return 0.0, []

    matched: list = []
    raw = 0.0
    for pat, name in _BOILERPLATE_PATTERNS:
        if pat.search(text):
            matched.append(name)
            raw += 1.0

    # Densite mentions du nom de la boite (si connu)
    if company_name:
        wc = max(1, len(text.split()))
        mentions = len(re.findall(r"\b" + re.escape(company_name) + r"\b", text, re.I))
        density = mentions / wc
        if density > 0.03:  # > 3% des mots = pub
            matched.append(f"high_company_density({density:.2%})")
            raw += 1.0

    # Densite absolue de majuscules (propre aux PR formatages corpo)
    all_caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    if all_caps >= 6:
        matched.append(f"many_all_caps({all_caps})")
        raw += 0.5

    # Normalisation : chaque pattern vaut ~1 point, on clippe vers [0, 1]
    # 3+ patterns = forte presomption PR
    score = min(1.0, raw / 4.0)
    return score, matched


def detect_press_release(
    text: str,
    url: Optional[str] = None,
    company_name: Optional[str] = None,
    content_threshold: float = 0.5,
) -> PRDetection:
    """
    Retourne une PRDetection. Deux signaux :
      1. URL / domaine dans la liste des fils PR
      2. Patterns de contenu
    Un seul des deux suffit pour classer en PR, avec confidence moduled.
    """
    domain_match = False
    if url and _domain_is_pr is not None:
        try:
            domain_match = bool(_domain_is_pr(url))
        except Exception:
            domain_match = False

    content_sc, reasons = _content_score(text or "", company_name)

    # Fusion
    if domain_match:
        confidence = max(0.9, 0.6 + content_sc * 0.4)
        is_pr = True
        reasons = ("domain_match",) + tuple(reasons)
    elif content_sc >= content_threshold:
        confidence = min(0.9, content_sc)
        is_pr = True
        reasons = tuple(reasons)
    else:
        confidence = content_sc
        is_pr = False
        reasons = tuple(reasons)

    return PRDetection(
        is_press_release=is_pr,
        confidence=float(confidence),
        reasons=reasons,
        domain_match=domain_match,
        content_score=float(content_sc),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pr_body = (
        "CUPERTINO, Calif., April 23, 2026 /PRNewswire/ -- Apple Inc. (NASDAQ: AAPL), "
        "a leading global provider of consumer technology products, today announced "
        "a strategic partnership with Acme Corp.\n"
        "About Apple\n"
        "Apple revolutionized personal technology with the introduction of the Macintosh in 1984.\n"
        "Forward-looking statements\n"
        "This press release contains forward-looking statements. For more information, contact: "
        "Investor Relations, investors@apple.com."
    )

    news_body = (
        "Apple's quarterly revenue climbed 3% to $89.5 billion, slightly above "
        "Wall Street's consensus estimates, as iPhone sales in emerging markets "
        "offset weakness in China. Analysts at Morgan Stanley said the result "
        "underscores the resilience of the services segment, which grew 14% "
        "year-over-year. The stock rose 2.1% in extended trading."
    )

    cases = [
        (pr_body, "https://www.prnewswire.com/news/apple/12345", "Apple", True),
        (pr_body, None, "Apple", True),  # pas d'URL, juste contenu
        (news_body, "https://www.reuters.com/markets/aapl-q3", "Apple", False),
        (news_body, None, "Apple", False),
    ]

    for body, url, company, expected in cases:
        det = detect_press_release(body, url=url, company_name=company)
        mark = "OK" if det.is_press_release == expected else "X"
        logger.info(f"[{mark}] expected={expected}  got={det.is_press_release}")
        logger.info(f"    url={url}")
        logger.info(f"    {det.summary()}")
        print()
