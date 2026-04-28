"""
lang_detect.py — Détection de langue rapide par signatures de caractères.

CONTEXTE
--------
Le pipeline passe tous les articles au modèle anglophone
(`distilroberta-finetuned-financial-news`) et aux prompts en français.
Finnhub/Yahoo renvoient occasionnellement du portugais, allemand,
espagnol, chinois, japonais. On doit :
  1. Détecter la langue
  2. Rejeter (ou router vers un modèle adapté) les non-{en, fr}

STRATÉGIE
---------
On privilégie `langdetect` (port Python de la lib Java de Google Compact
Language Detector), car :
  - Pas de dépendance native (contrairement à fasttext-langid qui exige libomp)
  - 55 langues supportées
  - Précision ≥ 99% sur textes > 50 caractères
  - Purement probabiliste (deterministe après seed fix)

Fallback si langdetect absent : détection n-gramme naive basée sur la
fréquence de stopwords et caractères Unicode non-latins.

RÉFÉRENCES
----------
- Shuyo, N. (2010). "Language Detection Library for Java."
  https://github.com/shuyo/language-detection
- Lui, M. & Baldwin, T. (2012). "langid.py: An Off-the-shelf Language
  Identification Tool." ACL 2012.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Loader lazy pour langdetect (optionnel)
# ---------------------------------------------------------------------------

_LANGDETECT = None


def _load_langdetect():
    global _LANGDETECT
    if _LANGDETECT is not None:
        return _LANGDETECT
    try:
        from langdetect import DetectorFactory, LangDetectException, detect

        DetectorFactory.seed = 42  # déterminisme
        _LANGDETECT = {
            "detect": detect,
            "exception": LangDetectException,
            "available": True,
        }
    except ImportError:
        _LANGDETECT = {"available": False}
    return _LANGDETECT


# ---------------------------------------------------------------------------
# Fallback naïf : heuristique stopwords + scripts non-latins
# ---------------------------------------------------------------------------

_STOPWORDS_PER_LANG = {
    "en": {
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "is",
        "on",
        "for",
        "that",
        "with",
        "as",
        "at",
        "by",
        "from",
        "was",
        "were",
        "be",
        "has",
        "have",
        "its",
        "this",
        "his",
        "her",
        "their",
        "they",
        "we",
        "an",
        "or",
        "but",
        "not",
        "are",
        "he",
        "she",
        "it",
        "which",
        "who",
        "what",
        "when",
        "where",
        "today",
        "will",
        "can",
        "may",
        "had",
        "been",
        "would",
        "could",
        "should",
        "about",
        "after",
        "before",
        "into",
        "over",
        "under",
        "between",
        "through",
        "more",
    },
    "fr": {
        "le",
        "la",
        "les",
        "de",
        "et",
        "un",
        "une",
        "des",
        "du",
        "est",
        "sont",
        "dans",
        "pour",
        "sur",
        "par",
        "aux",
        "avec",
        "que",
        "qui",
        "ce",
        "cette",
        "son",
        "sa",
        "ses",
        "au",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "de",
        "y",
        "en",
        "un",
        "una",
        "es",
        "por",
        "con",
        "para",
        "que",
        "su",
        "sus",
        "no",
        "se",
        "lo",
    },
    "pt": {
        "o",
        "a",
        "os",
        "as",
        "de",
        "e",
        "em",
        "um",
        "uma",
        "é",
        "por",
        "com",
        "para",
        "que",
        "seu",
        "sua",
        "não",
        "se",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "ist",
        "sind",
        "ein",
        "eine",
        "für",
        "mit",
        "auf",
        "in",
        "den",
        "des",
        "dem",
        "zu",
        "von",
        "im",
    },
    "it": {"il", "la", "lo", "i", "gli", "le", "di", "e", "in", "un", "una", "è", "per", "con", "che", "non", "sono"},
    "nl": {"de", "het", "een", "en", "van", "in", "is", "zijn", "op", "voor", "met", "aan", "door", "dat"},
}


_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")


def _script_detection(text: str) -> Optional[str]:
    """Détecte les scripts non-latins (chinois, japonais, russe, arabe)."""
    if not text:
        return None
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            counts["zh_ja"] = counts.get("zh_ja", 0) + 1
        elif "CYRILLIC" in name:
            counts["ru"] = counts.get("ru", 0) + 1
        elif "ARABIC" in name:
            counts["ar"] = counts.get("ar", 0) + 1
        elif "HANGUL" in name:
            counts["ko"] = counts.get("ko", 0) + 1
    if not counts:
        return None
    top = max(counts.items(), key=lambda kv: kv[1])
    if top[1] >= 5:
        return top[0]
    return None


def _fallback_detect(text: str) -> str:
    """Heuristique stopwords (en, fr, es, pt, de, it, nl) + script non-latin."""
    # 1) Script non-latin prioritaire
    script = _script_detection(text)
    if script:
        return script if script != "zh_ja" else "zh"  # on regroupe

    # 2) Matching stopwords
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return "unknown"
    scores: dict[str, int] = {}
    for lang, stops in _STOPWORDS_PER_LANG.items():
        scores[lang] = sum(1 for t in tokens if t in stops)
    total_matches = sum(scores.values())
    if total_matches < 3:
        return "unknown"  # texte trop court ou langue non-supportée
    return max(scores.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str:
    """
    Détecte la langue d'un texte. Retourne un code ISO 639-1 ('en', 'fr', ...)
    ou 'unknown' si indéterminé, 'zh'/'ja'/'ru'/'ar'/'ko' pour scripts non-latins.
    """
    if not text or len(text.strip()) < 10:
        return "unknown"

    ld = _load_langdetect()
    if ld.get("available"):
        try:
            return ld["detect"](text)
        except ld["exception"]:
            pass  # fallback
    return _fallback_detect(text)


def is_supported(lang: str, supported: set[str] = frozenset({"en", "fr"})) -> bool:
    """True si la langue fait partie de la liste supportée par le pipeline."""
    return lang in supported


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("en", "Apple reported strong quarterly earnings beating analyst estimates."),
        ("fr", "Apple a publie des resultats superieurs aux attentes des analystes."),
        ("es", "Apple ha informado de ganancias trimestrales superiores a las estimaciones."),
        ("de", "Apple hat starke Quartalsergebnisse gemeldet, die die Analystenschaetzungen uebertrafen."),
        ("unknown", "short"),
    ]
    for expected, text in tests:
        got = detect_language(text)
        ok = "OK" if got == expected else f"MISMATCH (expected {expected})"
        logger.info(f"  [{got:8}] {ok}  -- {text[:70]}")
