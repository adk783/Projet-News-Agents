"""
ner_gliner.py — Extraction d'entites nommees zero-shot pour articles financiers.

CONTEXTE
--------
Le pipeline actuel utilise une fonction `is_relevant(title, keywords)` basee
sur un simple matching de mots-cles dans news_pipeline.py. Cette approche :
  - Rate les articles sans le ticker explicite (ex: "Cupertino tech giant" pour AAPL)
  - Confond les homonymes ("Apple" fruit vs "Apple Inc.")
  - N'extrait aucune structure exploitable en aval (ticker, CEO, produit, montant)

MIGRATION VERS NER ZERO-SHOT
----------------------------
GLiNER (Zaratiana et al., 2024) est un modele compact (~200MB) qui fait du
zero-shot NER en predisant des spans pour des labels arbitraires fournis au
runtime. Plus besoin de fine-tuner un modele pour chaque nouveau label.

Labels financiers pertinents :
  - ORGANIZATION  : entreprise, banque centrale, regulateur
  - PERSON        : CEO, analyste, politicien
  - PRODUCT       : iPhone, GPT-5, Ozempic
  - EVENT         : earnings, FDA approval, M&A, lawsuit
  - MONEY         : montant de deal, revenu, amende
  - DATE          : echeance, fenetre d'execution
  - GPE           : pays, ville (impact geopolitique)

CASCADE DE FALLBACKS
--------------------
1. GLiNER (pip install gliner) si disponible  -> zero-shot propre
2. spaCy transformer (en_core_web_trf) si installe -> NER classique
3. Heuristique regex minimale (tickers, $montants, %) -> mieux que rien

REFERENCES
----------
- Zaratiana, U. et al. (2024). "GLiNER: Generalist Model for Named Entity
  Recognition using Bidirectional Transformer."
  https://arxiv.org/abs/2311.08526
- Honnibal & Montani (2017). spaCy 2: Natural language understanding.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Types publics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """Une entite nommee extraite d'un texte."""

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = "unknown"  # "gliner" | "spacy" | "regex"


DEFAULT_FINANCIAL_LABELS: tuple[str, ...] = (
    "company",
    "person",
    "product",
    "financial_event",
    "money",
    "percentage",
    "ticker",
    "country",
    "date",
)


# ---------------------------------------------------------------------------
# Cascade loader
# ---------------------------------------------------------------------------

_BACKEND: Optional[dict] = None


def _load_backend() -> dict:
    """Charge GLiNER -> spaCy -> regex (premier disponible)."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # 1) GLiNER
    try:
        from gliner import GLiNER  # type: ignore

        model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        _BACKEND = {"name": "gliner", "model": model}
        return _BACKEND
    except Exception:
        pass

    # 2) spaCy
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_trf")
        except OSError:
            nlp = spacy.load("en_core_web_sm")
        _BACKEND = {"name": "spacy", "model": nlp}
        return _BACKEND
    except Exception:
        pass

    # 3) Regex fallback (toujours disponible)
    _BACKEND = {"name": "regex", "model": None}
    return _BACKEND


# ---------------------------------------------------------------------------
# Backend GLiNER
# ---------------------------------------------------------------------------


def _extract_gliner(
    text: str,
    labels: Sequence[str],
    threshold: float = 0.45,
    model=None,
) -> List[Entity]:
    """Appel direct GLiNER avec les labels fournis."""
    preds = model.predict_entities(text, list(labels), threshold=threshold)
    out: List[Entity] = []
    for p in preds:
        out.append(
            Entity(
                text=p["text"],
                label=p["label"],
                start=p["start"],
                end=p["end"],
                confidence=float(p.get("score", 1.0)),
                source="gliner",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Backend spaCy
# ---------------------------------------------------------------------------

_SPACY_TO_FIN = {
    "ORG": "company",
    "PERSON": "person",
    "PRODUCT": "product",
    "EVENT": "financial_event",
    "MONEY": "money",
    "PERCENT": "percentage",
    "DATE": "date",
    "GPE": "country",
    "LOC": "country",
}


def _extract_spacy(text: str, model) -> List[Entity]:
    doc = model(text)
    out: List[Entity] = []
    for ent in doc.ents:
        mapped = _SPACY_TO_FIN.get(ent.label_)
        if not mapped:
            continue
        out.append(
            Entity(
                text=ent.text,
                label=mapped,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy ne renvoie pas de score par defaut
                source="spacy",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Backend regex (fallback minimal mais correct)
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])")
_MONEY_RE = re.compile(r"(?:\$|USD\s*|€|EUR\s*)(\d[\d,\.]*\s*(?:billion|million|bn|mn|k|B|M|K)?)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_EVENT_KEYWORDS = {
    "earnings": "financial_event",
    "beats": "financial_event",
    "misses": "financial_event",
    "acquires": "financial_event",
    "acquisition": "financial_event",
    "merger": "financial_event",
    "lawsuit": "financial_event",
    "fda approval": "financial_event",
    "ipo": "financial_event",
    "guidance": "financial_event",
    "downgrade": "financial_event",
    "upgrade": "financial_event",
    "dividend": "financial_event",
    "buyback": "financial_event",
}


# Mots courts toutes-capitales qui ne sont presque jamais des tickers
_TICKER_BLACKLIST = {
    "CEO",
    "CFO",
    "COO",
    "CTO",
    "IPO",
    "SEC",
    "FED",
    "ECB",
    "USA",
    "USD",
    "EUR",
    "GDP",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "AI",
    "ML",
    "API",
    "OK",
    "THE",
    "AND",
    "FOR",
    "NEW",
    "FDA",
    "DOJ",
    "FTC",
    "DOE",
    "EPA",
}


def _extract_regex(text: str) -> List[Entity]:
    out: List[Entity] = []

    for m in _TICKER_RE.finditer(text):
        sym = m.group(1)
        if sym in _TICKER_BLACKLIST:
            continue
        out.append(
            Entity(
                text=sym,
                label="ticker",
                start=m.start(1),
                end=m.end(1),
                confidence=0.5,
                source="regex",
            )
        )

    for m in _MONEY_RE.finditer(text):
        out.append(
            Entity(
                text=m.group(0),
                label="money",
                start=m.start(),
                end=m.end(),
                confidence=0.8,
                source="regex",
            )
        )

    for m in _PERCENT_RE.finditer(text):
        out.append(
            Entity(
                text=m.group(0),
                label="percentage",
                start=m.start(),
                end=m.end(),
                confidence=0.9,
                source="regex",
            )
        )

    lower = text.lower()
    for kw, label in _EVENT_KEYWORDS.items():
        idx = 0
        while True:
            pos = lower.find(kw, idx)
            if pos == -1:
                break
            out.append(
                Entity(
                    text=text[pos : pos + len(kw)],
                    label=label,
                    start=pos,
                    end=pos + len(kw),
                    confidence=0.6,
                    source="regex",
                )
            )
            idx = pos + len(kw)

    return out


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def extract_entities(
    text: str,
    labels: Optional[Sequence[str]] = None,
    threshold: float = 0.45,
) -> List[Entity]:
    """
    Extrait les entites d'un texte. Retourne une liste triee par position.

    Cascade GLiNER > spaCy > regex. Pas d'import top-level : les deux
    premiers sont charges lazy a la premiere invocation.

    Args:
        text: texte source
        labels: labels souhaites (seulement utilises par GLiNER).
                Defaut = DEFAULT_FINANCIAL_LABELS.
        threshold: seuil de confiance (GLiNER uniquement).

    Returns:
        Liste d'Entity, triee par .start.
    """
    if not text or len(text.strip()) < 3:
        return []

    labels = labels or DEFAULT_FINANCIAL_LABELS
    backend = _load_backend()

    if backend["name"] == "gliner":
        ents = _extract_gliner(text, labels, threshold=threshold, model=backend["model"])
    elif backend["name"] == "spacy":
        ents = _extract_spacy(text, backend["model"])
    else:
        ents = _extract_regex(text)

    # Filtrage par labels demandes (regex et spacy peuvent rendre plus que demande)
    wanted = set(labels)
    ents = [e for e in ents if e.label in wanted]
    ents.sort(key=lambda e: e.start)
    return ents


def entities_by_label(entities: Sequence[Entity]) -> dict[str, List[Entity]]:
    """Regroupe les entites par label."""
    out: dict[str, List[Entity]] = {}
    for e in entities:
        out.setdefault(e.label, []).append(e)
    return out


def unique_entity_texts(entities: Sequence[Entity], label: str) -> List[str]:
    """Renvoie les textes uniques pour un label (preserve l'ordre d'apparition)."""
    seen: set[str] = set()
    out: List[str] = []
    for e in entities:
        if e.label != label:
            continue
        key = e.text.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        out.append(key)
    return out


def has_relevant_entity(
    text: str,
    company_names: Sequence[str],
    ticker: Optional[str] = None,
) -> bool:
    """
    Remplace `is_relevant(title, keywords)` : retourne True si le texte
    mentionne au moins une des `company_names` OU le `ticker`, via un match
    insensible a la casse (et ancrage \\b pour eviter les faux-positifs).
    """
    if not text:
        return False
    text_lower = text.lower()
    for name in company_names:
        if not name:
            continue
        pat = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pat, text_lower):
            return True
    if ticker:
        pat = r"\b" + re.escape(ticker.upper()) + r"\b"
        if re.search(pat, text):
            return True
    return False


def get_active_backend() -> str:
    """Pour diagnostic : retourne 'gliner' | 'spacy' | 'regex'."""
    return _load_backend()["name"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Apple Inc. (AAPL) reported Q4 earnings of $89.5 billion, beating "
        "analyst estimates by 3.2%. CEO Tim Cook announced a new iPhone model "
        "for the European market. The stock rose 2.1% in pre-market trading."
    )

    backend = get_active_backend()
    logger.info(f"[NER] backend actif: {backend}")
    print()

    ents = extract_entities(sample)
    logger.info(f"[NER] {len(ents)} entites extraites:")
    for e in ents:
        print(f"  [{e.label:15}] '{e.text}' (conf={e.confidence:.2f}, src={e.source})")

    print()
    logger.info("[NER] regroupement par label:")
    for label, ents_lab in entities_by_label(ents).items():
        uniq = unique_entity_texts(ents, label)
        logger.info(f"  {label:15} -> {uniq}")

    print()
    logger.info("[NER] has_relevant_entity sur titre court:")
    t1 = "Cupertino giant beats expectations"
    t2 = "Apple Inc. beats expectations"
    t3 = "AAPL hits all-time high"
    for t in (t1, t2, t3):
        r = has_relevant_entity(t, ["Apple", "Apple Inc."], ticker="AAPL")
        print(f"  [{'MATCH ' if r else 'NO    '}] {t}")
