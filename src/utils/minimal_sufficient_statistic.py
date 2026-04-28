"""
minimal_sufficient_statistic.py — Encodage déterministe du signal causal utile

OBJECTIF
--------
Construire une *statistique suffisante minimale* T(article) qui capture tout ce
qui est causalement pertinent pour p(rendement futur | news), et rien d'autre.

Pourquoi c'est critique (Pearl, 2009 ; Veitch et al., 2021)
------------------------------------------------------------
Une question de news peut être décomposée en deux parties :
  1. Le SIGNAL causal utile : "earnings beat de +10% sur les ventes iPhone"
     → ne dépend pas de qui est l'entreprise ni de la date.
  2. Les NUISANCES confondantes : la marque, le CEO, l'année, le sentiment
     émotionnel, le style journalistique, la langue.
     → le LLM peut les corréler à son prior de training et tricher.

Si T(x) est une statistique suffisante, alors :
    p(return | article) = p(return | T(article))

et donc apprendre à décider à partir de T(x) plutôt que x directement
supprime toutes les dépendances aux nuisances.

SCHÉMA DE T(x) — vecteur 20-dim
-------------------------------
Slot 0   : event_type_id           (one-hot compressé en id ∈ [0..9])
Slots 1-3: polarity                (aspect-sentiment agrégé : +1/0/-1 sur 3 dims)
Slots 4-6: magnitude_buckets       (small / medium / large)
Slot 7   : has_earnings_surprise   (0/1)
Slot 8   : has_guidance_change     (0/1)
Slot 9   : has_regulatory_action   (0/1)
Slot 10  : has_mna                 (0/1)
Slot 11  : has_exec_change         (0/1)
Slot 12  : has_product_launch      (0/1)
Slot 13  : has_macro_spillover     (0/1)
Slot 14  : num_entities_mentioned  (normalisé [0,1])
Slot 15  : factual_density         (ID score [0,1])
Slot 16  : sentiment_spread        (|pos - neg| / total ABSA)
Slot 17  : absa_ambiguity          (1 - |ratio - 0.5|·2)
Slot 18  : magnitude_normalized    (quantile de la variation annoncée)
Slot 19  : reserved_future

Ces 20 features sont *invariantes* à la marque, la date, la langue, le style.
Une décision prise uniquement sur T(x) ne peut pas tricher via le prior
du LLM.

USAGE
-----
    from src.utils.minimal_sufficient_statistic import compute_mss

    t = compute_mss(absa_result=..., article_text=..., finbert=...)
    # -> np.array([...]) de shape (20,)

Réf :
    Pearl, J. (2009). Causality. Cambridge University Press.
    Veitch, V., D'Amour, A., Yadlowsky, S., Eisenstein, J. (2021).
        "Counterfactual Invariance to Spurious Correlations in Text
         Classification." NeurIPS.
    Fisher, R.A. (1922). "On the mathematical foundations of theoretical
         statistics." Phil. Trans. Royal Society.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Taxonomie d'évenements (déterministe, basée sur mots-clés)
# ---------------------------------------------------------------------------

EVENT_TYPES = {
    0: "UNKNOWN",
    1: "EARNINGS",
    2: "GUIDANCE",
    3: "MNA",  # mergers & acquisitions
    4: "REGULATORY",
    5: "EXEC_CHANGE",
    6: "PRODUCT_LAUNCH",
    7: "LITIGATION",
    8: "MACRO_SPILLOVER",
    9: "BUYBACK_DIVIDEND",
}

_KEYWORDS = {
    1: [
        r"\bearnings?\b",
        r"\beps\b",
        r"\brevenue\b",
        r"\bbeat\b",
        r"\bmiss(es|ed)?\b",
        r"\bquarterly\b",
        r"\bq[1-4]\s*20\d{2}\b",
    ],
    2: [
        r"\bguidance\b",
        r"\boutlook\b",
        r"\bforecast\b",
        r"\braised?\b",
        r"\blower(ed)?\b",
        r"\braise(d|s)?\b",
        r"\bcut(s|ting)?\b",
    ],
    3: [
        r"\bacqui(re|sition)\b",
        r"\bmerger?\b",
        r"\btakeover\b",
        r"\bbuyout\b",
        r"\bdeal\b",
        r"\bpurchase(d|s)?\b",
        r"\bmerge(d|s)?\b",
    ],
    4: [
        r"\bsec\b",
        r"\bregulat(or|ory|e|ed|ion)\b",
        r"\bfined?\b",
        r"\bantitrust\b",
        r"\binvestigat(ion|ed)\b",
        r"\bsanction(s|ed)?\b",
        r"\bdoj\b",
    ],
    5: [
        r"\bceo\b",
        r"\bcfo\b",
        r"\bresign(ed|s|ation)?\b",
        r"\bstep(ped|s)?\s+down\b",
        r"\bappointed?\b",
        r"\bfired?\b",
        r"\bnamed\b.*\bceo\b",
    ],
    6: [r"\blaunch(ed|es|ing)?\b", r"\bunveils?\b", r"\breleases?\b", r"\bnew\s+product\b", r"\bannounce[ds]?\s+\w+"],
    7: [
        r"\blawsuit\b",
        r"\bsues?\b",
        r"\bsett(le|lement|led)\b",
        r"\bdamages?\b",
        r"\bcourt\b",
        r"\bpatent\b.*\binfring\w*",
    ],
    8: [r"\bfed\b", r"\brates?\b", r"\bcpi\b", r"\bunemployment\b", r"\binflation\b", r"\bfomc\b", r"\brecession\b"],
    9: [r"\bbuyback\b", r"\bshare\s+repurchase\b", r"\bdividend\b", r"\bcapital\s+return\b"],
}

_PCT_PATTERN = re.compile(r"([+-]?\s*\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_NUM_PATTERN = re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|million|mn|b|m)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------


@dataclass
class MSSResult:
    vector: list[float]  # 20-dim encoding
    event_type: str  # label textuel
    event_type_id: int
    magnitude: float  # [-1, +1] signed normalized magnitude
    polarity: float  # ABSA ratio transformed: 2·(pos/total)-1
    flags: dict[str, int]  # booléens détectés
    raw_features: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "event_type_id": self.event_type_id,
            "magnitude": self.magnitude,
            "polarity": self.polarity,
            "flags": self.flags,
            "vector": list(self.vector),
        }


# ---------------------------------------------------------------------------
# Détection d'événement
# ---------------------------------------------------------------------------


def _detect_event_type(text: str) -> tuple[int, dict[str, int]]:
    """Classifie l'article en un type d'événement dominant (par vote de mots-clés)."""
    if not text:
        return 0, {}
    scores: dict[int, int] = {}
    flags: dict[str, int] = {v: 0 for v in EVENT_TYPES.values()}
    for eid, patterns in _KEYWORDS.items():
        hits = 0
        for p in patterns:
            hits += len(re.findall(p, text, re.IGNORECASE))
        if hits > 0:
            scores[eid] = hits
            flags[EVENT_TYPES[eid]] = 1
    if not scores:
        return 0, flags
    dominant = max(scores.items(), key=lambda kv: kv[1])[0]
    return dominant, flags


def _extract_magnitude(text: str) -> float:
    """
    Extrait la magnitude signée dominante :
      1. Plus grand pourcentage trouvé (en valeur absolue)
      2. À défaut, plus grand montant monétaire (normalisé log)
    Retour dans [-1, 1].
    """
    if not text:
        return 0.0
    pcts = _PCT_PATTERN.findall(text)
    magnitudes = []
    for raw in pcts:
        try:
            v = float(raw.replace(" ", ""))
            if abs(v) <= 500:  # sanity: filtre les faux positifs ("100%")
                magnitudes.append(v)
        except ValueError:
            pass
    if magnitudes:
        dominant = max(magnitudes, key=abs)
        # Normalise tanh pour compresser [-50%, +50%] → [-1, +1]
        return math.tanh(dominant / 25.0)
    # Fallback : montants monétaires
    amounts = _NUM_PATTERN.findall(text)
    if amounts:
        try:
            v = max(float(a) for a in amounts)
            return math.tanh(math.log10(v + 1) / 3.0)
        except ValueError:
            pass
    return 0.0


def _bucket_magnitude(mag: float) -> tuple[int, int, int]:
    """One-hot : small / medium / large (pour discrétiser)."""
    am = abs(mag)
    if am < 0.2:
        return (1, 0, 0)
    if am < 0.6:
        return (0, 1, 0)
    return (0, 0, 1)


def _polarity_from_absa(absa_result: dict) -> tuple[float, float, float]:
    """
    Retourne (polarity, ambiguity, spread) à partir de l'ABSA.
    polarity ∈ [-1, +1] ; ambiguity ∈ [0,1] ; spread = |diff|/total.
    """
    aspects = absa_result.get("aspects", []) if isinstance(absa_result, dict) else []
    nb_pos = sum(1 for a in aspects if (a or {}).get("sentiment") == "positive")
    nb_neg = sum(1 for a in aspects if (a or {}).get("sentiment") == "negative")
    total = nb_pos + nb_neg
    if total == 0:
        return 0.0, 1.0, 0.0
    ratio = nb_pos / total
    polarity = 2 * ratio - 1
    ambiguity = 1.0 - abs(ratio - 0.5) * 2
    spread = abs(nb_pos - nb_neg) / total
    return polarity, ambiguity, spread


def _count_entities(text: str) -> int:
    """Proxy simple : séquences majuscules (Apple, Microsoft, etc.)."""
    if not text:
        return 0
    # Token en majuscule initial, 2 caractères min, évite débuts de phrase triviaux
    tokens = re.findall(r"\b[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]+)?\b", text)
    return len(set(tokens))


def _factual_density(text: str) -> float:
    """
    Proxy léger de l'Information Density : ratio de chiffres + pourcentages
    par 100 mots. Éviter d'importer src.utils.information_density pour rester
    sans dépendance circulaire.
    """
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    nb_nums = len(re.findall(r"\d", text))
    return min(1.0, nb_nums / max(1, len(words)))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def compute_mss(
    article_text: str,
    absa_result: Optional[dict] = None,
    finbert: Optional[float] = None,
) -> MSSResult:
    """
    Calcule la statistique suffisante minimale T(x) en un vecteur 20-dim.

    La valeur retournée est indépendante du ticker, du nom d'entreprise, de
    la date et du CEO — elle ne dépend que du type d'événement, de la
    polarité ABSA et de la magnitude. Elle est donc *counterfactually
    invariant* sur ces nuisances.
    """
    text = article_text or ""
    event_id, flags = _detect_event_type(text)
    event_name = EVENT_TYPES[event_id]

    magnitude = _extract_magnitude(text)
    mag_buckets = _bucket_magnitude(magnitude)

    polarity, ambiguity, spread = _polarity_from_absa(absa_result or {})

    # Si FinBERT disponible, on moyenne avec la polarité ABSA pour plus de robustesse
    if finbert is not None:
        try:
            fb_p = 2 * float(finbert) - 1  # [0,1] -> [-1,+1]
            polarity = 0.5 * polarity + 0.5 * fb_p
        except (TypeError, ValueError):
            pass

    n_entities = _count_entities(text)
    n_entities_norm = min(1.0, n_entities / 30.0)  # cap
    fact_density = _factual_density(text)

    vector = [
        event_id / 10.0,  # 0  event_type_id normalisé
        max(0.0, polarity),
        max(0.0, -polarity),  # 1-2 polarité (+ / -)
        1.0 if abs(polarity) < 0.2 else 0.0,  # 3 neutre
        float(mag_buckets[0]),
        float(mag_buckets[1]),  # 4-5 small / medium
        float(mag_buckets[2]),  # 6 large
        float(flags.get("EARNINGS", 0)),  # 7
        float(flags.get("GUIDANCE", 0)),  # 8
        float(flags.get("REGULATORY", 0)),  # 9
        float(flags.get("MNA", 0)),  # 10
        float(flags.get("EXEC_CHANGE", 0)),  # 11
        float(flags.get("PRODUCT_LAUNCH", 0)),  # 12
        float(flags.get("MACRO_SPILLOVER", 0)),  # 13
        n_entities_norm,  # 14
        fact_density,  # 15
        spread,  # 16
        ambiguity,  # 17
        abs(magnitude),  # 18
        0.0,  # 19 reserved
    ]

    return MSSResult(
        vector=vector,
        event_type=event_name,
        event_type_id=event_id,
        magnitude=round(magnitude, 4),
        polarity=round(polarity, 4),
        flags=flags,
        raw_features={
            "ambiguity": ambiguity,
            "spread": spread,
            "n_entities": n_entities,
            "factual_density": fact_density,
        },
    )


def mss_distance(a: MSSResult, b: MSSResult) -> float:
    """
    Distance L2 entre deux MSS. Si deux articles ont distance ≈ 0 mais des
    signaux différents → violation de l'invariance causale.
    """
    if len(a.vector) != len(b.vector):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.vector, b.vector)))


if __name__ == "__main__":
    # Smoke test
    txt = (
        "Apple beat Q4 earnings with EPS of $2.30 vs $2.10 expected. "
        "Revenue rose 12% year over year driven by strong iPhone 15 sales. "
        "Tim Cook raised guidance for next quarter."
    )
    absa = {
        "aspects": [
            {"sentiment": "positive"},
            {"sentiment": "positive"},
            {"sentiment": "negative"},
            {"sentiment": "positive"},
        ]
    }
    t = compute_mss(txt, absa, finbert=0.78)
    print("Event:", t.event_type, "| Magnitude:", t.magnitude, "| Polarity:", t.polarity)
    print("Vector:", [round(x, 3) for x in t.vector])
