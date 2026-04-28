"""
simhash_dedup.py — Déduplication sémantique (Jaccard + SimHash).

CONTEXTE
--------
Le pipeline actuel dédupe par `normalize_title(title)` : lowercase + strip
ponctuation, puis match exact. Ça laisse passer toutes les republications
où Reuters/AP distribuent la même dépêche avec un titre légèrement modifié
("Apple beats Q3 earnings" vs "Apple's Q3 earnings top estimates").

DEUX ALGORITHMES COMPLÉMENTAIRES
--------------------------------
1. **Jaccard sur token-sets** (Broder 1997) : pour les titres et textes courts.
   sim(A, B) = |A ∩ B| / |A ∪ B|  après stopword removal.
   Seuil ≥ 0.5 = near-duplicate. Robust sur 5-15 tokens.

2. **SimHash** (Charikar 2002, Manku 2007) : pour les bodies longs (>150 tokens).
   Features : 1-gram + 2-gram + char-5-grams, hash 64-bit, Hamming ≤ 3.
   O(1) per comparison after indexing.

La fonction `is_near_duplicate_text(a, b)` route automatiquement vers
Jaccard ou SimHash selon la longueur.

COMPLEXITÉ
----------
Jaccard avec index inversé (post token) : O(|A|·log|B|).
SimHash : O(1) par paire (XOR + popcount).
Scan linéaire ici : O(N) articles existants. N < 10k par ticker → OK.

RÉFÉRENCES
----------
- Broder, A. Z. (1997). "On the resemblance and containment of documents."
  SEQUENCES '97.
- Charikar, M. (2002). "Similarity Estimation Techniques from Rounding
  Algorithms." STOC '02.
- Manku, G. S., Jain, A., & Sarma, A. D. (2007). "Detecting Near-Duplicates
  for Web Crawling." WWW '07.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import hashlib
import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Paramètres par défaut
# ---------------------------------------------------------------------------
SIMHASH_BITS = 64
SHINGLE_SIZE = 4  # n-grammes de mots → capture les paraphrases légères
CHAR_SHINGLE_SIZE = 5  # n-grammes de caractères → robustesse sur titres courts

# Seuils calibrés :
# - HAMMING_THRESHOLD_EXACT : quasi-identique (republication stricte) - Manku 2007
# - HAMMING_THRESHOLD_CANDIDATE : candidat pour confirmation Jaccard
#
# La stratégie est un two-stage filter : SimHash fournit un stockage O(64-bit)
# et une comparaison O(1), on sélectionne les candidats < HAMMING_CANDIDATE
# puis on confirme avec Jaccard sur le token-set. Ça maintient la scalabilité
# sans sacrifier la précision sur les paraphrases.
HAMMING_THRESHOLD_EXACT = 8  # quasi-identique (syndication stricte)
HAMMING_THRESHOLD_CANDIDATE = 24  # candidat probable -> vérifier par Jaccard
HAMMING_THRESHOLD = HAMMING_THRESHOLD_CANDIDATE  # défaut conservateur


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords EN/FR minimaux pour Jaccard titre (on ne veut pas que "the"
# gonfle artificiellement la similarité)
_STOPWORDS = frozenset(
    {
        # EN
        "a",
        "an",
        "the",
        "to",
        "of",
        "and",
        "or",
        "on",
        "in",
        "at",
        "by",
        "for",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "its",
        "it",
        "this",
        "that",
        "these",
        "those",
        "but",
        "not",
        "no",
        "yes",
        "s",
        "t",
        "m",
        "re",
        "ve",
        # FR
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "et",
        "ou",
        "mais",
        "donc",
        "ni",
        "car",
        "pour",
        "par",
        "sans",
        "sur",
        "dans",
        "aux",
        "sous",
        "vers",
        "chez",
        "que",
        "qui",
        "quoi",
        "dont",
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ton",
        "son",
        "nos",
        "vos",
        "leurs",
        "est",
        "sont",
        "\u00e9tait",
        "\u00e9taient",
    }
)


def _tokens(text: str, remove_stop: bool = False) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if remove_stop:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


# ---------------------------------------------------------------------------
# Jaccard (for short texts / titles)
# ---------------------------------------------------------------------------

JACCARD_THRESHOLD = 0.55  # ≥0.55 = near-duplicate sur titres (empirique)


def jaccard_similarity(a: str, b: str) -> float:
    """
    Similarité Jaccard token-set sans stopwords.
    Retourne un float ∈ [0, 1].
    """
    set_a = set(_tokens(a, remove_stop=True))
    set_b = set(_tokens(b, remove_stop=True))
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _shingles(
    tokens: list[str], k: int = SHINGLE_SIZE, raw_text: str | None = None, char_k: int = CHAR_SHINGLE_SIZE
) -> list[str]:
    """
    Retourne un mix :
      - 1-grams (mots)
      - 2-grams (collocations)
      - k-grams (paraphrases)
      - char-k-grams (robustesse aux erreurs/orthographe) si raw_text fourni

    Pour les titres courts, les char-shingles densifient l'espace de
    features et améliorent la discrimination near-dup vs différent.
    """
    feats: list[str] = []
    if tokens:
        feats.extend(tokens)
        feats.extend(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))
        if len(tokens) >= k:
            feats.extend(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))
    # Char-shingles (casse ignorée, espaces compressés)
    if raw_text:
        s = re.sub(r"\s+", " ", raw_text.lower())
        if len(s) >= char_k:
            feats.extend(s[i : i + char_k] for i in range(len(s) - char_k + 1))
    return feats


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------


def _feature_hash(feature: str) -> int:
    """Hash 64-bit stable (md5 truncated)."""
    h = hashlib.md5(feature.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def simhash(text: str, bits: int = SIMHASH_BITS, shingle_size: int = SHINGLE_SIZE) -> int:
    """
    Calcule la signature SimHash `bits`-bit du texte.
    Retourne un int (à XORer avec un autre pour Hamming distance).
    """
    if not text or not text.strip():
        return 0

    tokens = _tokens(text)
    if not tokens:
        return 0

    features = _shingles(tokens, shingle_size, raw_text=text)
    # Accumulateur signé par bit
    v = [0] * bits
    for f in features:
        h = _feature_hash(f)
        for i in range(bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1

    # Signature = 1 si accumulé > 0, sinon 0
    sig = 0
    for i in range(bits):
        if v[i] > 0:
            sig |= 1 << i
    return sig


def hamming_distance(a: int, b: int) -> int:
    """Nombre de bits différents entre a et b."""
    return bin(a ^ b).count("1")


def is_near_duplicate(sig_a: int, sig_b: int, threshold: int = HAMMING_THRESHOLD) -> bool:
    """True si Hamming(a, b) ≤ threshold."""
    return hamming_distance(sig_a, sig_b) <= threshold


# ---------------------------------------------------------------------------
# API haut niveau : find_duplicate
# ---------------------------------------------------------------------------


def auto_threshold(text: str) -> int:
    """Seuil Hamming adaptatif (défault = candidate)."""
    n = len(_tokens(text))
    if n >= 150:
        return HAMMING_THRESHOLD_CANDIDATE
    return HAMMING_THRESHOLD_EXACT + 4  # titres: seuil intermédiaire


# ---------------------------------------------------------------------------
# Routing auto : Jaccard pour court, SimHash pour long
# ---------------------------------------------------------------------------

SHORT_TEXT_TOKEN_THRESHOLD = 30  # < 30 tokens → on utilise Jaccard


def is_near_duplicate_text(
    a: str,
    b: str,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    hamming_threshold: int | None = None,
) -> tuple[bool, float, str]:
    """
    Détection générique near-duplicate avec pipeline two-stage.

    - Textes courts (< SHORT_TEXT_TOKEN_THRESHOLD) → Jaccard direct
    - Textes longs →
        (1) SimHash pour écarter rapidement les non-candidats
        (2) si Hamming <= EXACT : duplicate confirmé (syndication stricte)
        (3) si Hamming <= CANDIDATE : vérification Jaccard pour confirmer
            (évite les faux positifs sur thèmes similaires)
        (4) sinon : différent

    Returns
    -------
    (is_dup, score, method)
      score = similarity ∈ [0,1]
      method = "jaccard" | "simhash_exact" | "simhash+jaccard" | "simhash_far"
    """
    n_a = len(_tokens(a))
    n_b = len(_tokens(b))
    use_jaccard = min(n_a, n_b) < SHORT_TEXT_TOKEN_THRESHOLD

    if use_jaccard:
        sim = jaccard_similarity(a, b)
        return sim >= jaccard_threshold, sim, "jaccard"

    # Long texts : pipeline SimHash → Jaccard
    sa, sb = simhash(a), simhash(b)
    d = hamming_distance(sa, sb)

    if d <= HAMMING_THRESHOLD_EXACT:
        return True, 1.0 - (d / SIMHASH_BITS), "simhash_exact"

    if d <= HAMMING_THRESHOLD_CANDIDATE:
        # Candidat : confirmer avec Jaccard (évite faux positifs sur thèmes
        # proches mais articles distincts)
        jac = jaccard_similarity(a, b)
        if jac >= jaccard_threshold:
            return True, jac, "simhash+jaccard"
        return False, jac, "simhash_candidate_rejected"

    return False, 1.0 - (d / SIMHASH_BITS), "simhash_far"


# ---------------------------------------------------------------------------
# Extension future : dédup sémantique via embeddings
# ---------------------------------------------------------------------------
# Pour détecter les paraphrases agressives (même info, texte entièrement
# réécrit), il faut un dédup sémantique basé sur les embeddings :
#   cos(emb(a), emb(b)) >= 0.92 = near-duplicate.
# Coût : 1 forward MiniLM par article (≈10 ms CPU).
# Implémentation : réutiliser EmbeddingFunction de rag_store.py et comparer
# au top-K des articles récents (< 48h) du même ticker.
# Ce n'est pas fait ici pour éviter une dépendance croisée RAG <-> ingestion.


def find_near_duplicate(
    new_text: str,
    existing_signatures: Iterable[tuple[str, int]],
    threshold: int | None = None,
) -> tuple[str, int] | None:
    """
    Cherche un near-duplicate parmi les signatures existantes.

    Parameters
    ----------
    new_text : texte de l'article candidat (titre + extrait body suffit)
    existing_signatures : iterable de (id, signature_int) déjà en base
    threshold : Hamming max pour considérer duplicate

    Returns
    -------
    (id, distance) du plus proche si dup trouvé, None sinon.
    """
    if threshold is None:
        threshold = auto_threshold(new_text)

    sig_new = simhash(new_text)
    if sig_new == 0:
        return None

    best = None
    best_dist = threshold + 1
    for eid, esig in existing_signatures:
        d = hamming_distance(sig_new, esig)
        if d <= threshold and d < best_dist:
            best = (eid, d)
            best_dist = d
    return best


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # === Titres courts : Jaccard ===
    print("=" * 60)
    logger.info("TITRES (route -> Jaccard)")
    print("=" * 60)
    a = "Apple beats Q3 earnings estimates on strong iPhone sales"
    b = "Apple's Q3 earnings top estimates driven by iPhone demand"
    b2 = "Apple Q3 earnings beat estimates on iPhone sales strength"
    c = "Tesla recalls 1.2M vehicles over software glitch"

    for pair in [(a, b, "paraphrase moyenne"), (a, b2, "paraphrase forte"), (a, c, "sans rapport")]:
        is_dup, score, method = is_near_duplicate_text(pair[0], pair[1])
        logger.info(f"  [{method:7}] sim={score:.3f}  dup={is_dup}  <- {pair[2]}")

    # === Bodies longs : SimHash ===
    print("\n" + "=" * 60)
    logger.info("BODIES (route -> SimHash)")
    print("=" * 60)
    body_a = (
        "Apple reported fiscal Q3 earnings that beat Wall Street estimates, "
        "driven by stronger than expected iPhone sales in emerging markets. "
        "Revenue came in at $81.8 billion versus $81.3 billion expected. "
        "CEO Tim Cook cited services growth and India expansion. Margins "
        "improved 80 basis points year-over-year. Services revenue reached "
        "$21.2 billion marking an all-time high. Analysts praised the quarter "
        "noting resilience amid consumer spending pressure."
    )
    body_b = (
        "Apple topped fiscal Q3 estimates as iPhone revenue grew in emerging "
        "markets. Total revenue was $81.8B vs the $81.3B consensus. Cook "
        "highlighted India and services expansion. Gross margin expanded 80 "
        "bps YoY. Services hit a record $21.2B. Analysts welcomed resilience "
        "amid tightening consumer spending environment for the quarter."
    )
    body_c = (
        "Tesla announced a voluntary recall of 1.2 million vehicles due to a "
        "software glitch affecting the autopilot feature. NHTSA had opened an "
        "investigation last month. The fix will be delivered over-the-air at "
        "no cost to owners. Shares fell 3% in after-hours trading."
    )
    for body_pair in [(body_a, body_b, "paraphrase body"), (body_a, body_c, "bodies distincts")]:
        is_dup, score, method = is_near_duplicate_text(body_pair[0], body_pair[1])
        logger.info(f"  [{method:7}] sim={score:.3f}  dup={is_dup}  <- {body_pair[2]}")
