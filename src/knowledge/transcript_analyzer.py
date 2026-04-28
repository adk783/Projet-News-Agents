"""
transcript_analyzer.py — Analyse comportementale des Earnings Call Transcripts.

PROBLEME
--------
Le tone scoring Loughran-McDonald (earnings_calls.py) capture le sentiment
lexical (positif/negatif), mais PAS les signaux COMPORTEMENTAUX :
  - Un CEO qui HESITE ("maybe", "we'll see", "it's hard to say")
  - Un CEO qui affiche de l'EXCES de confiance ("record", "unprecedented",
    "without a doubt") — souvent correle aux earnings misses suivants
  - Un CEO qui EVITE les questions ("let me get back to that", deflection)
  - Une DENSITE de jargon qui augmente (obfuscation)

Ces signaux sont CE QUE LES LLM EXCELLENT a detecter : ils voient le delta
entre ce qui est DIT et ce qui est OMIS.

BASE SCIENTIFIQUE
-----------------
- Larcker & Zakolyukina (2012). "Detecting Deceptive Discussions in Conference
  Calls." Journal of Accounting Research. AUC 0.56 en linguistique pure,
  signal faible mais orthogonal au lexical.
- Hobson et al. (2012). "Analyzing Speech to Detect Financial Misreporting."
  Journal of Accounting Research. Les markers vocaux (pas applicables ici)
  + markers textuels.
- Price et al. (2012). "Earnings conference calls and stock returns: The
  incremental informativeness of textual tone." Journal of Banking & Finance.
- Bushee et al. (2018). "Linguistic Complexity in Firm Disclosures: Obfuscation
  or Information?" Journal of Accounting Research. La complexite augmente
  quand les managers veulent cacher de l'info.
- DeHaan et al. (2019). "Do Weather-Induced Moods Affect the Processing of
  Earnings Disclosures?" JAR.

ARCHITECTURE
------------
TranscriptAnalyzer.analyze(text) -> TranscriptAnalysis
  - Prepared vs Q&A : separation en deux phases (prepared = scripted,
    Q&A = spontaneous => plus revelateur).
  - Hedging score : fraction de mots hedging sur total.
  - Overconfidence score : fraction de superlatifs + quantifieurs absolus.
  - Evasion score : Q&A non-answers, redirection patterns.
  - Obfuscation score : Gunning-Fog index + jargon density.
  - Uncertainty delta : hedging(Q&A) - hedging(prepared) > 0.02 => signal.

Les scores sont INDEPENDANTS du ticker ou du secteur pour etre reutilisables
dans tout le pipeline.

REUTILISATION
-------------
Fonctionne sur :
  - Earnings Call transcripts (SEC Form 8-K 2.02)
  - Analyst day transcripts
  - M&A conference calls
  - Toute prose spontanee de direction

USAGE
-----
    from knowledge.transcript_analyzer import TranscriptAnalyzer
    tr = TranscriptAnalyzer()
    analysis = tr.analyze(transcript_text)
    if analysis.red_flag_score > 0.6:
        # pondere a la baisse la confidence du debatteur haussier
        ...
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexiques
# ---------------------------------------------------------------------------

# Hedging : formulation d'incertitude du speaker
# Base : Loughran-McDonald "Uncertainty" list + Hyland (2005) hedging lexicon
_HEDGING_WORDS = {
    # Incertitude directe
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "likely",
    "unlikely",
    "approximately",
    "about",
    "around",
    "roughly",
    "somewhat",
    "somehow",
    "seemingly",
    "apparently",
    # Modaux faibles
    "might",
    "could",
    "would",
    "should",
    "may",
    # Verbes de croyance/opinion
    "believe",
    "think",
    "feel",
    "assume",
    "suppose",
    "suspect",
    "expect",
    "anticipate",
    "hope",
    "guess",
    "imagine",
    # Expressions
    "kind of",
    "sort of",
    "more or less",
    "to some extent",
    "in a sense",
    "if i recall",
    "if memory serves",
    # Deflection / hedging Q&A
    "hard to say",
    "difficult to predict",
    "depends on",
    "conditional on",
    "subject to",
    "contingent",
}

# Overconfidence : langage de conviction absolue
# Base : superlatifs, intensifieurs, quantifieurs absolus
_OVERCONFIDENCE_WORDS = {
    # Superlatifs de performance
    "record",
    "unprecedented",
    "best-ever",
    "highest",
    "strongest",
    "unparalleled",
    "unmatched",
    "exceptional",
    "extraordinary",
    "remarkable",
    # Intensifieurs absolus
    "definitely",
    "certainly",
    "absolutely",
    "undoubtedly",
    "clearly",
    "obviously",
    "unquestionably",
    "without a doubt",
    "without question",
    # Quantifieurs absolus
    "always",
    "never",
    "every",
    "all",
    "none",
    "nothing",
    # Narrative de domination
    "dominant",
    "leading",
    "premier",
    "unique",
    "revolutionary",
    "transformational",
    "game-changing",
    "industry-defining",
    # Minimisation du risque
    "no risk",
    "no downside",
    "guaranteed",
    "proven",
}

# Evasion patterns : non-answers a une question d'analyste
_EVASION_PATTERNS = [
    re.compile(r"\bget back to you\b", re.I),
    re.compile(r"\bfollow.?up (?:offline|later)\b", re.I),
    re.compile(r"\btake (?:that|this) offline\b", re.I),
    re.compile(r"\bnot going to comment\b", re.I),
    re.compile(r"\bcan\'?t comment\b", re.I),
    re.compile(r"\bnot prepared to discuss\b", re.I),
    re.compile(r"\bprefer not to (?:share|disclose|comment)\b", re.I),
    re.compile(r"\bwe don\'?t (?:provide|disclose|comment on)\b", re.I),
    re.compile(r"\btoo early to (?:say|tell|comment)\b", re.I),
    re.compile(r"\bstay tuned\b", re.I),
]

# Jargon financier "vide de sens" (Bushee et al. 2018)
_VAGUE_JARGON = {
    "synergies",
    "optimization",
    "strategic",
    "leverage",
    "ecosystem",
    "framework",
    "paradigm",
    "alignment",
    "engagement",
    "penetration",
    "disruption",
    "innovation",
    "transformation",
    "realignment",
    "rightsizing",
    "streamlining",
    "headwinds",
    "tailwinds",
    "narrative",
    "thesis",
}

# Patterns de debut / fin de Q&A pour splitter prepared vs Q&A
_QA_BOUNDARY_PATTERNS = [
    re.compile(r"(question.?and.?answer|q\s*&\s*a|q\s*and\s*a)", re.I),
    re.compile(r"we\'?ll now (?:take|begin|start|open)\s+(?:the\s+)?questions?", re.I),
    re.compile(r"operator\s*[,:]?\s*(?:please\s+)?open (?:the\s+)?line", re.I),
    re.compile(r"first question", re.I),
]

# Speaker patterns (pour comptage interventions)
_SPEAKER_PATTERN = re.compile(
    r"(?:^|\n)\s*([A-Z][a-zA-Z\-\.\']+(?:\s+[A-Z][a-zA-Z\-\.\']+){1,3})\s*[-:](?:\s|\n)",
    re.MULTILINE,
)

_OPERATOR_PATTERN = re.compile(r"operator\s*[-:]", re.I)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SectionScore:
    """Scores linguistiques d'une section du transcript."""

    word_count: int = 0
    hedging_count: int = 0
    overconfidence_count: int = 0
    evasion_count: int = 0
    jargon_count: int = 0
    avg_sentence_len: float = 0.0
    complex_word_ratio: float = 0.0  # > 3 syllabes

    @property
    def hedging_ratio(self) -> float:
        return self.hedging_count / self.word_count if self.word_count else 0.0

    @property
    def overconfidence_ratio(self) -> float:
        return self.overconfidence_count / self.word_count if self.word_count else 0.0

    @property
    def jargon_ratio(self) -> float:
        return self.jargon_count / self.word_count if self.word_count else 0.0

    @property
    def gunning_fog(self) -> float:
        """Index de Gunning-Fog : annees d'education requises."""
        return 0.4 * (self.avg_sentence_len + 100.0 * self.complex_word_ratio)


@dataclass
class TranscriptAnalysis:
    """Resultat complet de l'analyse comportementale."""

    prepared: SectionScore
    qna: SectionScore
    full: SectionScore

    # Scores agreges normalises [0, 1]
    hedging_score: float = 0.0  # plus eleve = plus d'hesitation
    overconfidence_score: float = 0.0  # plus eleve = plus d'exces de conviction
    evasion_score: float = 0.0  # plus eleve = plus d'evitement de questions
    obfuscation_score: float = 0.0  # plus eleve = jargon/complexite elevee
    uncertainty_delta: float = 0.0  # hedging(qna) - hedging(prepared)

    # Red flag global [0,1]
    red_flag_score: float = 0.0
    red_flag_label: str = "clean"  # clean | mild | elevated | high

    # Patterns detectes (pour explicabilite)
    evasion_patterns_found: List[str] = field(default_factory=list)
    top_hedging_words: List[str] = field(default_factory=list)
    top_overconfidence_words: List[str] = field(default_factory=list)

    # Meta
    qna_split_found: bool = False  # True si separation prepared/qna reussie
    total_words: int = 0

    def summary(self) -> str:
        return (
            f"hedging={self.hedging_score:.2f} overconf={self.overconfidence_score:.2f} "
            f"evasion={self.evasion_score:.2f} obfusc={self.obfuscation_score:.2f} "
            f"d_uncert={self.uncertainty_delta:+.3f} red_flag={self.red_flag_score:.2f} "
            f"[{self.red_flag_label}]"
        )

    def to_prompt_block(self) -> str:
        """Format condense pour injection dans le prompt des debatteurs."""
        lines = [
            "<transcript_behavioral>",
            f"  hedging_ratio={self.hedging_score:.3f}",
            f"  overconfidence_ratio={self.overconfidence_score:.3f}",
            f"  evasion_events={len(self.evasion_patterns_found)}",
            f"  obfuscation={self.obfuscation_score:.3f}",
            f"  uncertainty_delta_QnA_vs_prepared={self.uncertainty_delta:+.3f}",
            f"  red_flag_score={self.red_flag_score:.2f} ({self.red_flag_label})",
        ]
        if self.evasion_patterns_found:
            lines.append(f"  evasion_examples={self.evasion_patterns_found[:3]}")
        if self.top_hedging_words:
            lines.append(f"  top_hedging={self.top_hedging_words[:5]}")
        if self.top_overconfidence_words:
            lines.append(f"  top_overconf={self.top_overconfidence_words[:5]}")
        lines.append("</transcript_behavioral>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\'\-]*")
_SENT_RE = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    """Approximation rapide du nombre de syllabes (heuristic vowel groups)."""
    w = word.lower()
    if not w:
        return 0
    # Retire 'e' final muet
    if w.endswith("e") and len(w) > 2:
        w = w[:-1]
    # Compte les groupes de voyelles
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def _split_prepared_qna(text: str) -> Tuple[str, str, bool]:
    """
    Split heuristique du transcript en (prepared_remarks, qna_section, found).

    Cherche la premiere boundary Q&A ; tout ce qui precede = prepared.
    Si aucune boundary trouvee, retourne (full_text, "", False).
    """
    best_idx = -1
    for pattern in _QA_BOUNDARY_PATTERNS:
        m = pattern.search(text)
        if m:
            if best_idx == -1 or m.start() < best_idx:
                best_idx = m.start()

    if best_idx < 0:
        return text, "", False
    return text[:best_idx], text[best_idx:], True


def _score_section(text: str) -> SectionScore:
    """Calcule les scores linguistiques d'une section."""
    score = SectionScore()
    if not text or not text.strip():
        return score

    tokens = _WORD_RE.findall(text.lower())
    score.word_count = len(tokens)
    if score.word_count == 0:
        return score

    # Hedging / overconfidence : bag-of-words + bigrams pour expressions
    text_low = " " + text.lower() + " "
    for w in _HEDGING_WORDS:
        if " " in w:
            score.hedging_count += text_low.count(" " + w + " ")
        else:
            score.hedging_count += sum(1 for t in tokens if t == w)

    for w in _OVERCONFIDENCE_WORDS:
        if " " in w:
            score.overconfidence_count += text_low.count(" " + w + " ")
        else:
            score.overconfidence_count += sum(1 for t in tokens if t == w)

    for w in _VAGUE_JARGON:
        score.jargon_count += sum(1 for t in tokens if t == w)

    # Evasion patterns (Q&A typiquement)
    for pat in _EVASION_PATTERNS:
        score.evasion_count += len(pat.findall(text))

    # Sentence length + complex words
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if sentences:
        total_len = sum(len(_WORD_RE.findall(s)) for s in sentences)
        score.avg_sentence_len = total_len / len(sentences)

    complex_words = sum(1 for t in tokens if _count_syllables(t) >= 3)
    score.complex_word_ratio = complex_words / score.word_count

    return score


def _normalize_score(raw: float, target_max: float) -> float:
    """Normalise [0, target_max] -> [0, 1] avec saturation."""
    if target_max <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / target_max))


def _top_k_occurrences(text: str, vocab: set, k: int = 5) -> List[str]:
    """Retourne les k mots du vocab les plus frequents dans text."""
    tokens = _WORD_RE.findall(text.lower())
    counts: Dict[str, int] = {}
    for t in tokens:
        if t in vocab:
            counts[t] = counts.get(t, 0) + 1
    # On ajoute aussi les multi-words
    text_low = " " + text.lower() + " "
    for w in vocab:
        if " " in w:
            c = text_low.count(" " + w + " ")
            if c > 0:
                counts[w] = c
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TranscriptAnalyzer:
    """Analyse comportementale d'un transcript Earnings Call."""

    # Seuils de calibration (derives des moyennes sur 10-K / 8-K Item 2.02)
    # Volontairement bas : on normalise sur des deciles empiriques.
    HEDGING_TYPICAL_MAX = 0.035  # 3.5% = deja eleve
    OVERCONF_TYPICAL_MAX = 0.015  # 1.5% = deja tres eleve
    EVASION_TYPICAL_MAX = 8.0  # 8 non-answers = eleve
    OBFUSC_TYPICAL_MAX = 18.0  # Gunning-Fog 18 = obscure

    def __init__(
        self,
        hedging_weight: float = 0.30,
        overconfidence_weight: float = 0.25,
        evasion_weight: float = 0.25,
        obfuscation_weight: float = 0.10,
        uncertainty_delta_weight: float = 0.10,
    ):
        total = hedging_weight + overconfidence_weight + evasion_weight + obfuscation_weight + uncertainty_delta_weight
        if abs(total - 1.0) > 0.001:
            logger.warning("[TranscriptAnalyzer] poids != 1.0 (sum=%.3f)", total)
        self.w_hedge = hedging_weight
        self.w_overconf = overconfidence_weight
        self.w_evasion = evasion_weight
        self.w_obfusc = obfuscation_weight
        self.w_udelta = uncertainty_delta_weight

    # -- Public API --

    def analyze(self, text: str) -> TranscriptAnalysis:
        """Analyse comportementale complete d'un transcript."""
        if not text or not text.strip():
            return TranscriptAnalysis(
                prepared=SectionScore(),
                qna=SectionScore(),
                full=SectionScore(),
                red_flag_label="clean",
            )

        prepared_txt, qna_txt, qna_found = _split_prepared_qna(text)
        full = _score_section(text)
        prepared = _score_section(prepared_txt)
        qna = _score_section(qna_txt) if qna_found else SectionScore()

        # -- Scores normalises --
        hedging_score = _normalize_score(full.hedging_ratio, self.HEDGING_TYPICAL_MAX)
        overconf_score = _normalize_score(full.overconfidence_ratio, self.OVERCONF_TYPICAL_MAX)
        evasion_score = _normalize_score(full.evasion_count, self.EVASION_TYPICAL_MAX)
        obfusc_score = _normalize_score(full.gunning_fog, self.OBFUSC_TYPICAL_MAX)

        # -- Uncertainty delta : Q&A vs prepared remarks --
        # Un CEO qui hesite plus en Q&A qu'en prepared => signal fort
        uncertainty_delta = 0.0
        if qna_found and prepared.word_count > 0 and qna.word_count > 0:
            uncertainty_delta = qna.hedging_ratio - prepared.hedging_ratio
        udelta_score = _normalize_score(uncertainty_delta, 0.02)  # 2% delta = eleve

        # -- Red flag score pondere --
        red_flag = (
            self.w_hedge * hedging_score
            + self.w_overconf * overconf_score
            + self.w_evasion * evasion_score
            + self.w_obfusc * obfusc_score
            + self.w_udelta * udelta_score
        )

        label = self._flag_label(red_flag)

        # -- Patterns detectes (explicabilite) --
        evasion_examples = []
        for pat in _EVASION_PATTERNS:
            for m in pat.finditer(text):
                snippet = text[max(0, m.start() - 20) : m.end() + 20]
                evasion_examples.append(snippet.strip())
                if len(evasion_examples) >= 5:
                    break
            if len(evasion_examples) >= 5:
                break

        top_hedge = _top_k_occurrences(text, _HEDGING_WORDS, k=5)
        top_overconf = _top_k_occurrences(text, _OVERCONFIDENCE_WORDS, k=5)

        return TranscriptAnalysis(
            prepared=prepared,
            qna=qna,
            full=full,
            hedging_score=hedging_score,
            overconfidence_score=overconf_score,
            evasion_score=evasion_score,
            obfuscation_score=obfusc_score,
            uncertainty_delta=uncertainty_delta,
            red_flag_score=red_flag,
            red_flag_label=label,
            evasion_patterns_found=evasion_examples,
            top_hedging_words=top_hedge,
            top_overconfidence_words=top_overconf,
            qna_split_found=qna_found,
            total_words=full.word_count,
        )

    def _flag_label(self, red_flag: float) -> str:
        if red_flag < 0.25:
            return "clean"
        if red_flag < 0.45:
            return "mild"
        if red_flag < 0.65:
            return "elevated"
        return "high"


# ---------------------------------------------------------------------------
# Confidence multiplier (plug pour le debate augmentation)
# ---------------------------------------------------------------------------


def transcript_red_flag_to_confidence_multiplier(red_flag_score: float) -> float:
    """
    red_flag_score [0,1] -> multiplicateur de confiance sur la these HAUSSIERE
    associee a ce transcript.

    0.00 -> 1.00 (aucune penalite)
    0.25 -> 0.95
    0.50 -> 0.85
    0.75 -> 0.70
    1.00 -> 0.60

    Ne s'applique qu'a la these haussiere : les behavioral red flags sont
    un VENT CONTRAIRE aux arguments bull, mais pas a un bear (un bear
    malhonnete est plus rare et pas forcement refletable linguistiquement).
    """
    r = max(0.0, min(1.0, red_flag_score))
    # Quadratique : effet croissant quand r augmente
    return 1.0 - 0.4 * (r**1.5)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario : CEO confiant mais evasif en Q&A
    transcript = """
    John Smith - CEO

    Good morning everyone. Thank you for joining our Q3 earnings call.
    I'm pleased to report that we delivered a record quarter. Revenue
    was up 15%, reaching an unprecedented $12.5 billion. Our services
    segment absolutely crushed expectations with best-ever margins.
    We are clearly the dominant player in our industry, and our
    leading position is unparalleled.

    Our guidance for Q4 is strong. We definitely see continued momentum
    and expect record growth. There is no doubt our strategy is working.

    Operator: We'll now take questions.

    First question comes from Jane Doe, Morgan Stanley.

    Jane Doe: Thank you. Can you give us more color on the China
    segment? It seemed weaker than expected.

    John Smith: Well, that's a good question. I think the China situation
    is somewhat complex. We believe there may be some headwinds but it's
    difficult to predict. We might see some pressure in the near term,
    but perhaps things will improve. I'd rather not go into too much
    detail. We can get back to you with more specifics offline.

    Jane Doe: On the gross margin decline - was that one-time or structural?

    John Smith: Hard to say at this point. It depends on a number of
    factors. I think we'll get back to you in the next quarter with
    more color. Too early to tell whether the trend continues. We prefer
    not to disclose more at this time.
    """

    analyzer = TranscriptAnalyzer()
    analysis = analyzer.analyze(transcript)

    logger.info("=== Transcript Analysis ===")
    print(analysis.summary())
    logger.info(f"\nqna_split_found: {analysis.qna_split_found}")
    logger.info(f"total_words: {analysis.total_words}")
    print(
        f"\nSection prepared : hedge_ratio={analysis.prepared.hedging_ratio:.4f} "
        f"overconf_ratio={analysis.prepared.overconfidence_ratio:.4f}"
    )
    print(
        f"Section Q&A      : hedge_ratio={analysis.qna.hedging_ratio:.4f} "
        f"overconf_ratio={analysis.qna.overconfidence_ratio:.4f}"
    )
    logger.info(f"\nUncertainty delta (Q&A - prepared) : {analysis.uncertainty_delta:+.4f}")
    logger.info(f"Red flag label : {analysis.red_flag_label}")
    logger.info(f"\nTop hedging : {analysis.top_hedging_words}")
    logger.info(f"Top overconfidence : {analysis.top_overconfidence_words}")
    logger.info("Evasion examples :")
    for e in analysis.evasion_patterns_found[:3]:
        logger.info(f"  - {e[:80]}")

    logger.info("\n--- Prompt block ---")
    print(analysis.to_prompt_block())

    # Confidence multiplier
    mult = transcript_red_flag_to_confidence_multiplier(analysis.red_flag_score)
    logger.info(f"\nBull confidence multiplier : {mult:.2f}")

    # Validation : on doit avoir overconf eleve (prepared) ET hedge eleve (Q&A)
    assert analysis.prepared.overconfidence_ratio > 0.005, "Prepared should show overconfidence"
    assert analysis.qna.hedging_ratio > 0.02, "Q&A should show hedging"
    assert analysis.uncertainty_delta > 0.0, "Q&A should hedge more than prepared"
    assert len(analysis.evasion_patterns_found) >= 2, "Should detect evasion patterns"
    logger.info("\nOK - tous les signaux comportementaux detectes correctement")
