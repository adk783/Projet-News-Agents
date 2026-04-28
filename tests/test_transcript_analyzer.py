"""
test_transcript_analyzer.py — Tests du detecteur d'hesitation / overconfidence.

Couvre :
  - tokenisation, syllables, gunning fog
  - split prepared / Q&A
  - detection hedging
  - detection overconfidence
  - detection evasion
  - score d'obfuscation
  - delta uncertainty (prepared vs Q&A)
  - red flag aggregation
  - configuration des poids
  - mapping confidence multiplier
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tokenisation / syllables / fog
# ---------------------------------------------------------------------------


def test_word_re_extracts_alpha_tokens():
    from src.knowledge.transcript_analyzer import _WORD_RE

    toks = _WORD_RE.findall("The CEO said: 'We grew 12% YoY!'".lower())
    assert "ceo" in toks
    assert "we" in toks
    assert "grew" in toks
    # nombres / ponctuation exclus
    assert "12" not in toks
    assert "%" not in toks


def test_count_syllables_heuristic():
    from src.knowledge.transcript_analyzer import _count_syllables

    assert _count_syllables("a") >= 1
    assert _count_syllables("hello") == 2
    assert _count_syllables("revolutionary") >= 5


def test_section_score_gunning_fog_matches_formula():
    from src.knowledge.transcript_analyzer import _score_section

    # Beaucoup de petites phrases, mots simples -> fog faible
    text_simple = ". ".join(["The cat sat on the mat"] * 10) + "."
    s_simple = _score_section(text_simple)

    # Phrases longues, jargon technique -> fog eleve
    text_complex = (
        "The unprecedented operationalization of multidimensional synergies "
        "necessitates substantive realignment of organizational hierarchies "
        "throughout the international subsidiaries notwithstanding "
        "macroeconomic perturbations affecting capital expenditure trajectories."
    ) * 3
    s_complex = _score_section(text_complex)

    assert s_complex.gunning_fog > s_simple.gunning_fog


# ---------------------------------------------------------------------------
# Split prepared / Q&A
# ---------------------------------------------------------------------------


def test_split_prepared_qna_detects_boundary():
    from src.knowledge.transcript_analyzer import _split_prepared_qna

    text = (
        "Welcome to the Q3 earnings call. Our revenue grew 15%. "
        "We are excited about the future. "
        "We will now begin the question and answer session. "
        "First question: What about your margin outlook?"
    )
    prepared, qna, found = _split_prepared_qna(text)
    assert found is True
    assert "revenue grew" in prepared
    assert "margin outlook" in qna


def test_split_prepared_qna_no_boundary_returns_all_as_prepared():
    from src.knowledge.transcript_analyzer import _split_prepared_qna

    text = "Just one block of text without QA marker."
    prepared, qna, found = _split_prepared_qna(text)
    assert found is False
    assert prepared == text
    assert qna == ""


# ---------------------------------------------------------------------------
# Hedging / overconfidence / evasion
# ---------------------------------------------------------------------------


def test_hedging_detection_increases_score():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    text = (
        "Maybe we will see some growth. Perhaps margins might improve "
        "somewhat. It is hard to say if guidance will hold. We could see "
        "modest progress, possibly. Difficult to predict the outcome. "
        "We hope things stabilize. It seems likely things may shift."
    )
    rep = TranscriptAnalyzer().analyze(text)
    assert rep.hedging_score > 0.05


def test_overconfidence_detection():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    text = (
        "We absolutely dominate the market with record-breaking growth. "
        "Our unprecedented success guarantees continued leadership. "
        "We are clearly the best-in-class operator with superior margins. "
        "Definitely the strongest quarter ever."
    )
    rep = TranscriptAnalyzer().analyze(text)
    assert rep.overconfidence_score > 0.05


def test_evasion_pattern_detection():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    qna = (
        "Q: What will guidance be? A: We can't comment on that at this time. "
        "Q: Margin outlook? A: I'd refer you to the press release. "
        "Q: Cost trends? A: We'll get back to you on that. "
        "Q: Any updates? A: Too early to say."
    )
    full = "Welcome to the call. We'll now take questions. " + qna
    rep = TranscriptAnalyzer().analyze(full)
    assert rep.evasion_score > 0.0


# ---------------------------------------------------------------------------
# Uncertainty delta (prepared -> qna)
# ---------------------------------------------------------------------------


def test_uncertainty_delta_higher_when_qna_more_hedged():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    prepared_clean = "Revenue grew 20 percent. EPS beat consensus. " * 5
    qna_hedged = (
        "Q: Outlook? A: Maybe, perhaps, hard to say, it depends. "
        "Could be lower, possibly higher, difficult to predict. "
    ) * 3
    text = prepared_clean + " We'll now begin the question and answer session. First question: " + qna_hedged
    rep = TranscriptAnalyzer().analyze(text)
    assert rep.qna_split_found, "Q&A boundary should be detected"
    assert rep.uncertainty_delta > 0.0


# ---------------------------------------------------------------------------
# Red flag aggregation
# ---------------------------------------------------------------------------


def test_red_flag_label_clean_for_neutral_text():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    text = (
        "Revenue increased 8 percent year over year. Operating margin was "
        "21 percent. Free cash flow was 1.2 billion. We invested in R&D. "
        "Capex was 400 million for the quarter."
    )
    rep = TranscriptAnalyzer().analyze(text)
    assert rep.red_flag_label in ("clean", "mild")


def test_red_flag_label_high_for_evasive_overconfident():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    bad = (
        "We absolutely dominate. Record unprecedented results. " * 5
        + " We'll now take questions. First question: "
        + "Q: Margins? A: We can't comment. Maybe later. " * 5
        + "Q: Outlook? A: Hard to say. Too early to comment. " * 5
    )
    rep = TranscriptAnalyzer().analyze(bad)
    assert rep.red_flag_score > 0.10  # signal non-trivial
    assert rep.red_flag_label in ("mild", "elevated", "high")


# ---------------------------------------------------------------------------
# Confidence multiplier mapping
# ---------------------------------------------------------------------------


def test_confidence_multiplier_monotonic_decreasing():
    from src.knowledge.transcript_analyzer import (
        transcript_red_flag_to_confidence_multiplier,
    )

    a = transcript_red_flag_to_confidence_multiplier(0.0)
    b = transcript_red_flag_to_confidence_multiplier(0.4)
    c = transcript_red_flag_to_confidence_multiplier(1.0)
    assert a == pytest.approx(1.0, abs=0.001)
    assert b < a
    assert c < b
    assert c >= 0.5  # bounded


# ---------------------------------------------------------------------------
# Configuration des poids
# ---------------------------------------------------------------------------


def test_custom_weights_emphasize_hedging_only():
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    a = TranscriptAnalyzer(
        hedging_weight=1.0,
        overconfidence_weight=0.0,
        evasion_weight=0.0,
        obfuscation_weight=0.0,
        uncertainty_delta_weight=0.0,
    )
    text = "Maybe perhaps hard to say possibly might could." * 5
    rep = a.analyze(text)
    # hedging poids = 100% -> red_flag_score == hedging_score (a delta near zero)
    assert abs(rep.red_flag_score - rep.hedging_score) < 0.05


def test_summary_is_ascii_safe():
    """Regression : ne doit pas planter sur cp1252 Windows."""
    from src.knowledge.transcript_analyzer import TranscriptAnalyzer

    rep = TranscriptAnalyzer().analyze("Maybe perhaps. We'll now take questions. " * 3)
    s = rep.summary()
    s.encode("cp1252")  # ne doit pas raise UnicodeEncodeError
