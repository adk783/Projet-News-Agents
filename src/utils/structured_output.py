"""
structured_output.py — Schemas Pydantic + helpers Instructor pour sorties LLM.

POURQUOI
--------
Les prompts actuels demandent "JSON output" et parsent la reponse avec
`json.loads` entoure d'un try/except. Cela se casse regulierement :
  - LLM ajoute du markdown (```json ... ```)
  - Trailing text apres la derniere "}"
  - Guillemets francais ou caracteres de contr\u00f4le non-escaped
  - Champs manquants ou type-mismatchs silencieusement accept\u00e9s

Pydantic + Instructor eliminent ce couche de fragilite :
  - Le schema Pydantic est la source de verite (champs, types, defaults)
  - Instructor patch le client LLM pour re-tenter sur validation error
  - Les retries sont transparents avec feedback au modele

FALLBACK
--------
Instructor n'est pas toujours installable (cas CI leger, bench). On expose :
  1. Les schemas Pydantic purs (utilisables pour validation manuelle)
  2. `parse_llm_json(text, schema)` qui nettoie et parse avec tolerance
  3. Un hook `structured_call` qui utilise Instructor si dispo, sinon
     tombe sur parse_llm_json + retry manuel

SCHEMAS DEFINIS
---------------
- ArticleAnalysis     : output de l'agent ABSA (sentiment, aspects)
- DebateArgument      : argument d'un debatteur (position, confiance)
- CriticFeedback      : sortie de l'Actor-Critic (biais, faiblesses)
- VerificationResult  : sortie de la Chain-of-Verification
- FinalDecision       : decision finale de la boucle debat

REFERENCES
----------
- Jason Liu (2024). "Instructor: Structured outputs with LLMs."
  https://github.com/jxnl/instructor
- Shinn et al. (2023). "Reflexion: Language Agents with Verbal RL."
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
import re
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator

    _PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDANTIC_AVAILABLE = False

    # Stubs minimaux pour permettre l'import du module en l'absence de pydantic
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        def model_dump_json(self):
            return json.dumps(self.model_dump(), ensure_ascii=False)

    def Field(default=None, **kw):  # type: ignore
        return default

    class ValidationError(Exception):  # type: ignore
        pass

    def field_validator(*args, **kw):  # type: ignore
        def deco(fn):
            return fn

        return deco


try:
    import instructor  # type: ignore

    _INSTRUCTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _INSTRUCTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Signal(str, Enum):
    ACHAT = "Achat"
    VENTE = "Vente"
    NEUTRE = "Neutre"
    HOLD = "Hold"


class Position(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------


class ArticleAnalysis(BaseModel):
    """Sortie de l'agent ABSA : sentiment + aspects."""

    ticker: str
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(max_length=500)
    aspects: List[str] = Field(default_factory=list, max_length=8)
    key_numbers: List[str] = Field(default_factory=list, max_length=6)
    time_horizon_days: Optional[int] = Field(default=None, ge=0, le=365)

    if _PYDANTIC_AVAILABLE:

        @field_validator("summary")
        @classmethod
        def _summary_nonempty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("summary must be non-empty")
            return v.strip()


class DebateArgument(BaseModel):
    """Argument produit par un debatteur pendant une ronde."""

    debater_id: str
    round: int = Field(ge=0)
    position: Position
    thesis: str = Field(max_length=1000)
    evidence: List[str] = Field(default_factory=list, max_length=8)
    counters_opponent: Optional[str] = Field(default=None, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class Claim(BaseModel):
    """Une affirmation factuelle extraite d'un texte, pour verification."""

    text: str = Field(max_length=300)
    category: Literal["numeric", "entity", "causal", "temporal", "other"] = "other"
    verifiable: bool = True


class VerificationResult(BaseModel):
    """Resultat d'une Chain-of-Verification sur une liste de claims."""

    claim: Claim
    verdict: Literal["supported", "contradicted", "unverifiable"]
    evidence: str = Field(max_length=500, default="")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class CriticFeedback(BaseModel):
    """Sortie de l'Actor-Critic : ce qui cloche dans l'argument produit."""

    target_debater_id: str
    target_round: int
    biases_detected: List[str] = Field(default_factory=list, max_length=6)
    logical_gaps: List[str] = Field(default_factory=list, max_length=6)
    unsupported_claims: List[str] = Field(default_factory=list, max_length=6)
    severity: Severity = Severity.LOW
    suggested_revisions: List[str] = Field(default_factory=list, max_length=6)


class FinalDecision(BaseModel):
    """Decision finale du pipeline debat."""

    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=1000)
    dissenting_opinion: Optional[str] = Field(default=None, max_length=500)
    trigger_conditions: List[str] = Field(default_factory=list, max_length=5)
    risk_flags: List[str] = Field(default_factory=list, max_length=5)


# ---------------------------------------------------------------------------
# Parsing tolerant
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    s = _FENCE_RE.sub("", text or "").strip()
    return s


def _extract_first_json_object(text: str) -> Optional[str]:
    """Renvoie le premier objet JSON (ou tableau) equilibre trouve dans le texte."""
    if not text:
        return None
    # trouver la premiere '{' ou '['
    start = -1
    opener = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start == -1:
        return None
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_json(text: str, schema: Optional[type] = None):
    """
    Parse tolerant d'une sortie LLM supposee JSON.

    Etapes :
      1. Strip markdown fences
      2. Extract first balanced {...} ou [...]
      3. json.loads
      4. Si schema fourni et pydantic dispo, valide avec schema(**obj)

    Leve ValueError en cas d'echec.
    """
    if not text:
        raise ValueError("empty text")
    cleaned = _strip_markdown_fences(text)
    candidate = _extract_first_json_object(cleaned) or cleaned
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"json decode failed: {e}") from e

    if schema is None or not _PYDANTIC_AVAILABLE:
        return obj

    try:
        return schema(**obj) if isinstance(obj, dict) else schema.model_validate(obj)
    except ValidationError as e:
        raise ValueError(f"schema validation failed: {e}") from e
    except TypeError as e:
        raise ValueError(f"schema construction failed: {e}") from e


# ---------------------------------------------------------------------------
# Helper de retry avec feedback
# ---------------------------------------------------------------------------


def _build_retry_prompt(last_output: str, error: str, schema_hint: str) -> str:
    """Construit un message de feedback pour aider le LLM a corriger sa sortie."""
    return (
        "Ta reponse precedente n'a pas pu etre parsee.\n"
        f"Erreur : {error}\n"
        f"Reponse precedente (tronquee) :\n---\n{last_output[:800]}\n---\n"
        f"Schema attendu : {schema_hint}\n"
        "Renvoie UNIQUEMENT un JSON valide qui respecte le schema, sans "
        "markdown, sans texte autour, sans commentaire."
    )


def structured_call(
    call_llm,
    prompt: str,
    schema: type,
    *,
    max_retries: int = 2,
    schema_hint: Optional[str] = None,
):
    """
    Wrapper autour d'un `call_llm(prompt: str) -> str` qui garantit une sortie
    validee par `schema`.

    Args:
        call_llm : callable qui prend un prompt string, renvoie une string.
        prompt   : le prompt systeme/user initial.
        schema   : classe Pydantic attendue en sortie.
        max_retries : nb de tentatives supplementaires en cas d'echec de parsing.
        schema_hint : description textuelle du schema (fallback sur model_json_schema).

    Returns:
        Instance de `schema`.
    Raises:
        ValueError si echec apres max_retries.
    """
    if schema_hint is None and _PYDANTIC_AVAILABLE and hasattr(schema, "model_json_schema"):
        try:
            schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        except Exception:
            schema_hint = schema.__name__
    elif schema_hint is None:
        schema_hint = schema.__name__

    last_err = ""
    last_out = ""
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        raw = call_llm(current_prompt) or ""
        last_out = raw
        try:
            return parse_llm_json(raw, schema=schema)
        except ValueError as e:
            last_err = str(e)
            current_prompt = prompt + "\n\n" + _build_retry_prompt(raw, last_err, schema_hint)
    raise ValueError(
        f"structured_call failed after {max_retries + 1} attempts. "
        f"Last error: {last_err}. Last output: {last_out[:300]}"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"pydantic: {_PYDANTIC_AVAILABLE}  instructor: {_INSTRUCTOR_AVAILABLE}")
    print()

    # Cas 1 : JSON propre
    raw1 = '{"ticker":"AAPL","signal":"Achat","confidence":0.82,"summary":"Strong Q3"}'
    obj1 = parse_llm_json(raw1, ArticleAnalysis)
    logger.info(f"[1 JSON propre] -> {obj1}")

    # Cas 2 : JSON entoure de markdown
    raw2 = '```json\n{"ticker":"MSFT","signal":"Vente","confidence":0.6,"summary":"Miss guidance"}\n```'
    obj2 = parse_llm_json(raw2, ArticleAnalysis)
    logger.info(f"[2 fenced    ] -> {obj2}")

    # Cas 3 : texte parasite avant le JSON
    raw3 = 'Voici mon analyse :\n{"ticker":"TSLA","signal":"Neutre","confidence":0.4,"summary":"Mixed"}\n\nEnd.'
    obj3 = parse_llm_json(raw3, ArticleAnalysis)
    logger.info(f"[3 text noise] -> {obj3}")

    # Cas 4 : JSON invalide
    try:
        parse_llm_json("not json at all", ArticleAnalysis)
        logger.info("[4 invalid  ] FAIL - should have raised")
    except ValueError as e:
        logger.info(f"[4 invalid  ] OK - raised: {str(e)[:80]}")

    # Cas 5 : schema violation (confidence > 1)
    if _PYDANTIC_AVAILABLE:
        try:
            parse_llm_json('{"ticker":"X","signal":"Achat","confidence":1.5,"summary":"S"}', ArticleAnalysis)
            logger.info("[5 schema   ] FAIL - should have raised")
        except ValueError as e:
            logger.info(f"[5 schema   ] OK - raised: {str(e)[:80]}")

    # Cas 6 : structured_call retry
    attempts = {"n": 0}

    def fake_llm(p):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "not json"
        return '{"ticker":"AAPL","signal":"Achat","confidence":0.7,"summary":"OK"}'

    obj6 = structured_call(fake_llm, "prompt", ArticleAnalysis, max_retries=2)
    print(f"[6 retry    ] OK after {attempts['n']} attempts -> signal={obj6.signal}")
