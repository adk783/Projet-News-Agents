"""
debate_augmentation.py — Glue code : plug CriticAgent + VerifierAgent dans
le debat AutoGen existant sans tout recrire.

POURQUOI
--------
`agent_debat.py` utilise AutoGen RoundRobinGroupChat + MaxMessageTermination +
un Shared Scratchpad XML. Cela fonctionne. On ne veut pas tout casser pour
brancher les nouveaux agents de raisonnement (Critic, Verifier, Reflector).

A la place, on ajoute une etape d'AUGMENTATION apres le debat mais AVANT le
Consensus :

    debat AutoGen  -->  scratchpad XML
                        |
                        v
                     [AUGMENTATION]
                     - CriticAgent sur chaque argument post-debate
                     - VerifierAgent sur les claims factuels
                     - adjusted_confidence(aggregated)
                        |
                        v
                     Consensus (lit scratchpad + augmentation)

ADAPTATEUR LLM
--------------
Les agents Critic/Verifier attendent `call_llm: (prompt) -> str`. Les clients
AutoGen sont des `OpenAIChatCompletionClient`. On fournit `wrap_autogen_client`
qui convertit l'un en l'autre (async → sync via event loop local).
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

# Imports absolus (cf. ADR-008, mode editable `pip install -e .`).
from src.utils.structured_output import (
    CriticFeedback,
    DebateArgument,
    Position,
    Severity,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptateur AutoGen -> call_llm synchrone
# ---------------------------------------------------------------------------


def wrap_autogen_client(client: Any, timeout: float = 45.0) -> Callable[[str], str]:
    """
    Convertit un OpenAIChatCompletionClient AutoGen en callable sync
    `call_llm(prompt: str) -> str`.

    Fait tourner client.create([UserMessage]) dans un event loop. Si un
    event loop tourne deja (appel depuis coroutine), on schedule la coroutine
    via `run_coroutine_threadsafe` sur une thread worker dediee.
    """
    # Import lazy pour eviter la dependance dure sur autogen_core
    try:
        from autogen_core.models import UserMessage  # type: ignore
    except Exception:
        UserMessage = None  # type: ignore

    def _call(prompt: str) -> str:
        if UserMessage is None:
            raise RuntimeError("autogen_core non installe — wrap_autogen_client indisponible")
        msgs = [UserMessage(content=prompt, source="augmentation")]

        async def _go():
            result = await client.create(msgs)
            # result.content peut etre str ou list[FunctionCall]
            out = getattr(result, "content", "")
            if isinstance(out, list):
                out = "".join(getattr(p, "text", str(p)) for p in out)
            return str(out)

        try:
            loop = asyncio.get_running_loop()
            # Une loop tourne deja dans ce thread. On ne peut pas run_until_complete().
            # On delegue a un thread worker pour executer le run() proprement.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, asyncio.wait_for(_go(), timeout=timeout))
                return future.result()
        except RuntimeError:
            # Pas d'event loop dans ce thread -> on en cree une en toute securite
            return asyncio.run(asyncio.wait_for(_go(), timeout=timeout))

    return _call


# ---------------------------------------------------------------------------
# Extraction des arguments depuis scratchpad
# ---------------------------------------------------------------------------


def scratchpad_to_arguments(scratchpad) -> List[DebateArgument]:
    """
    Parse un SharedScratchpad (structure definie dans agent_debat.py) en liste
    de DebateArgument. On cree un DebateArgument par section x round-index.

    La scratchpad contient des listes de strings ; la confiance apparait dans
    chaque string comme "... [confiance: 0.7]".
    """
    import re as _re

    CONF_RE = _re.compile(r"\[confiance:\s*([\-\d\.]+)\]")

    out: List[DebateArgument] = []
    sections = {
        "Haussier": (getattr(scratchpad, "haussier_arguments", []), "bull"),
        "Baissier": (getattr(scratchpad, "baissier_arguments", []), "bear"),
        "Neutre": (getattr(scratchpad, "neutre_arguments", []), "neutral"),
    }
    for agent_name, (args, pos_val) in sections.items():
        for i, a in enumerate(args or []):
            m = CONF_RE.search(a or "")
            try:
                conf = float(m.group(1)) if m else 0.5
            except (ValueError, AttributeError):
                conf = 0.5
            thesis = CONF_RE.sub("", a or "").strip()
            out.append(
                DebateArgument(
                    debater_id=agent_name,
                    round=i + 1,
                    position=Position(pos_val),
                    thesis=thesis[:1000],
                    evidence=[],
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Agregation
# ---------------------------------------------------------------------------


@dataclass
class AugmentationReport:
    arguments: List[DebateArgument] = field(default_factory=list)
    critic_feedbacks: List[CriticFeedback] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)

    # Signaux aggregees
    avg_severity: float = 0.0
    max_severity: int = 0
    verification_ratio: float = 1.0
    confidence_multiplier: float = 1.0

    def to_prompt_block(self) -> str:
        """Rend le resume textuel injectable dans le prompt Consensus."""
        sev_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        lines = [
            "<augmentation>",
            f"  critic_avg_severity={self.avg_severity:.2f}",
            f"  critic_max_severity={sev_map.get(self.max_severity, '?')}",
            f"  verification_ratio={self.verification_ratio:.2%}",
            f"  confidence_multiplier={self.confidence_multiplier:.2f}",
        ]
        # Top issues par debatteur
        per_deb: dict[str, list[str]] = {}
        for fb in self.critic_feedbacks:
            bag = per_deb.setdefault(fb.target_debater_id, [])
            bag.extend(fb.biases_detected[:2])
            bag.extend(fb.logical_gaps[:1])
            bag.extend(fb.unsupported_claims[:2])
        for deb, issues in per_deb.items():
            if not issues:
                continue
            lines.append(f'  <critique agent="{deb}">')
            for issue in issues[:4]:
                safe = issue.replace("<", "(").replace(">", ")")
                lines.append(f"    <issue>{safe}</issue>")
            lines.append("  </critique>")
        # Verification contradictions
        contradicted = [r for r in self.verification_results if r.verdict == "contradicted"]
        if contradicted:
            lines.append("  <contradictions>")
            for r in contradicted[:5]:
                safe = r.claim.text.replace("<", "(").replace(">", ")")
                lines.append(f"    <claim>{safe}</claim>")
            lines.append("  </contradictions>")
        lines.append("</augmentation>")
        return "\n".join(lines)


def _compute_signals(report: AugmentationReport) -> None:
    """Remplit les champs aggreges."""
    sev_map = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
    if report.critic_feedbacks:
        sevs = [sev_map[f.severity] for f in report.critic_feedbacks]
        report.avg_severity = sum(sevs) / len(sevs)
        report.max_severity = max(sevs)
    else:
        report.avg_severity = 0.0
        report.max_severity = 0

    if report.verification_results:
        supported = sum(1 for r in report.verification_results if r.verdict == "supported")
        report.verification_ratio = supported / len(report.verification_results)
    else:
        report.verification_ratio = 1.0

    # Multiplier = produit de (1 - 0.15 * avg_sev) et (0.6 + 0.4 * ratio)
    sev_factor = max(0.5, 1.0 - 0.15 * report.avg_severity)
    ver_factor = 0.6 + 0.4 * report.verification_ratio
    report.confidence_multiplier = round(sev_factor * ver_factor, 3)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_augmentation(
    scratchpad,
    article_text: str,
    critic_call_llm: Callable[[str], str],
    verifier_extractor_llm: Optional[Callable[[str], str]] = None,
    verifier_checker_llm: Optional[Callable[[str], str]] = None,
    context: Optional[dict] = None,
    max_arguments: int = 6,
) -> AugmentationReport:
    """
    Execute la pipeline d'augmentation :
      1. Extrait DebateArgument depuis scratchpad
      2. CriticAgent.review sur chaque
      3. (optionnel) VerifierAgent sur le texte agrege du scratchpad

    Args:
        scratchpad : SharedScratchpad d'agent_debat
        article_text : sources (utilise par VerifierAgent)
        critic_call_llm : callable wrappe
        verifier_extractor_llm / verifier_checker_llm : optionnel ; si None,
            on skip la verification (juste Critic).
        context : dict additionnel pour Critic (absa_summary, event_type, ...)
        max_arguments : cap pour ne pas exploser la latence

    Returns:
        AugmentationReport
    """
    # Import dynamique pour eviter les dependances circulaires (utils <-> agents).
    # Imports absolus uniquement (cf. ADR-008).
    from src.agents.agent_critic import CriticAgent
    from src.agents.agent_verifier import VerifierAgent

    report = AugmentationReport(arguments=scratchpad_to_arguments(scratchpad))
    if not report.arguments:
        logger.info("[Augmentation] scratchpad vide, rien a auditer")
        _compute_signals(report)
        return report

    # Limiter les arguments auditees (latence)
    to_audit = report.arguments[:max_arguments]

    # --- Critic pass ----------------------------------------------------------
    critic = CriticAgent(call_llm=critic_call_llm)
    for arg in to_audit:
        try:
            fb = critic.review(argument=arg, context=context)
            report.critic_feedbacks.append(fb)
        except Exception as e:
            logger.warning("[Augmentation] critic failure on %s r=%d: %s", arg.debater_id, arg.round, e)

    # --- Verifier pass (optionnel) -------------------------------------------
    if verifier_extractor_llm is not None and verifier_checker_llm is not None:
        try:
            verifier = VerifierAgent(
                claim_extractor_llm=verifier_extractor_llm,
                verifier_llm=verifier_checker_llm,
                max_claims=6,
            )
            # On agrege les theses pour un passage unique (plus efficient que 1 par arg)
            aggregated_text = "\n".join(f"[{a.debater_id} r{a.round}] {a.thesis}" for a in to_audit)
            vr = verifier.verify(text=aggregated_text, sources=article_text[:6000])
            report.verification_results = vr.results
        except Exception as e:
            logger.warning("[Augmentation] verifier failure: %s", e)

    _compute_signals(report)
    return report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mock scratchpad
    @dataclass
    class MockScratchpad:
        ticker: str
        haussier_arguments: list
        baissier_arguments: list
        neutre_arguments: list

    sp = MockScratchpad(
        ticker="AAPL",
        haussier_arguments=[
            "Revenue +8%, services margin fort, guidance raised [confiance: 0.8]",
            "China pricing power offsets macro weakness [confiance: 0.75]",
        ],
        baissier_arguments=[
            "Smartphone saturation limit upside, services fee scrutiny [confiance: 0.6]",
        ],
        neutre_arguments=[
            "Fundamentals ok, valuation tendue a 30x FY25 [confiance: 0.55]",
        ],
    )

    args = scratchpad_to_arguments(sp)
    logger.info(f"[Extracted {len(args)} arguments]")
    for a in args:
        logger.info(f"  {a.debater_id} r{a.round} pos={a.position.value} conf={a.confidence}")
        logger.info(f"    thesis: {a.thesis[:60]}")
    print()

    # Mock critic LLM : renvoie severity=MEDIUM pour Haussier r=1, LOW autrement
    def critic_llm(prompt: str) -> str:
        import re as _re

        m = _re.search(r"Debatteur\s*:\s*(\w+)\s*\nRound\s*:\s*(\d+)", prompt)
        deb, rnd = (m.group(1), int(m.group(2))) if m else ("?", 0)
        sev = "medium" if (deb == "Haussier" and rnd == 1) else "low"
        return json.dumps(
            {
                "target_debater_id": deb,
                "target_round": rnd,
                "biases_detected": ["confirmation"] if sev == "medium" else [],
                "logical_gaps": [],
                "unsupported_claims": ["PE non source"] if sev == "medium" else [],
                "severity": sev,
                "suggested_revisions": ["Citer source PE"] if sev == "medium" else [],
            }
        )

    report = run_augmentation(
        scratchpad=sp,
        article_text="Apple revenue grew 8% in Q3 to $89.5B. Tim Cook is CEO.",
        critic_call_llm=critic_llm,
    )
    _compute_signals(report)

    print(
        f"[avg_severity={report.avg_severity:.2f}] "
        f"[max_severity={report.max_severity}] "
        f"[ver_ratio={report.verification_ratio:.2f}] "
        f"[conf_mult={report.confidence_multiplier:.2f}]"
    )
    print()
    print(report.to_prompt_block())
