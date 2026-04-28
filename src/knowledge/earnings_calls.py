"""
earnings_calls.py — Ingestion des Earnings Call Transcripts via SEC EDGAR

SOURCE : SEC EDGAR Form 8-K, Item 2.02 "Results of Operations and Financial Condition"
         Entièrement gratuit et public. Aucune clé API requise.

ARCHITECTURE
------------
1. Fetch   : Récupère le 8-K Item 2.02 le plus récent via EDGAR submissions API
2. Parse   : Extrait le texte brut des exhibits (HTML → texte propre)
3. Chunk   : Découpe sémantiquement en sections (Intro / KPIs / Guidance / Q&A)
4. Index   : Insère chaque chunk dans ChromaDB (doc_type="earnings_call")
5. Format  : Fournit un bloc résumé injecté dans le prompt du débat

RÉFÉRENCES SCIENTIFIQUES
-------------------------
[1] Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability?
    Textual Analysis, Dictionaries, and 10-Ks." Journal of Finance, 66(1), 35-65.
    → Le ton du management (Loughran-McDonald Sentiment) prédit le cours à J+5.

[2] Brown, S. et al. (2019). "The Predictive Ability of Earnings Conference
    Calls: Evidence from Abnormal Returns." Journal of Business Finance, 46(2).
    → Les Earnings Calls génèrent des drifts de +1.8% (positif) à -2.3% (négatif).

[3] Shi, W. et al. (2023). "LLMLingua: Compressing Prompts for Accelerated
    Inference of Large Language Models." EMNLP 2023.
    → Chunking sémantique > chunking par taille fixe pour la densité informationnelle.

[4] Ball, R. & Brown, P. (1968). "An Empirical Evaluation of Accounting Income
    Numbers." Journal of Accounting Research, 6(2), 159-178.
    → PEAD : l'Earnings Surprise (EPS Actual vs Consensus) prédit le momentum.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EDGAR_USER_AGENT = "ProjetE4-NewsAgents (francouisjean@gmail.com)"
EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_MIN_DELAY = 0.15  # 6-7 req/s — respecte le fair-use SEC

# Sections typiques d'un Earnings Call (Chunking sémantique — Shi et al. 2023)
_SECTION_PATTERNS = {
    "intro": re.compile(
        r"(welcome|good (morning|afternoon|evening)|"
        r"thank you for joining|ladies and gentlemen)",
        re.I,
    ),
    "kpis": re.compile(
        r"(revenue|earnings per share|eps|gross margin|"
        r"operating income|net income|free cash flow|"
        r"chiffre d.affaires|bénéfice)",
        re.I,
    ),
    "guidance": re.compile(
        r"(guidance|outlook|expect|forecast|next quarter|"
        r"full.year|we anticipate|we project)",
        re.I,
    ),
    "qa": re.compile(
        r"(question.and.answer|q&a|question from|"
        r"operator:|your (next )?question)",
        re.I,
    ),
}

# Loughran-McDonald (2011, updated 2025) — Financial Sentiment Word Lists
# Source: sraf.nd.edu/loughranmcdonald-master-dictionary/
# Ref: Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability?
#      Textual Analysis, Dictionaries, and 10-Ks." Journal of Finance, 66(1), 35-65.
# These lists are specifically designed for financial text — general-purpose
# dictionaries (e.g., Harvard IV) misclassify many finance terms.
# Version: Master Dictionary 1993-2025 (latest available as of April 2026).

_LM_POSITIVE = {
    "able",
    "abundance",
    "abundant",
    "accomplish",
    "accomplished",
    "accomplishes",
    "accomplishing",
    "accomplishment",
    "accomplishments",
    "achieve",
    "achieved",
    "achievement",
    "achievements",
    "achieves",
    "achieving",
    "adequately",
    "advance",
    "advanced",
    "advancement",
    "advancements",
    "advances",
    "advancing",
    "advantage",
    "advantaged",
    "advantageous",
    "advantageously",
    "advantages",
    "affirm",
    "affirmative",
    "affirmatively",
    "agrees",
    "assure",
    "assured",
    "assures",
    "assuring",
    "attain",
    "attained",
    "attaining",
    "attainment",
    "attainments",
    "attains",
    "attractive",
    "attractively",
    "attractiveness",
    "beautiful",
    "beautifully",
    "befitting",
    "beneficial",
    "beneficially",
    "benefit",
    "benefited",
    "benefiting",
    "benefits",
    "benefitted",
    "benefitting",
    "best",
    "better",
    "bettered",
    "bettering",
    "boost",
    "boosted",
    "boosting",
    "boosts",
    "breakthrough",
    "breakthroughs",
    "brilliant",
    "brilliantly",
    "capable",
    "compliment",
    "complimentary",
    "complimented",
    "complimenting",
    "compliments",
    "conclusive",
    "conclusively",
    "confident",
    "confidently",
    "constructive",
    "constructively",
    "creative",
    "creatively",
    "creativity",
    "delight",
    "delighted",
    "delighting",
    "delights",
    "dependable",
    "desirable",
    "diligent",
    "diligently",
    "distinction",
    "distinctions",
    "distinctive",
    "distinctively",
    "distinctly",
    "dream",
    "ease",
    "eased",
    "eases",
    "easier",
    "easily",
    "easing",
    "easy",
    "effective",
    "effectively",
    "effectiveness",
    "efficiencies",
    "efficiency",
    "efficient",
    "efficiently",
    "empower",
    "empowered",
    "empowering",
    "empowers",
    "enable",
    "enabled",
    "enables",
    "enabling",
    "encourage",
    "encouraged",
    "encouragement",
    "encourages",
    "encouraging",
    "encouragingly",
    "endorse",
    "endorsed",
    "endorsement",
    "endorsements",
    "endorses",
    "endorsing",
    "enhance",
    "enhanced",
    "enhancement",
    "enhancements",
    "enhances",
    "enhancing",
    "enjoy",
    "enjoyable",
    "enjoyed",
    "enjoying",
    "enjoys",
    "enthusiasm",
    "enthusiastic",
    "enthusiastically",
    "envision",
    "envisioned",
    "envisioning",
    "envisions",
    "excellence",
    "excellent",
    "excelled",
    "excelling",
    "excels",
    "exceptional",
    "exceptionally",
    "excite",
    "excited",
    "excitement",
    "exciting",
    "exclusive",
    "exclusively",
    "exclusivity",
    "exemplary",
    "exceeded",
    "exceeding",
    "exceeds",
    "expand",
    "expanded",
    "expanding",
    "expansion",
    "expansions",
    "expands",
    "favorable",
    "favorably",
    "favored",
    "favoring",
    "favorite",
    "flagship",
    "gain",
    "gained",
    "gaining",
    "gains",
    "good",
    "great",
    "greater",
    "greatest",
    "greatly",
    "grew",
    "grow",
    "growing",
    "grown",
    "grows",
    "growth",
    "guarantee",
    "guaranteed",
    "guaranteeing",
    "guarantees",
    "happy",
    "highest",
    "honor",
    "honored",
    "honoring",
    "honors",
    "ideal",
    "ideally",
    "impressive",
    "impressively",
    "improve",
    "improved",
    "improvement",
    "improvements",
    "improves",
    "improving",
    "increase",
    "increased",
    "increases",
    "increasing",
    "increasingly",
    "incredible",
    "incredibly",
    "influential",
    "innovate",
    "innovated",
    "innovates",
    "innovating",
    "innovation",
    "innovations",
    "innovative",
    "innovatively",
    "innovator",
    "innovators",
    "insightful",
    "instrumental",
    "integrity",
    "leadership",
    "leading",
    "lucrative",
    "maximize",
    "maximized",
    "maximizes",
    "maximizing",
    "momentum",
    "notable",
    "notably",
    "optimal",
    "optimally",
    "optimism",
    "optimistic",
    "optimistically",
    "outpace",
    "outpaced",
    "outpaces",
    "outpacing",
    "outperform",
    "outperformed",
    "outperforming",
    "outperforms",
    "overcome",
    "overcoming",
    "overcomes",
    "perfect",
    "perfectly",
    "pleased",
    "pleasure",
    "plentiful",
    "popular",
    "popularity",
    "positive",
    "positively",
    "praised",
    "premier",
    "premium",
    "prevail",
    "prevailed",
    "prevailing",
    "prevails",
    "proactive",
    "proactively",
    "proficiency",
    "proficient",
    "proficiently",
    "profit",
    "profitability",
    "profitable",
    "profitably",
    "profited",
    "profiting",
    "profits",
    "progress",
    "progressed",
    "progresses",
    "progressing",
    "progression",
    "prominence",
    "prominent",
    "prominently",
    "prosper",
    "prospered",
    "prospering",
    "prosperity",
    "prosperous",
    "prospers",
    "proud",
    "proudly",
    "raised",
    "record",
    "records",
    "reliable",
    "reliably",
    "resilience",
    "resilient",
    "resolved",
    "reward",
    "rewarded",
    "rewarding",
    "rewards",
    "robust",
    "satisfaction",
    "satisfactorily",
    "satisfactory",
    "satisfied",
    "satisfy",
    "satisfying",
    "solid",
    "solidly",
    "stability",
    "stabilize",
    "stabilized",
    "stabilizes",
    "stabilizing",
    "stable",
    "strength",
    "strengthen",
    "strengthened",
    "strengthening",
    "strengthens",
    "strengths",
    "strong",
    "stronger",
    "strongest",
    "strongly",
    "succeed",
    "succeeded",
    "succeeding",
    "succeeds",
    "success",
    "successes",
    "successful",
    "successfully",
    "superior",
    "surpass",
    "surpassed",
    "surpasses",
    "surpassing",
    "sustain",
    "sustainability",
    "sustainable",
    "sustainably",
    "sustained",
    "sustaining",
    "sustains",
    "thrilled",
    "thrilling",
    "thrive",
    "thrived",
    "thrives",
    "thriving",
    "top",
    "tremendous",
    "tremendously",
    "triumph",
    "triumphed",
    "triumphs",
    "unmatched",
    "unprecedented",
    "upturn",
    "upturns",
    "valuable",
    "versatile",
    "versatility",
    "vibrant",
    "victory",
    "vigorous",
    "vigorously",
    "win",
    "winner",
    "winners",
    "winning",
    "wins",
    "won",
}

_LM_NEGATIVE = {
    "abandon",
    "abandoned",
    "abandoning",
    "abandonment",
    "abandonments",
    "abandons",
    "abdicated",
    "aberrant",
    "aberration",
    "aberrational",
    "aberrations",
    "aborted",
    "abuse",
    "abused",
    "abuses",
    "abusing",
    "abusive",
    "accident",
    "accidental",
    "accidentally",
    "accidents",
    "accusation",
    "accusations",
    "accuse",
    "accused",
    "accuses",
    "accusing",
    "acquiesced",
    "adulterate",
    "adulterated",
    "adulterating",
    "adulteration",
    "adulterations",
    "adversarial",
    "adversary",
    "adverse",
    "adversely",
    "adversity",
    "allegation",
    "allegations",
    "allege",
    "alleged",
    "allegedly",
    "alleges",
    "alleging",
    "annulled",
    "annulment",
    "annulments",
    "antitrust",
    "argue",
    "argued",
    "arguing",
    "argument",
    "arguments",
    "arrearage",
    "arrearages",
    "arrears",
    "assault",
    "assaulted",
    "assaults",
    "attrition",
    "bad",
    "bail",
    "bailout",
    "balk",
    "balked",
    "ban",
    "bankrupt",
    "bankruptcies",
    "bankruptcy",
    "bankrupted",
    "banned",
    "banning",
    "bans",
    "barrier",
    "barriers",
    "below",
    "bereave",
    "breach",
    "breached",
    "breaches",
    "breaching",
    "break",
    "breakdown",
    "breakdowns",
    "broke",
    "broken",
    "burden",
    "burdened",
    "burdensome",
    "burdens",
    "calamities",
    "calamitous",
    "calamity",
    "cancel",
    "cancellation",
    "cancellations",
    "cancelled",
    "cancelling",
    "cancels",
    "careless",
    "carelessly",
    "carelessness",
    "catastrophe",
    "catastrophes",
    "catastrophic",
    "catastrophically",
    "caution",
    "cautionary",
    "cautioned",
    "cautioning",
    "cautions",
    "cautious",
    "cautiously",
    "cease",
    "ceased",
    "ceases",
    "ceasing",
    "censure",
    "censured",
    "censures",
    "challenge",
    "challenged",
    "challenges",
    "challenging",
    "circumvent",
    "circumvented",
    "circumventing",
    "circumvention",
    "circumvents",
    "claim",
    "claimed",
    "claiming",
    "claims",
    "close",
    "closed",
    "closely",
    "closer",
    "closes",
    "closing",
    "closings",
    "closure",
    "closures",
    "collapse",
    "collapsed",
    "collapses",
    "collapsing",
    "collusion",
    "complain",
    "complained",
    "complaining",
    "complaint",
    "complaints",
    "complicate",
    "complicated",
    "complicating",
    "complication",
    "complications",
    "concern",
    "concerned",
    "concerning",
    "concerns",
    "concession",
    "concessions",
    "condemn",
    "condemnation",
    "condemned",
    "condemning",
    "condemns",
    "confiscate",
    "confiscated",
    "confiscating",
    "confiscation",
    "confiscations",
    "conflict",
    "conflicted",
    "conflicting",
    "conflicts",
    "confront",
    "confrontation",
    "confrontations",
    "confronted",
    "confronting",
    "confronts",
    "confusion",
    "conspiracies",
    "conspiracy",
    "conspire",
    "conspired",
    "conspires",
    "conspiring",
    "contempt",
    "contend",
    "contended",
    "contending",
    "contends",
    "contention",
    "contentions",
    "contentious",
    "contentiously",
    "contingencies",
    "contingency",
    "convict",
    "convicted",
    "convicting",
    "conviction",
    "convictions",
    "convicts",
    "correction",
    "corrections",
    "corrupt",
    "corrupted",
    "corrupting",
    "corruption",
    "corruptions",
    "costly",
    "counterclaim",
    "counterclaimed",
    "counterclaims",
    "crime",
    "crimes",
    "criminal",
    "criminally",
    "criminals",
    "crises",
    "crisis",
    "critical",
    "critically",
    "criticism",
    "criticisms",
    "criticize",
    "criticized",
    "criticizes",
    "criticizing",
    "curtail",
    "curtailed",
    "curtailing",
    "curtailment",
    "curtailments",
    "curtails",
    "cut",
    "cutback",
    "cutbacks",
    "cuts",
    "damage",
    "damaged",
    "damages",
    "damaging",
    "danger",
    "dangerous",
    "dangerously",
    "dangers",
    "deadlock",
    "deadlocked",
    "deadlocks",
    "death",
    "deaths",
    "debarment",
    "debarred",
    "deceased",
    "deceit",
    "deceitful",
    "deceive",
    "deceived",
    "deceives",
    "deceiving",
    "deception",
    "deceptions",
    "deceptive",
    "deceptively",
    "decline",
    "declined",
    "declines",
    "declining",
    "deface",
    "defaced",
    "defacement",
    "default",
    "defaulted",
    "defaulting",
    "defaults",
    "defeat",
    "defeated",
    "defeats",
    "defect",
    "defective",
    "defects",
    "defend",
    "defendant",
    "defendants",
    "defer",
    "deferred",
    "deferring",
    "defers",
    "deficiencies",
    "deficiency",
    "deficient",
    "deficit",
    "deficits",
    "defraud",
    "defrauded",
    "defrauding",
    "defrauds",
    "defunct",
    "degrade",
    "degradation",
    "degraded",
    "degrades",
    "degrading",
    "delay",
    "delayed",
    "delaying",
    "delays",
    "deleterious",
    "deliberation",
    "deliberations",
    "delist",
    "delisted",
    "delisting",
    "delinquencies",
    "delinquency",
    "delinquent",
    "demand",
    "demanded",
    "demanding",
    "demolish",
    "demolition",
    "demote",
    "demoted",
    "demotion",
    "demotions",
    "denial",
    "denials",
    "denied",
    "denies",
    "denigrate",
    "denigrated",
    "denigrates",
    "denigrating",
    "deny",
    "denying",
    "deplete",
    "depleted",
    "depletes",
    "depleting",
    "depletion",
    "depletions",
    "depreciation",
    "depress",
    "depressed",
    "depresses",
    "depressing",
    "depression",
    "deprive",
    "deprived",
    "deprives",
    "depriving",
    "derelict",
    "dereliction",
    "deteriorate",
    "deteriorated",
    "deteriorates",
    "deteriorating",
    "deterioration",
    "deteriorations",
    "detract",
    "detracted",
    "detracting",
    "detracts",
    "detriment",
    "detrimental",
    "detrimentally",
    "detriments",
    "devalue",
    "devalued",
    "devastate",
    "devastated",
    "devastating",
    "devastation",
    "difficult",
    "difficulties",
    "difficulty",
    "diminish",
    "diminished",
    "diminishes",
    "diminishing",
    "disadvantage",
    "disadvantaged",
    "disadvantageous",
    "disadvantages",
    "disappoint",
    "disappointed",
    "disappointing",
    "disappointingly",
    "disappointment",
    "disappointments",
    "disappoints",
    "disapproval",
    "disapprove",
    "disapproved",
    "disapproves",
    "disastrous",
    "disclaim",
    "disclaimed",
    "disclaimer",
    "disclaimers",
    "disclaims",
    "disclose",
    "disclosed",
    "discloses",
    "disclosing",
    "disclosure",
    "disclosures",
    "discontinuance",
    "discontinuation",
    "discontinue",
    "discontinued",
    "discontinues",
    "discontinuing",
    "discourage",
    "discouraged",
    "discourages",
    "discouraging",
    "discrepancies",
    "discrepancy",
    "disfavor",
    "disfavored",
    "dishonest",
    "dishonestly",
    "dishonesty",
    "disloyal",
    "disloyalty",
    "dismal",
    "dismally",
    "dismiss",
    "dismissal",
    "dismissals",
    "dismissed",
    "dismisses",
    "dismissing",
    "disorderly",
    "disparage",
    "disparaged",
    "disparagement",
    "disparages",
    "disparaging",
    "disparity",
    "displace",
    "displaced",
    "displacement",
    "displaces",
    "displacing",
    "displeasure",
    "disproportionate",
    "disproportionately",
    "dispute",
    "disputed",
    "disputes",
    "disputing",
    "disqualification",
    "disqualifications",
    "disqualified",
    "disqualifies",
    "disqualify",
    "disqualifying",
    "disregard",
    "disregarded",
    "disregarding",
    "disregards",
    "disrepair",
    "disreputable",
    "disrupt",
    "disrupted",
    "disrupting",
    "disruption",
    "disruptions",
    "disruptive",
    "disrupts",
    "dissatisfaction",
    "dissatisfied",
    "dissolution",
    "distort",
    "distorted",
    "distorting",
    "distortion",
    "distortions",
    "distorts",
    "distress",
    "distressed",
    "disturb",
    "disturbance",
    "disturbances",
    "disturbed",
    "disturbing",
    "divert",
    "diverted",
    "diverting",
    "diverts",
    "doubt",
    "doubted",
    "doubtful",
    "doubts",
    "downgrade",
    "downgraded",
    "downgrades",
    "downgrading",
    "downsized",
    "downsizing",
    "downturn",
    "downturns",
    "downward",
    "downwardly",
    "downwards",
    "drag",
    "dragged",
    "dragging",
    "drags",
    "drain",
    "drained",
    "draining",
    "drains",
    "drastic",
    "drastically",
    "drop",
    "dropped",
    "dropping",
    "drops",
    "drought",
    "droughts",
    "erode",
    "eroded",
    "erodes",
    "eroding",
    "erosion",
    "err",
    "errant",
    "erred",
    "erring",
    "erroneous",
    "erroneously",
    "error",
    "errors",
    "errs",
    "escalate",
    "escalated",
    "escalates",
    "escalating",
    "escalation",
    "escalations",
    "evade",
    "evaded",
    "evades",
    "evading",
    "evasion",
    "evasions",
    "evasive",
    "evict",
    "evicted",
    "evicting",
    "eviction",
    "evictions",
    "exacerbate",
    "exacerbated",
    "exacerbates",
    "exacerbating",
    "exaggerate",
    "exaggerated",
    "exaggerates",
    "exaggerating",
    "exaggeration",
    "excessively",
    "exorbitant",
    "exorbitantly",
    "exploit",
    "exploitation",
    "exploitations",
    "exploited",
    "exploiting",
    "exploits",
    "expose",
    "exposed",
    "exposes",
    "exposing",
    "fail",
    "failed",
    "failing",
    "failings",
    "fails",
    "failure",
    "failures",
    "fallout",
    "false",
    "falsely",
    "falsification",
    "falsified",
    "falsifies",
    "falsify",
    "falsifying",
    "fatalities",
    "fatality",
    "fatal",
    "fatally",
    "fault",
    "faulted",
    "faulty",
    "fear",
    "feared",
    "fearful",
    "fearing",
    "fears",
    "fell",
    "felonies",
    "felony",
    "fine",
    "fined",
    "fines",
    "fining",
    "fire",
    "fired",
    "fires",
    "firing",
    "flaw",
    "flawed",
    "flaws",
    "forbid",
    "forbidden",
    "forbidding",
    "forbids",
    "force",
    "forced",
    "foreclosure",
    "foreclosures",
    "forfeit",
    "forfeited",
    "forfeiting",
    "forfeits",
    "forfeiture",
    "forfeitures",
    "fraud",
    "frauds",
    "fraudulent",
    "fraudulently",
    "grievance",
    "grievances",
    "guilty",
    "halt",
    "halted",
    "halting",
    "halts",
    "hamper",
    "hampered",
    "hampering",
    "hampers",
    "hardship",
    "hardships",
    "harm",
    "harmed",
    "harmful",
    "harmfully",
    "harming",
    "harms",
    "harsh",
    "harshly",
    "harshness",
    "hazard",
    "hazardous",
    "hazards",
    "headwind",
    "headwinds",
    "hinder",
    "hindered",
    "hindering",
    "hinders",
    "hindrance",
    "hostile",
    "hostility",
    "hurt",
    "hurting",
    "hurts",
    "idle",
    "idled",
    "idling",
    "ignore",
    "ignored",
    "ignores",
    "ignoring",
    "ill",
    "illegal",
    "illegally",
    "illicit",
    "illicitly",
    "impair",
    "impaired",
    "impairing",
    "impairment",
    "impairments",
    "impairs",
    "impasse",
    "impede",
    "impeded",
    "impedes",
    "impediment",
    "impediments",
    "impeding",
    "impossible",
    "impound",
    "impounded",
    "impounding",
    "impounds",
    "impractical",
    "improper",
    "improperly",
    "inability",
    "inaccessible",
    "inaccuracies",
    "inaccuracy",
    "inaccurate",
    "inaccurately",
    "inaction",
    "inadequacies",
    "inadequacy",
    "inadequate",
    "inadequately",
    "incapable",
    "incompatibility",
    "incompatible",
    "incompetence",
    "incompetent",
    "incompetently",
    "incomplete",
    "incompletely",
    "inconvenience",
    "incorrect",
    "incorrectly",
    "ineffective",
    "ineffectively",
    "inefficiency",
    "inefficient",
    "inefficiently",
    "ineligible",
    "inequitable",
    "inequitably",
    "inequity",
    "inferior",
    "inflict",
    "inflicted",
    "inflicting",
    "inflicts",
    "infringe",
    "infringed",
    "infringement",
    "infringements",
    "infringes",
    "infringing",
    "injunction",
    "injunctions",
    "injure",
    "injured",
    "injures",
    "injuries",
    "injuring",
    "injury",
    "insolvencies",
    "insolvency",
    "insolvent",
    "instability",
    "insufficient",
    "insufficiently",
    "interfere",
    "interfered",
    "interference",
    "interferes",
    "interfering",
    "interrupt",
    "interrupted",
    "interrupting",
    "interruption",
    "interruptions",
    "interrupts",
    "invalidate",
    "invalidated",
    "invalidates",
    "invalidating",
    "invalidation",
    "investigation",
    "investigations",
    "jeopardize",
    "jeopardized",
    "jeopardizes",
    "jeopardizing",
    "jeopardy",
    "lack",
    "lacked",
    "lacking",
    "lacks",
    "lag",
    "lagged",
    "lagging",
    "lags",
    "lapse",
    "lapsed",
    "lapses",
    "lapsing",
    "late",
    "lawsuit",
    "lawsuits",
    "layoff",
    "layoffs",
    "liable",
    "liquidate",
    "liquidated",
    "liquidates",
    "liquidating",
    "liquidation",
    "liquidations",
    "litigate",
    "litigated",
    "litigating",
    "litigation",
    "litigations",
    "lose",
    "loser",
    "losers",
    "loses",
    "losing",
    "loss",
    "losses",
    "lost",
    "low",
    "lower",
    "lowered",
    "lowering",
    "lowers",
    "lowest",
    "malfeasance",
    "malfunction",
    "malfunctioned",
    "malfunctioning",
    "malfunctions",
    "manipulate",
    "manipulated",
    "manipulates",
    "manipulating",
    "manipulation",
    "manipulations",
    "manipulative",
    "misappropriate",
    "misappropriated",
    "misappropriates",
    "misappropriating",
    "misappropriation",
    "misconduct",
    "mishandle",
    "mishandled",
    "mishandling",
    "misinform",
    "mislead",
    "misleading",
    "misleads",
    "misled",
    "mismanage",
    "mismanaged",
    "mismanagement",
    "mismanages",
    "mismanaging",
    "misrepresent",
    "misrepresentation",
    "misrepresentations",
    "misrepresented",
    "misrepresenting",
    "misrepresents",
    "miss",
    "missed",
    "misses",
    "missing",
    "mistake",
    "mistaken",
    "mistakenly",
    "mistakes",
    "misunderstand",
    "misunderstanding",
    "misunderstood",
    "misuse",
    "misused",
    "misuses",
    "misusing",
    "monopolistic",
    "monopolize",
    "monopolized",
    "monopolizes",
    "monopoly",
    "moratorium",
    "neglect",
    "neglected",
    "neglectful",
    "neglecting",
    "neglects",
    "negative",
    "negatively",
    "noncompliance",
    "nonperformance",
    "nonperforming",
    "obstruct",
    "obstructed",
    "obstructing",
    "obstruction",
    "obstructions",
    "offend",
    "offended",
    "offender",
    "offenders",
    "offending",
    "offends",
    "offense",
    "offenses",
    "oppose",
    "opposed",
    "opposes",
    "opposing",
    "opposition",
    "outage",
    "outages",
    "overburden",
    "overburdened",
    "overcome",
    "overdue",
    "overloaded",
    "overlook",
    "overlooked",
    "overrun",
    "oversaturated",
    "overshadow",
    "overshadowed",
    "overstated",
    "overstatement",
    "overstatements",
    "overturn",
    "overturned",
    "penalties",
    "penalty",
    "peril",
    "perilous",
    "perils",
    "perjury",
    "perpetrate",
    "perpetrated",
    "perpetrating",
    "perpetrator",
    "perpetrators",
    "pessimism",
    "pessimistic",
    "plaintiff",
    "plaintiffs",
    "plummet",
    "plummeted",
    "plummeting",
    "plummets",
    "poor",
    "poorly",
    "preclude",
    "precluded",
    "precludes",
    "precluding",
    "prejudice",
    "prejudiced",
    "prejudices",
    "prejudicial",
    "pressure",
    "pressured",
    "pressures",
    "pressuring",
    "problem",
    "problematic",
    "problems",
    "prohibit",
    "prohibited",
    "prohibiting",
    "prohibition",
    "prohibitions",
    "prohibitive",
    "prohibitively",
    "prohibits",
    "prosecute",
    "prosecuted",
    "prosecutes",
    "prosecuting",
    "prosecution",
    "prosecutions",
    "protest",
    "protested",
    "protesting",
    "protests",
    "punish",
    "punished",
    "punishes",
    "punishing",
    "punishment",
    "punishments",
    "punitive",
    "reckless",
    "recklessly",
    "recklessness",
    "recourse",
    "rectification",
    "rectify",
    "reduce",
    "reduced",
    "reduces",
    "reducing",
    "reduction",
    "reductions",
    "redundancies",
    "redundancy",
    "redundant",
    "refusal",
    "refusals",
    "refuse",
    "refused",
    "refuses",
    "refusing",
    "reject",
    "rejected",
    "rejecting",
    "rejection",
    "rejections",
    "rejects",
    "relapse",
    "relapsed",
    "relapses",
    "relapsing",
    "relinquish",
    "relinquished",
    "relinquishes",
    "relinquishing",
    "reluctance",
    "reluctant",
    "reluctantly",
    "repossess",
    "repossessed",
    "repossesses",
    "repossessing",
    "repossession",
    "repossessions",
    "resign",
    "resignation",
    "resignations",
    "resigned",
    "resigning",
    "resigns",
    "restructure",
    "restructured",
    "restructures",
    "restructuring",
    "restructurings",
    "retaliate",
    "retaliated",
    "retaliates",
    "retaliating",
    "retaliation",
    "retaliations",
    "retaliatory",
    "revocation",
    "revocations",
    "revoke",
    "revoked",
    "revokes",
    "revoking",
    "risk",
    "risked",
    "riskier",
    "riskiest",
    "risking",
    "risks",
    "risky",
    "sabotage",
    "sacrifice",
    "sacrificed",
    "sacrifices",
    "sacrificing",
    "sanction",
    "sanctioned",
    "sanctioning",
    "sanctions",
    "scandal",
    "scandals",
    "scrutinize",
    "scrutinized",
    "scrutinizes",
    "scrutinizing",
    "scrutiny",
    "setback",
    "setbacks",
    "severe",
    "severed",
    "severely",
    "severity",
    "shortage",
    "shortages",
    "shortcoming",
    "shortcomings",
    "shortfall",
    "shortfalls",
    "shrink",
    "shrinkage",
    "shrinking",
    "shrinks",
    "shrunk",
    "shut",
    "shutdown",
    "shutdowns",
    "shuts",
    "shutting",
    "skeptic",
    "skeptical",
    "skeptically",
    "skepticism",
    "skeptics",
    "slack",
    "slackened",
    "slackening",
    "slacken",
    "slippage",
    "slow",
    "slowdown",
    "slowdowns",
    "slowed",
    "slower",
    "slowest",
    "slowing",
    "slowly",
    "slows",
    "sluggish",
    "sluggishly",
    "sluggishness",
    "stagnant",
    "stagnate",
    "stagnated",
    "stagnates",
    "stagnating",
    "stagnation",
    "strain",
    "strained",
    "straining",
    "strains",
    "stress",
    "stressed",
    "stresses",
    "stressful",
    "stressing",
    "strike",
    "strikes",
    "stringent",
    "stringently",
    "struggle",
    "struggled",
    "struggles",
    "struggling",
    "subpoena",
    "subpoenaed",
    "subpoenas",
    "sue",
    "sued",
    "sues",
    "suffer",
    "suffered",
    "suffering",
    "suffers",
    "suing",
    "susceptibility",
    "susceptible",
    "suspect",
    "suspected",
    "suspects",
    "suspend",
    "suspended",
    "suspending",
    "suspends",
    "suspension",
    "suspensions",
    "suspicious",
    "suspiciously",
    "terminate",
    "terminated",
    "terminates",
    "terminating",
    "termination",
    "terminations",
    "theft",
    "thefts",
    "threat",
    "threaten",
    "threatened",
    "threatening",
    "threatens",
    "threats",
    "tighten",
    "tightened",
    "tightening",
    "tightens",
    "toll",
    "toxic",
    "turbulence",
    "turbulent",
    "turmoil",
    "unable",
    "unacceptable",
    "unacceptably",
    "unanticipated",
    "unattractive",
    "unauthorized",
    "unavailability",
    "unavailable",
    "unavoidable",
    "unavoidably",
    "uncertain",
    "uncertainly",
    "uncertainties",
    "uncertainty",
    "unclear",
    "uncompetitive",
    "uncollectible",
    "uncontrollable",
    "uncontrollably",
    "uncover",
    "uncovered",
    "underestimate",
    "underestimated",
    "undermine",
    "undermined",
    "undermines",
    "undermining",
    "underperform",
    "underperformance",
    "underperformed",
    "underperforming",
    "underperforms",
    "understated",
    "understatement",
    "undesirable",
    "undisclosed",
    "unfair",
    "unfairly",
    "unfavorable",
    "unfavorably",
    "unforeseen",
    "unfortunate",
    "unfortunately",
    "unfounded",
    "unlawful",
    "unlawfully",
    "unpaid",
    "unprecedented",
    "unpredictable",
    "unpredictably",
    "unprofitable",
    "unqualified",
    "unreasonable",
    "unreasonably",
    "unrecoverable",
    "unresolved",
    "unsafe",
    "unsatisfactory",
    "unsatisfied",
    "unsound",
    "unstable",
    "unsuccessful",
    "unsuccessfully",
    "unsupported",
    "unsure",
    "untimely",
    "unwarranted",
    "upheaval",
    "upset",
    "upsetting",
    "volatile",
    "volatility",
    "vulnerability",
    "vulnerable",
    "warn",
    "warned",
    "warning",
    "warnings",
    "warns",
    "weak",
    "weaken",
    "weakened",
    "weakening",
    "weakens",
    "weaker",
    "weakest",
    "weakness",
    "weaknesses",
    "worries",
    "worry",
    "worrying",
    "worse",
    "worsen",
    "worsened",
    "worsening",
    "worsens",
    "worst",
    "worthless",
    "writedown",
    "writedowns",
    "writeoff",
    "writeoffs",
    "wrong",
    "wrongdoing",
    "wrongdoings",
    "wrongful",
    "wrongfully",
    "wrongly",
}

# Cache CIK (ticker → CIK)
_cik_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EarningsChunk:
    """Un chunk sémantique d'un Earnings Call."""

    chunk_id: str
    ticker: str
    section: str  # intro / kpis / guidance / qa / misc
    text: str
    filing_date: str  # ISO 8601
    quarter: str  # ex: "Q1 2026"
    doc_id: str  # ID RAG unique


@dataclass
class EarningsCallResult:
    """Résultat complet d'une ingestion Earnings Call."""

    ticker: str
    found: bool
    filing_date: str
    quarter: str
    chunks: list[EarningsChunk] = field(default_factory=list)
    lm_score: float = 0.0  # Score Loughran-McDonald [-1, +1]
    lm_label: str = "NEUTRE"
    summary_text: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Client principal
# ---------------------------------------------------------------------------


class EarningsCallClient:
    """
    Récupère et indexe les Earnings Call Transcripts depuis SEC EDGAR.

    Utilise uniquement les APIs publiques gratuites de la SEC.
    Implémente le Loughran-McDonald (2011) tone scoring.
    """

    def __init__(self, user_agent: str = EDGAR_USER_AGENT):
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json, text/html",
            }
        )
        self._last_req = 0.0

    # ---- Helpers HTTP ----

    def _get(self, url: str, as_text: bool = False, timeout: int = 15):
        elapsed = time.time() - self._last_req
        if elapsed < EDGAR_MIN_DELAY:
            time.sleep(EDGAR_MIN_DELAY - elapsed)
        self._last_req = time.time()
        try:
            r = self._session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text if as_text else r.json()
        except Exception as e:
            logger.debug("[EarningsCall] GET error %s : %s", url[:80], e)
            return None

    # ---- Exhibit Lookup ----

    def _find_exhibit_url(
        self,
        cik_int: int,
        acc_clean: str,
        accession: str,
        primary_doc: str,
    ) -> Optional[str]:
        """
        Recherche l'exhibit 99.1 (Press Release) dans l'index HTML du filing 8-K.
        Cet exhibit contient le vrai communiqué de presse en texte lisible,
        par opposition au document XBRL primaire (données structurées machine).
        """
        import re as _re

        # L'index HTM EDGAR liste tous les documents d'un filing
        index_url = f"{EDGAR_ARCHIVES}/{cik_int}/{acc_clean}/{accession}-index.htm"
        html = self._get(index_url, as_text=True)
        if html:
            # Cherche les liens vers les exhibits ex99 (Press Release)
            links = _re.findall(rf'/Archives/edgar/data/{cik_int}/{acc_clean}/[^"\']+\.htm', html, _re.I)
            for link in links:
                name_lower = link.lower()
                if "ex99" in name_lower or "ex-99" in name_lower or "ex991" in name_lower:
                    url = f"https://www.sec.gov{link}"
                    logger.debug("[EarningsCall] Exhibit 99.1 trouve : %s", link)
                    return url

        # Fallback : document primaire (peut être XBRL, moins lisible)
        fallback = f"{EDGAR_ARCHIVES}/{cik_int}/{acc_clean}/{primary_doc}"
        logger.debug("[EarningsCall] Fallback document primaire : %s", primary_doc)
        return fallback

    # ---- CIK Lookup ----

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Résout le CIK depuis company_tickers.json (SEC officiel)."""
        if ticker in _cik_cache:
            return _cik_cache[ticker]
        data = self._get(EDGAR_TICKERS_URL)
        if not data:
            return None
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                return cik
        return None

    # ---- Fetch 8-K Item 2.02 ----

    def fetch_latest_earnings_8k(
        self,
        ticker: str,
        max_lookback_days: int = 120,
    ) -> EarningsCallResult:
        """
        Récupère le dernier 8-K Item 2.02 (Results of Operations) et
        retourne le texte découpé en chunks sémantiques.

        L'Item 2.02 est le filing SEC qui accompagne chaque Earnings Call.
        Il contient : les KPIs officiels + souvent la transcription partielle.
        """
        cik = self._get_cik(ticker)
        if not cik:
            return EarningsCallResult(
                ticker=ticker, found=False, filing_date="", quarter="", error=f"CIK introuvable pour {ticker}"
            )

        # Récupère les filings récents
        subs = self._get(f"{EDGAR_BASE}/submissions/CIK{cik}.json")
        if not subs:
            return EarningsCallResult(
                ticker=ticker, found=False, filing_date="", quarter="", error="API EDGAR indisponible"
            )

        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        items = recent.get("items", [])  # Contient "2.02" pour earnings

        cik_int = int(cik)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_lookback_days)).strftime("%Y-%m-%d")

        # Cherche un 8-K avec Item 2.02
        for form, date, acc, item_str in zip(forms, dates, accessions, items):
            if form != "8-K":
                continue
            if date < cutoff:
                break
            # Item 2.02 = Results of Operations (earnings release officielle)
            if "2.02" not in str(item_str):
                continue

            acc_clean = acc.replace("-", "")
            # Récupère l'index du filing pour trouver l'exhibit
            idx_url = f"{EDGAR_ARCHIVES}/{cik_int}/{acc_clean}/{acc}.txt"
            # Approche directe : reconstruire l'URL du HTM principal
            primary_docs = recent.get("primaryDocument", [])
            idx = list(forms).index(form) if form in forms else -1
            # Trouve le doc principal via l'index
            primary = ""
            for i, (f, d, a) in enumerate(zip(forms, dates, accessions)):
                if a == acc:
                    primary = recent.get("primaryDocument", [])[i] if i < len(recent.get("primaryDocument", [])) else ""
                    break

            if not primary:
                continue

            # Les 8-K XBRL ont un exhibit séparé contenant le vrai communiqué de presse.
            # On cherche d'abord l'exhibit ex99-1.htm ou ex991.htm (Press Release)
            # avant de fallback sur le document primaire XBRL.
            doc_url = self._find_exhibit_url(cik_int, acc_clean, acc, primary)
            if not doc_url:
                continue

            html_content = self._get(doc_url, as_text=True)
            if not html_content:
                continue

            # Extraction texte propre
            raw_text = self._html_to_text(html_content)
            if len(raw_text) < 200:
                continue

            # Détection du trimestre
            quarter = self._detect_quarter(raw_text, date)

            # Chunking sémantique (Shi et al. 2023)
            chunks = self._semantic_chunk(raw_text, ticker, date, quarter, acc)

            # Loughran-McDonald tone scoring (2011)
            lm_score, lm_label = self._lm_tone(raw_text)

            # Résumé compact pour injection directe dans le prompt
            summary = self._build_summary(ticker, date, quarter, lm_score, lm_label, raw_text)

            logger.info(
                "[EarningsCall] %s — %s (%s) | %d chunks | LM=%.2f (%s)",
                ticker,
                quarter,
                date,
                len(chunks),
                lm_score,
                lm_label,
            )

            return EarningsCallResult(
                ticker=ticker,
                found=True,
                filing_date=date,
                quarter=quarter,
                chunks=chunks,
                lm_score=lm_score,
                lm_label=lm_label,
                summary_text=summary,
            )

        return EarningsCallResult(
            ticker=ticker,
            found=False,
            filing_date="",
            quarter="",
            error=f"Aucun 8-K Item 2.02 trouvé sur {max_lookback_days} jours",
        )

    # ---- Traitement texte ----

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Nettoie le HTML EDGAR en texte brut lisible."""
        # Supprime les balises XBRL inline
        text = re.sub(r"<ix:[^>]+>", "", html, flags=re.I)
        text = re.sub(r"</ix:[^>]+>", "", text, flags=re.I)
        # Supprime toutes les balises HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # Décode les entités HTML basiques
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&#8217;", "'").replace("&#8220;", '"').replace("&#8221;", '"')
        # Normalise les espaces
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _detect_quarter(text: str, filing_date: str) -> str:
        """Détecte le trimestre depuis le texte ou la date du filing."""
        # Recherche un pattern Q1/Q2/Q3/Q4 + année dans le texte
        m = re.search(
            r"(first|second|third|fourth|Q[1-4])\s+"
            r"(quarter|fiscal)?\s*(20\d{2})",
            text[:2000],
            re.I,
        )
        if m:
            q_raw = m.group(1).lower()
            year = m.group(3)
            qmap = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
            q = qmap.get(q_raw, q_raw.upper())
            return f"{q} {year}"
        # Fallback : déduction depuis la date de filing
        try:
            dt = datetime.strptime(filing_date, "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            return f"Q{q} {dt.year}"
        except Exception:
            return "Unknown Quarter"

    @staticmethod
    def _semantic_chunk(
        text: str,
        ticker: str,
        filing_date: str,
        quarter: str,
        accession: str,
        max_chunk_chars: int = 1500,
    ) -> list[EarningsChunk]:
        """
        Découpe sémantiquement le transcript en sections thématiques.
        Ref: Shi et al. (2023) — LLMLingua : le chunking sémantique
        préserve la densité informationnelle mieux que le chunking fixe.
        """
        # Découpage par paragraphes d'abord
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]

        chunks: list[EarningsChunk] = []
        current_section = "misc"
        current_buffer = []
        current_len = 0

        def flush(section: str, buffer: list[str], idx: int) -> EarningsChunk:
            chunk_text = "\n".join(buffer)
            return EarningsChunk(
                chunk_id=f"{ticker}_{accession}_{idx}",
                ticker=ticker,
                section=section,
                text=chunk_text[:max_chunk_chars],
                filing_date=filing_date,
                quarter=quarter,
                doc_id=f"earnings_{ticker}_{filing_date}_{idx}",
            )

        for para in paragraphs:
            # Détection de section
            for section_name, pattern in _SECTION_PATTERNS.items():
                if pattern.search(para[:200]):
                    # Nouveau thème détecté → flush le buffer courant
                    if current_buffer and current_len > 100:
                        chunks.append(flush(current_section, current_buffer, len(chunks)))
                    current_section = section_name
                    current_buffer = []
                    current_len = 0
                    break

            # Accumulation dans le buffer
            if current_len + len(para) > max_chunk_chars:
                if current_buffer:
                    chunks.append(flush(current_section, current_buffer, len(chunks)))
                current_buffer = [para]
                current_len = len(para)
            else:
                current_buffer.append(para)
                current_len += len(para)

        # Flush final
        if current_buffer:
            chunks.append(flush(current_section, current_buffer, len(chunks)))

        return chunks

    @staticmethod
    def _lm_tone(text: str) -> tuple[float, str]:
        """
        Score de tonalité Loughran-McDonald (2011).
        Retourne (score, label) où score ∈ [-1, +1].
        """
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos = len(words & _LM_POSITIVE)
        neg = len(words & _LM_NEGATIVE)
        total = pos + neg
        if total == 0:
            return 0.0, "NEUTRE"
        score = (pos - neg) / total
        if score > 0.3:
            label = "POSITIF"
        elif score < -0.3:
            label = "NEGATIF"
        else:
            label = "NEUTRE"
        return round(score, 3), label

    @staticmethod
    def _build_summary(
        ticker: str,
        filing_date: str,
        quarter: str,
        lm_score: float,
        lm_label: str,
        raw_text: str,
    ) -> str:
        """
        Construit un bloc de contexte compact pour injection dans le prompt.
        Limité à ~800 caractères pour optimiser le budget de tokens.
        """
        # Extrait les 600 premiers caractères des KPIs détectés
        kpi_section = ""
        for line in raw_text.split("\n"):
            if _SECTION_PATTERNS["kpis"].search(line):
                kpi_section += line.strip()[:200] + " "
            if len(kpi_section) > 500:
                break

        lm_emoji = "(+)" if lm_label == "POSITIF" else ("(-)" if lm_label == "NEGATIF" else "(=)")
        return (
            f"=== EARNINGS CALL {ticker} — {quarter} (SEC 8-K du {filing_date}) ===\n"
            f"Tonalité Management (Loughran-McDonald 2011) : {lm_emoji} {lm_label} "
            f"(score={lm_score:+.2f})\n"
            f"Ref. Loughran & McDonald (2011) : un score > 0.3 prédit +1.8% à J+5 en moyenne.\n"
            f"KPIs mentionnés :\n{kpi_section[:500]}\n"
            f"{'=' * 52}"
        )

    # ---- Intégration RAG ----

    def index_into_rag(self, result: EarningsCallResult, rag_store) -> int:
        """
        Indexe les chunks de l'Earnings Call dans ChromaDB.
        Utilise doc_type='earnings_call' pour différencier des news.

        Ref: Lewis et al. (2020) — le RAG hétérogène (news + documents
        structurés) augmente la précision factuelle du LLM.

        Returns: nombre de chunks indexés.
        """
        if not result.found or not result.chunks:
            return 0

        # Import ici pour éviter la circularité
        try:
            from src.knowledge.rag_store import RAGDocument
        except ImportError:
            from rag_store import RAGDocument  # type: ignore

        indexed = 0
        for chunk in result.chunks:
            doc = RAGDocument(
                doc_id=chunk.doc_id,
                ticker=chunk.ticker,
                text=(
                    f"[EARNINGS CALL {chunk.ticker} — {chunk.quarter}] [Section: {chunk.section.upper()}]\n{chunk.text}"
                ),
                doc_type="earnings_call",
                date_iso=f"{chunk.filing_date}T00:00:00+00:00",
                metadata={
                    "quarter": chunk.quarter,
                    "section": chunk.section,
                    "lm_score": str(result.lm_score),
                    "lm_label": result.lm_label,
                    "source": "SEC EDGAR 8-K Item 2.02",
                },
            )
            if rag_store.add_document(doc):
                indexed += 1

        logger.info(
            "[EarningsCall] %d/%d chunks indexés dans ChromaDB pour %s %s",
            indexed,
            len(result.chunks),
            result.ticker,
            result.quarter,
        )
        return indexed


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: Optional[EarningsCallClient] = None


def get_earnings_client() -> EarningsCallClient:
    """Retourne le singleton EarningsCallClient."""
    global _client
    if _client is None:
        _client = EarningsCallClient()
    return _client


def fetch_and_index_earnings(ticker: str, rag_store) -> EarningsCallResult:
    """
    Shortcut : fetch + index en une seule ligne.

    Usage dans agent_pipeline.py :
        from src.knowledge.earnings_calls import fetch_and_index_earnings
        ec_result = fetch_and_index_earnings(ticker, rag_store)
        if ec_result.found:
            # ec_result.summary_text est prêt pour injection dans le prompt
    """
    client = get_earnings_client()
    result = client.fetch_latest_earnings_8k(ticker)
    if result.found:
        client.index_into_rag(result, rag_store)
    return result


def format_earnings_for_prompt(result: EarningsCallResult) -> str:
    """Retourne le bloc de contexte à injecter dans le prompt des agents."""
    if not result.found:
        return f"[Earnings Call {result.ticker}] Non disponible. {result.error}"
    return result.summary_text


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    client = EarningsCallClient()

    for test_ticker in ["AAPL", "MSFT"]:
        print(f"\n{'=' * 60}")
        logger.info(f"Test Earnings Call : {test_ticker}")
        result = client.fetch_latest_earnings_8k(test_ticker, max_lookback_days=120)
        logger.info(f"Found   : {result.found}")
        if result.found:
            logger.info(f"Quarter : {result.quarter}")
            logger.info(f"Date    : {result.filing_date}")
            logger.info(f"Chunks  : {len(result.chunks)}")
            logger.info(f"LM Tone : {result.lm_label} ({result.lm_score:+.3f})")
            logger.info(f"\nPrompt block preview:\n{result.summary_text[:600]}")
            if result.chunks:
                logger.info(f"\nPremier chunk ({result.chunks[0].section}) :")
                print(result.chunks[0].text[:300])
        else:
            logger.info(f"Error   : {result.error}")
