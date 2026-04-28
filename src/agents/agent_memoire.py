"""
agent_memoire.py — Agent de Nuit (inspiré du système "AutoDream" d'Anthropic)

Référence architecture : coordinatorMode.ts / AutoDream (Claude Code leak)

Le système fonctionne en 4 phases, fidèles au pattern AutoDream :
  1. Orientation    — charge l'index MEMORY.md (< 200 lignes) pour rappel rapide
  2. Collecte       — lit les décisions du jour depuis SQLite
  3. Consolidation  — un LLM fusionne, déduplique, et enrichit les souvenirs
  4. Élagage/Index  — réécrit MEMORY.md + les fichiers de mémoire par ticker

Règles strictes (issues du code source d'Anthropic) :
  - Toutes les dates relatives → horodatages absolus (ISO 8601)
  - Les faits contradictoires sont effacés (le plus récent prime)
  - MEMORY.md reste sous 200 lignes (index pur, pas de contenu)
  - Chaque ticker a son propre fichier de mémoire (data/memory/<TICKER>.md)

Usage :
    python -m src.agents.agent_memoire            # run immédiat
    python -m src.agents.agent_memoire --dry-run  # affiche sans écrire
"""

import argparse
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.llm_client import AllProvidersFailedError, LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client LLM unifie (cf. ADR-003)
# ---------------------------------------------------------------------------
_MEMOIRE_CLIENT: LLMClient | None = None


def _get_memoire_client() -> LLMClient:
    """Lazy-init du LLMClient partage pour la consolidation memoire."""
    global _MEMOIRE_CLIENT  # noqa: PLW0603
    if _MEMOIRE_CLIENT is None:
        _MEMOIRE_CLIENT = LLMClient.from_env()
    return _MEMOIRE_CLIENT


# ---------------------------------------------------------------------------
# Chemins & constantes
# ---------------------------------------------------------------------------

DATABASE_PATH = "data/news_database.db"
MEMORY_DIR = Path("data/memory")
MEMORY_INDEX = Path("data/memory/MEMORY.md")
MAX_INDEX_LINES = 200  # Règle AutoDream : index < 200 lignes


# ---------------------------------------------------------------------------
# Client LLM (consolidation) — préfère un gros modèle fiable
# ---------------------------------------------------------------------------

# `_get_consolidation_client()` retire — remplace par `_get_memoire_client()`
# en haut du module qui utilise LLMClient avec fallback automatique.


# ---------------------------------------------------------------------------
# PHASE 1 — Orientation : lecture de l'index MEMORY.md
# ---------------------------------------------------------------------------


def _load_memory_index() -> str:
    """
    Charge le fichier MEMORY.md s'il existe.
    Retourne son contenu ou une chaîne vide.
    Ce fichier est conçu pour être chargé à chaque démarrage de session.
    """
    if not MEMORY_INDEX.exists():
        logger.info("[AutoDream] Aucun index MEMORY.md trouvé — première exécution.")
        return ""
    content = MEMORY_INDEX.read_text(encoding="utf-8")
    logger.info("[AutoDream] Index MEMORY.md chargé (%d lignes).", content.count("\n"))
    return content


def _load_ticker_memory(ticker: str) -> str:
    """Charge le fichier de mémoire d'un ticker spécifique."""
    path = MEMORY_DIR / f"{ticker}.md"
    if not path.exists():
        return f"# Mémoire {ticker}\n*Aucun historique enregistré.*\n"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# PHASE 2 — Collecte des signaux du jour depuis SQLite
# ---------------------------------------------------------------------------


def _collect_todays_signals(date_str: Optional[str] = None) -> dict[str, list[dict]]:
    """
    Lit les décisions finalisées depuis SQLite et les groupe par ticker.

    Args:
        date_str : date au format YYYY-MM-DD. Défaut = aujourd'hui UTC.

    Returns:
        dict { "AAPL": [{"title": ..., "signal": ..., ...}, ...], ... }
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    signals_by_ticker: dict[str, list[dict]] = {}

    try:
        with sqlite3.connect(DATABASE_PATH, timeout=15) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ticker, title, date_utc, signal_final, consensus_rate,
                       impact_strength, argument_dominant, absa_json
                FROM articles
                WHERE signal_final IS NOT NULL
                  AND signal_final != 'Rejeté (Filtre)'
                  AND signal_final != 'Erreur'
                  AND date_utc LIKE ?
                ORDER BY ticker, date_utc DESC
                """,
                (f"{date_str}%",),
            )
            rows = cursor.fetchall()

        for row in rows:
            ticker = row["ticker"]
            if ticker not in signals_by_ticker:
                signals_by_ticker[ticker] = []
            # Conversion des dates relatives → horodatages absolus (règle AutoDream)
            date_absolute = _normalize_date(row["date_utc"])
            signals_by_ticker[ticker].append(
                {
                    "date": date_absolute,
                    "title": row["title"],
                    "signal": row["signal_final"],
                    "consensus_rate": row["consensus_rate"],
                    "impact_strength": row["impact_strength"],
                    "argument_dominant": row["argument_dominant"],
                    "absa_aspects": _parse_absa_aspects(row["absa_json"]),
                }
            )

        logger.info("[AutoDream] Collecte : %d ticker(s) avec décisions pour %s.", len(signals_by_ticker), date_str)
    except Exception as e:
        logger.error("[AutoDream] Erreur collecte SQLite : %s", e)

    return signals_by_ticker


def _normalize_date(date_raw: str) -> str:
    """
    Règle AutoDream #1 : convertit TOUJOURS les dates en horodatage absolu ISO 8601.
    Évite les expressions relatives ('aujourd'hui', 'hier') dans la mémoire.
    """
    if not date_raw:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Si déjà ISO-like, on retourne tel quel
    if re.match(r"\d{4}-\d{2}-\d{2}", date_raw):
        return date_raw
    # Tentative de parse
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_raw[:19], fmt[: len(fmt)])
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return date_raw


def _parse_absa_aspects(absa_json: Optional[str]) -> list[str]:
    """Extrait les aspects détectés de la colonne absa_json."""
    if not absa_json:
        return []
    try:
        data = json.loads(absa_json)
        return [f"{a['aspect']}({a['sentiment']})" for a in data.get("aspects", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# PHASE 3 — Consolidation LLM : fusion + déduplication + enrichissement
# ---------------------------------------------------------------------------

PROMPT_CONSOLIDATION_SYSTEM = """Tu es un gestionnaire de mémoire financière à long terme.
Tu reçois :
  1. La mémoire existante d'un ticker (historique des semaines précédentes)
  2. Les nouvelles décisions d'aujourd'hui pour ce même ticker

Ta mission est de produire une MÉMOIRE MISE À JOUR selon ces règles strictes :

RÈGLES DE CONSOLIDATION (obligatoires) :
1. FORMAT DES DATES : Utilise TOUJOURS des horodatages absolus ISO 8601 (YYYY-MM-DD).
   JAMAIS "aujourd'hui", "hier", "la semaine dernière". Toujours "Le 2026-04-13".
2. DÉDUPLICATION : Si un fait existe déjà dans la mémoire, ne le répète pas.
   Si le nouveau fait contredit l'ancien, le nouveau prime — supprime l'ancien.
3. STRUCTURE FIXE : La mémoire d'un ticker DOIT suivre ce format exact :

## [TICKER] — Mémoire consolidée

### Tendance récente (30 derniers jours)
- [YYYY-MM-DD] Signal: [Achat|Vente|Neutre] | Force: [0.0-1.0] | Aspects: [liste]
  Argument clé : [une phrase]

### Faits persistants (valides > 30 jours)
- [fait durable avec date de première observation]

### Signaux contradictoires actifs
- [si deux signaux opposés récents existent, les noter ici]

### Contexte sectoriel
- [informations sur le secteur/industrie utiles pour les prochaines analyses]

4. LONGUEUR MAX : 60 lignes par ticker. Élague les entrées > 30 jours dans "Tendance récente".
5. NEUTRALITÉ : Ne fais pas de recommandations. Résume les faits observés.

Réponds UNIQUEMENT avec le contenu markdown mis à jour, sans preamble ni explication."""


def _consolidate_ticker_memory(
    ticker: str,
    existing_memory: str,
    new_signals: list[dict],
    client: LLMClient,
    model: str = "auto",  # conserve la signature, model gere par LLMClient
) -> str:
    """
    Appelle le LLM pour consolider la mémoire d'un ticker.
    Retourne la mémoire mise à jour en markdown.
    """
    # Formatage des nouvelles décisions pour le prompt
    new_signals_text = "\n".join(
        [
            f"- [{s['date']}] Signal: {s['signal']} | "
            f"Consensus: {s['consensus_rate']:.0%} | "
            f"Force: {s['impact_strength']:.2f} | "
            f"Aspects: {', '.join(s['absa_aspects']) or 'aucun'}\n"
            f"  Titre: {s['title']}\n"
            f"  Argument: {s['argument_dominant']}"
            for s in new_signals
        ]
    )

    user_prompt = (
        f"=== MÉMOIRE EXISTANTE DE {ticker} ===\n"
        f"{existing_memory}\n\n"
        f"=== NOUVELLES DÉCISIONS DU JOUR ({datetime.now(timezone.utc).strftime('%Y-%m-%d')}) ===\n"
        f"{new_signals_text}\n\n"
        f"Produis la mémoire consolidée mise à jour pour {ticker}."
    )

    # Strategie de modele (ADR-016, fonde sur bench live 2026-04-26) :
    # - PREMIER essai : NIM Llama 3.1 405B pour qualite de synthese maximale.
    #   Cette tache est BATCH (consolidation hebdo nightly), 2 min de latence
    #   est totalement acceptable, et le 405B domine le 70B sur LongBench
    #   (Meta Llama 3.1 paper, juillet 2024).
    # - FALLBACK : Groq -> Mistral -> Cerebras si NIM down ou cle absente.
    # - Si tout echoue : append brut sans LLM (degrade gracieux, pas de crash).
    try:
        # 1ere tentative : modele haute qualite via NIM (long_summarization)
        if "nvidia_nim" in client.available_providers():
            try:
                response = client.complete(
                    messages=[
                        {"role": "system", "content": PROMPT_CONSOLIDATION_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    model_preference=["nvidia_nim"],
                    model_override="meta/llama-3.1-405b-instruct",
                    temperature=0.1,
                    max_tokens=2000,
                )
                return response.content.strip()
            except (AllProvidersFailedError, Exception) as nim_err:
                logger.warning(
                    "[AutoDream] NIM 405B indisponible (%s), fallback Groq/Mistral/Cerebras",
                    type(nim_err).__name__,
                )

        # Fallback : modeles plus petits mais fiables
        response = client.complete(
            messages=[
                {"role": "system", "content": PROMPT_CONSOLIDATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model_preference=["groq", "mistral", "cerebras"],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.content.strip()
    except (AllProvidersFailedError, Exception) as e:
        logger.error("[AutoDream] Erreur LLM consolidation %s : %s", ticker, e)
        # Fallback degrade : on appende les nouvelles decisions sans LLM
        return (
            existing_memory
            + f"\n\n### Ajout brut {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n{new_signals_text}"
        )


# ---------------------------------------------------------------------------
# PHASE 4 — Élagage & Mise à jour de l'index MEMORY.md
# ---------------------------------------------------------------------------


def _write_ticker_memory(ticker: str, content: str, dry_run: bool = False) -> None:
    """Écrit le fichier de mémoire d'un ticker."""
    path = MEMORY_DIR / f"{ticker}.md"
    if dry_run:
        logger.info("[AutoDream][DRY-RUN] Écriture simulée : %s (%d lignes)", path, content.count("\n"))
        return
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("[AutoDream] Mémoire %s mise à jour → %s", ticker, path)


def _rebuild_memory_index(tickers_updated: list[str], dry_run: bool = False) -> None:
    """
    Règle AutoDream #4 : reconstruit MEMORY.md < 200 lignes.
    C'est un index pur — il liste les fichiers disponibles et leur résumé en 1 ligne.
    Il est chargé à chaque démarrage de session pour un rappel instantané.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# MEMORY.md — Index de mémoire financière\n",
        f"*Dernière mise à jour : {now}*  \n",
        "*Généré par Agent de Nuit (AutoDream pattern)*\n\n",
        "## Fichiers de mémoire disponibles\n\n",
        "| Ticker | Fichier | Dernière MAJ | Résumé |\n",
        "|--------|---------|--------------|--------|\n",
    ]

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    for md_file in sorted(MEMORY_DIR.glob("*.md")):
        if md_file.name == "MEMORY.md":
            continue
        ticker = md_file.stem
        content = md_file.read_text(encoding="utf-8")
        # Extraction de la première ligne de tendance pour le résumé
        summary = "—"
        for line in content.splitlines():
            if line.startswith("- [20"):  # ligne de tendance ISO date
                summary = line[2:80] + ("…" if len(line) > 80 else "")
                break
        mtime = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        lines.append(f"| {ticker} | [{ticker}.md](memory/{ticker}.md) | {mtime} | {summary} |\n")

    lines.append("\n## Règles de consolidation actives\n\n")
    lines.append("- Dates : horodatages absolus ISO 8601 uniquement\n")
    lines.append("- Déduplication : le fait le plus récent prime\n")
    lines.append("- Élagage : tendances > 30j déplacées dans 'Faits persistants'\n")
    lines.append(f"- Cet index est limité à {MAX_INDEX_LINES} lignes\n")
    lines.append("\n## Usage\n\n")
    lines.append("```python\n")
    lines.append("from src.agents.agent_memoire import load_context_for_session\n")
    lines.append("context = load_context_for_session(['AAPL', 'MSFT'])  # inject dans les prompts\n")
    lines.append("```\n")

    index_content = "".join(lines)
    line_count = index_content.count("\n")

    # Règle AutoDream : troncature si > MAX_INDEX_LINES
    if line_count > MAX_INDEX_LINES:
        truncated = "\n".join(index_content.splitlines()[:MAX_INDEX_LINES])
        index_content = truncated + f"\n\n*[Index tronqué à {MAX_INDEX_LINES} lignes — voir fichiers individuels]*\n"
        logger.warning("[AutoDream] MEMORY.md tronqué à %d lignes.", MAX_INDEX_LINES)

    if dry_run:
        logger.info("[AutoDream][DRY-RUN] MEMORY.md simulé (%d lignes).", line_count)
        print(index_content)
        return

    MEMORY_INDEX.write_text(index_content, encoding="utf-8")
    logger.info("[AutoDream] MEMORY.md mis à jour (%d lignes).", line_count)


# ---------------------------------------------------------------------------
# API publique : chargement de contexte pour injection en session
# ---------------------------------------------------------------------------


def load_context_for_session(tickers: list[str]) -> str:
    """
    Point d'entrée pour les pipelines : charge la mémoire des tickers demandés
    depuis les fichiers .md et retourne un contexte prêt à injecter dans un prompt.

    Usage dans agent_pipeline.py :
        from src.agents.agent_memoire import load_context_for_session
        memory_ctx = load_context_for_session(["AAPL"])
        # → injecter dans le task_prompt du débat
    """
    if not MEMORY_DIR.exists():
        return ""

    sections = []
    for ticker in tickers:
        content = _load_ticker_memory(ticker)
        if "*Aucun historique" not in content:
            sections.append(content)

    if not sections:
        return ""

    return (
        "=== MÉMOIRE HISTORIQUE (Agent de Nuit) ===\n"
        + "\n---\n".join(sections)
        + "\n==========================================\n"
    )


# ---------------------------------------------------------------------------
# Orchestration complète (4 phases AutoDream)
# ---------------------------------------------------------------------------


def run_nightly_consolidation(date_str: Optional[str] = None, dry_run: bool = False) -> None:
    """
    Exécute le cycle complet de consolidation mémoire (AutoDream).

    Args:
        date_str : YYYY-MM-DD de la journée à consolider. Défaut = aujourd'hui UTC.
        dry_run  : si True, affiche sans écrire sur disque.
    """
    logger.info("=" * 60)
    logger.info("[AutoDream] Démarrage de la consolidation mémoire.")
    logger.info("=" * 60)

    # Phase 1 — Orientation
    _load_memory_index()  # log uniquement, pas utilisé dans la consolidation elle-même

    # Phase 2 — Collecte
    signals_by_ticker = _collect_todays_signals(date_str)

    if not signals_by_ticker:
        logger.info("[AutoDream] Aucune décision à consolider pour cette journée.")
        return

    # Initialisation du client LLM de consolidation
    client = _get_memoire_client()
    available = client.available_providers()
    if not available:
        logger.error("[AutoDream] Aucun provider LLM disponible — consolidation annulee.")
        return
    logger.info("[AutoDream] LLM de consolidation via LLMClient (chain: %s)", available)
    model = "auto"  # selectionne par LLMClient au runtime

    # Phase 3 — Consolidation par ticker
    tickers_updated = []
    for ticker, new_signals in signals_by_ticker.items():
        logger.info("[AutoDream] Consolidation %s (%d décision(s))...", ticker, len(new_signals))
        existing = _load_ticker_memory(ticker)
        updated = _consolidate_ticker_memory(ticker, existing, new_signals, client, model)

        # Phase 4a — Écriture du fichier de mémoire
        _write_ticker_memory(ticker, updated, dry_run=dry_run)
        tickers_updated.append(ticker)

    # Phase 4b — Reconstruction de l'index MEMORY.md
    _rebuild_memory_index(tickers_updated, dry_run=dry_run)

    logger.info(
        "[AutoDream] Consolidation terminée. %d ticker(s) mis à jour : %s", len(tickers_updated), tickers_updated
    )


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Agent de Nuit — Consolidation mémoire AutoDream")
    parser.add_argument("--date", default=None, help="Date à consolider (YYYY-MM-DD). Défaut = aujourd'hui UTC.")
    parser.add_argument("--dry-run", action="store_true", help="Simule la consolidation sans écrire sur disque.")
    args = parser.parse_args()

    run_nightly_consolidation(date_str=args.date, dry_run=args.dry_run)
