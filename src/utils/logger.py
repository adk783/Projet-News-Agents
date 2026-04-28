"""Logger central pour Locus News-Agents.

Pourquoi ce module ?
--------------------
Le projet historique utilise majoritairement des `print()` (252 occurrences)
qui posent 3 problemes operationnels :

1. **Pas de niveau** : impossible de filtrer DEBUG vs ERROR.
2. **Pas d'agregation** : un autopilot d'une semaine produit des Mo de stdout
   non structures, ininspectables apres coup.
3. **Encodage Windows** : `print()` echoue sur cp1252 si l'objet contient des
   emojis ou caracteres non-ASCII -> crash silencieux ou UnicodeEncodeError.

Ce module fournit :

- `get_logger(name)` : logger module-level pre-configure (handler stdout +
  formatter selon LOG_FORMAT env var).
- Format `"human"` (defaut) : `2026-04-25 14:32:01 INFO [discovery.orchestrator] Run started`
- Format `"json"`  : 1 JSON par ligne, prêt a etre ingere par Loki/ELK/Datadog.
- ASCII-safe par construction : `replace_errors=True` sur le StreamHandler.

Usage
-----
>>> from src.utils.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("ticker fetched", extra={"ticker": "AAPL", "n_articles": 3})

Politique d'adoption
--------------------
- **Nouveau code** : utiliser `get_logger()` exclusivement.
- **Code historique** : la migration `print -> log` est en cours, voir
  ADR-002 dans ARCHITECTURE_DECISIONS.md. Pas de big-bang.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Variables d'environnement (defauts dans .env.example).
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.getenv("LOG_FORMAT", "human").lower()

# --- Rotation des logs ---
# LOG_FILE : chemin du fichier (vide -> pas de fichier, console only).
# LOG_FILE_MAX_BYTES : taille max avant rotation (defaut 10 Mo).
# LOG_FILE_BACKUP_COUNT : nombre de fichiers historiques conserves
# (defaut 5, soit ~50 Mo max au total avec MAX_BYTES=10Mo).
_LOG_FILE = os.getenv("LOG_FILE", "").strip()
_LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
_LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

# Cache des loggers : evite la double-configuration si get_logger() est appele
# plusieurs fois pour le meme nom (chaque appel ajouterait un handler).
_LOGGERS: dict[str, logging.Logger] = {}


class _AsciiSafeStreamHandler(logging.StreamHandler):
    """StreamHandler tolerant aux caracteres non-encodables (cp1252 Windows).

    Sans ca, un `log.info("Société française")` peut crash le pipeline entier
    sous PowerShell par defaut (encoding='cp1252'). On force errors='replace'
    pour que les caracteres exotiques deviennent '?' au lieu de lever.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            # Encoder/decoder pour mapper les caracteres non-cp1252.
            encoding = getattr(stream, "encoding", None) or "utf-8"
            try:
                msg.encode(encoding)
                safe_msg = msg
            except UnicodeEncodeError:
                safe_msg = msg.encode(encoding, errors="replace").decode(encoding)
            stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            # Dernier rempart : on ne fait JAMAIS crasher le pipeline pour un
            # probleme de log. C'est la regle d'or des systemes long-running.
            self.handleError(record)


class _JsonFormatter(logging.Formatter):
    """Formatter JSON 1-line pour ingestion machine.

    Champs systematiques : timestamp ISO, level, logger, message.
    Tout ce qui est passe via `extra={...}` est inclus comme champ top-level.
    """

    # Champs internes du LogRecord qu'on ne reexpose pas (eviter le bruit).
    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Champs `extra={...}` du caller.
        for key, val in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                # On serialise uniquement les types JSON-natifs.
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = repr(val)
        # Exception traceback si present.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _make_formatter() -> logging.Formatter:
    """Retourne le formatter selon LOG_FORMAT (factorise stream + file)."""
    if _LOG_FORMAT == "json":
        return _JsonFormatter()
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_stream_handler() -> logging.Handler:
    """Construit le handler stdout (toujours actif)."""
    handler = _AsciiSafeStreamHandler(sys.stdout)
    handler.setFormatter(_make_formatter())
    handler.setLevel(_LOG_LEVEL)
    return handler


def _build_file_handler() -> logging.Handler | None:
    """Construit le RotatingFileHandler si LOG_FILE est configure.

    Politique : rotation par taille (10 Mo defaut) avec 5 backups, soit
    ~50 Mo max au total. Encodage UTF-8 force pour eviter les problemes
    cp1252 sur Windows lors de l'ecriture (la lecture stream gere deja
    le replace via _AsciiSafeStreamHandler).

    Retourne None si LOG_FILE n'est pas configure (mode console-only).
    """
    if not _LOG_FILE:
        return None

    log_path = Path(_LOG_FILE)
    # Cree le repertoire parent si necessaire (idempotent).
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=_LOG_FILE_MAX_BYTES,
        backupCount=_LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",  # explicite : evite ambiguite cp1252 Windows
    )
    handler.setFormatter(_make_formatter())
    handler.setLevel(_LOG_LEVEL)
    return handler


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger module-level pre-configure.

    Usage canonique :

        from src.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Hello")

    Le logger est mis en cache : appels successifs avec le meme `name`
    retournent la meme instance (pas de duplication de handlers).

    Parameters
    ----------
    name : str
        Nom du logger, typiquement `__name__` (ex: "src.discovery.orchestrator").

    Returns
    -------
    logging.Logger
        Logger configure avec le handler unique du projet.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # On evite la propagation au root logger pour ne pas dupliquer les messages
    # si un autre code (autogen, langchain) a configure son propre handler racine.
    logger.propagate = False

    # Handlers idempotents : on les attache uniquement si le logger n'en a pas.
    # - Stream handler : toujours present (stdout).
    # - File handler   : seulement si LOG_FILE configure (rotation 10Mo x 5).
    if not logger.handlers:
        logger.addHandler(_build_stream_handler())
        file_handler = _build_file_handler()
        if file_handler is not None:
            logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def configure_root(
    level: str | None = None,
    fmt: str | None = None,
    log_file: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """Reconfigure le logger root (appel unique au demarrage si besoin).

    Utile si le pipeline veut surcharger LOG_LEVEL/LOG_FORMAT/LOG_FILE a chaud :
        configure_root(level="DEBUG", log_file="logs/autopilot.log")

    Tout argument None est ignore (la config existante est conservee).
    Un appel a configure_root() invalide le cache : les prochains get_logger()
    construisent de nouveaux handlers avec la nouvelle config.
    """
    global _LOG_LEVEL, _LOG_FORMAT, _LOG_FILE, _LOG_FILE_MAX_BYTES, _LOG_FILE_BACKUP_COUNT  # noqa: PLW0603
    if level is not None:
        _LOG_LEVEL = level.upper()
    if fmt is not None:
        _LOG_FORMAT = fmt.lower()
    if log_file is not None:
        _LOG_FILE = log_file
    if max_bytes is not None:
        _LOG_FILE_MAX_BYTES = max_bytes
    if backup_count is not None:
        _LOG_FILE_BACKUP_COUNT = backup_count
    # Invalider le cache : les prochains get_logger() rebuild leur handler.
    _LOGGERS.clear()


__all__ = ["get_logger", "configure_root"]
