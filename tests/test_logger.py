"""Tests unitaires pour src.utils.logger.

Couvre :
- Format human et JSON.
- Cache des loggers (pas de duplication de handlers).
- Tolerance aux caracteres non-ASCII (Windows cp1252).
- Reconfiguration via configure_root().
- Champ `extra={...}` correctement serialise en JSON.
"""

from __future__ import annotations

import io
import json
import logging
from unittest.mock import patch

import pytest


# Helper pour reset le module logger entre tests (cache global).
@pytest.fixture(autouse=True)
def reset_logger_module():
    """Vide le cache _LOGGERS et re-import a chaque test."""
    import src.utils.logger as lg

    lg._LOGGERS.clear()
    # Reset aussi les loggers Python sous-jacents (logging.getLogger() est globale).
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("test_"):
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
    yield
    lg._LOGGERS.clear()


# -----------------------------------------------------------------------------
# Tests format human
# -----------------------------------------------------------------------------
def test_human_format_contains_level_and_message(capsys):
    """Format 'human' doit contenir level + nom logger + message."""
    with patch.dict("os.environ", {"LOG_FORMAT": "human", "LOG_LEVEL": "INFO"}):
        # Reimport pour que les env vars soient relues.
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_human")
        log.info("hello world")

    captured = capsys.readouterr()
    assert "INFO" in captured.out
    assert "test_human" in captured.out
    assert "hello world" in captured.out


def test_warning_level_visible(capsys):
    """WARNING doit etre visible meme avec LOG_LEVEL=INFO."""
    with patch.dict("os.environ", {"LOG_FORMAT": "human", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_warn")
        log.warning("attention")

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "attention" in captured.out


def test_debug_filtered_at_info_level(capsys):
    """DEBUG ne doit PAS apparaitre quand LOG_LEVEL=INFO."""
    with patch.dict("os.environ", {"LOG_FORMAT": "human", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_debug_filter")
        log.debug("invisible")
        log.info("visible")

    captured = capsys.readouterr()
    assert "invisible" not in captured.out
    assert "visible" in captured.out


# -----------------------------------------------------------------------------
# Tests format JSON
# -----------------------------------------------------------------------------
def test_json_format_is_valid_json_per_line(capsys):
    """Mode JSON : chaque ligne est un JSON valide avec les champs canoniques."""
    with patch.dict("os.environ", {"LOG_FORMAT": "json", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_json")
        log.info("event one")
        log.warning("event two")

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) == 2

    payload1 = json.loads(lines[0])
    assert payload1["level"] == "INFO"
    assert payload1["logger"] == "test_json"
    assert payload1["message"] == "event one"
    assert "ts" in payload1

    payload2 = json.loads(lines[1])
    assert payload2["level"] == "WARNING"
    assert payload2["message"] == "event two"


def test_json_extra_fields_serialized(capsys):
    """Le `extra={...}` doit apparaitre comme champs top-level dans le JSON."""
    with patch.dict("os.environ", {"LOG_FORMAT": "json", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_extra")
        log.info("trade", extra={"ticker": "AAPL", "pnl": 123.45, "side": "LONG"})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["ticker"] == "AAPL"
    assert payload["pnl"] == 123.45
    assert payload["side"] == "LONG"
    assert payload["message"] == "trade"


def test_json_non_serializable_extra_falls_back_to_repr(capsys):
    """Si `extra` contient un objet non-JSON, on log son repr() au lieu de crasher."""
    with patch.dict("os.environ", {"LOG_FORMAT": "json", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_repr")

        class CustomObj:
            def __repr__(self):
                return "CustomObj(x=1)"

        log.info("obj", extra={"obj": CustomObj()})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["obj"] == "CustomObj(x=1)"


# -----------------------------------------------------------------------------
# Tests robustesse
# -----------------------------------------------------------------------------
def test_logger_cache_returns_same_instance():
    """get_logger(name) doit retourner la MEME instance pour le meme nom."""
    import importlib

    import src.utils.logger as lg

    importlib.reload(lg)
    a = lg.get_logger("test_cache")
    b = lg.get_logger("test_cache")
    assert a is b


def test_logger_no_handler_duplication():
    """Appeler get_logger(name) plusieurs fois ne doit PAS empiler de handlers."""
    import importlib

    import src.utils.logger as lg

    importlib.reload(lg)
    log = lg.get_logger("test_dup")
    n_handlers_first = len(log.handlers)
    _ = lg.get_logger("test_dup")
    _ = lg.get_logger("test_dup")
    assert len(log.handlers) == n_handlers_first


def test_unicode_safe_no_crash_on_non_ascii(capsys):
    """Caracteres non-ASCII (accents, emojis) ne doivent JAMAIS crasher."""
    with patch.dict("os.environ", {"LOG_FORMAT": "human", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_unicode")
        # Pas d'assertion sur le contenu : selon l'encodage du stream pytest
        # capture, les accents passent ou sont remplaces par '?'. La regle est
        # qu'aucune exception ne doit etre levee.
        log.info("societe francaise — emoji rocket 🚀")
        log.info("ascii fallback OK")

    captured = capsys.readouterr()
    # Le 2eme message doit toujours apparaitre, prouvant qu'aucun crash n'a
    # ete declenche par le 1er.
    assert "ascii fallback OK" in captured.out


def test_propagate_disabled():
    """propagate=False evite la duplication via le root logger."""
    import importlib

    import src.utils.logger as lg

    importlib.reload(lg)
    log = lg.get_logger("test_propagate")
    assert log.propagate is False


def test_configure_root_invalidates_cache():
    """configure_root() doit vider le cache pour que les prochains get_logger
    rebuild avec le nouveau format."""
    import importlib

    import src.utils.logger as lg

    importlib.reload(lg)

    # On ne stocke pas la reference : ce qui nous interesse c'est l'effet
    # de bord (entree dans le cache _LOGGERS).
    lg.get_logger("test_reconf")
    assert "test_reconf" in lg._LOGGERS

    lg.configure_root(level="DEBUG", fmt="json")
    assert lg._LOGGERS == {}, "le cache doit etre vide apres configure_root"

    log_b = lg.get_logger("test_reconf")
    # log_b doit etre une nouvelle instance reconfiguree.
    assert log_b is not None
    assert lg._LOG_LEVEL == "DEBUG"
    assert lg._LOG_FORMAT == "json"


# -----------------------------------------------------------------------------
# Test exception traceback (bonus : couvre la branche record.exc_info)
# -----------------------------------------------------------------------------
def test_exception_includes_traceback_in_json(capsys):
    """log.exception() doit inclure le traceback dans le payload JSON."""
    with patch.dict("os.environ", {"LOG_FORMAT": "json", "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_exc")
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("erreur attrapee")

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["level"] == "ERROR"
    assert "exception" in payload
    assert "ValueError" in payload["exception"]
    assert "boom" in payload["exception"]


# -----------------------------------------------------------------------------
# Tests rotation des logs (RotatingFileHandler)
# -----------------------------------------------------------------------------
def test_log_file_handler_disabled_by_default(tmp_path):
    """LOG_FILE absent -> pas de file handler attache."""
    with patch.dict("os.environ", {}, clear=True):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_no_file")
        # Un seul handler : le stream stdout, pas de file handler.
        from logging.handlers import RotatingFileHandler

        file_handlers = [h for h in log.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers == []


def test_log_file_handler_writes_to_file(tmp_path):
    """LOG_FILE configure -> les messages sont ecrits dans le fichier."""
    log_path = tmp_path / "subdir" / "app.log"  # subdir = teste creation parent
    with patch.dict("os.environ", {"LOG_FILE": str(log_path), "LOG_LEVEL": "INFO"}):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_file_write")
        log.info("message persiste")
        # Force le flush des handlers avant lecture.
        for h in log.handlers:
            h.flush()

    assert log_path.exists(), "le fichier de log doit etre cree (parent inclus)"
    content = log_path.read_text(encoding="utf-8")
    assert "message persiste" in content
    assert "INFO" in content


def test_log_file_rotation_creates_backup(tmp_path):
    """Quand le fichier depasse maxBytes, un backup .1 est cree."""
    log_path = tmp_path / "rotate.log"
    # Taille tres petite (200 octets) pour declencher rotation rapidement.
    with patch.dict(
        "os.environ",
        {
            "LOG_FILE": str(log_path),
            "LOG_FILE_MAX_BYTES": "200",
            "LOG_FILE_BACKUP_COUNT": "3",
            "LOG_LEVEL": "INFO",
        },
    ):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_rotation")
        # Ecrit assez de bytes pour declencher au moins 1 rotation.
        for i in range(20):
            log.info("ligne assez longue pour gonfler le fichier rapidement %d", i)
        for h in log.handlers:
            h.flush()

    # Le fichier principal existe + au moins 1 backup .1
    assert log_path.exists()
    backup = log_path.with_name(log_path.name + ".1")
    assert backup.exists(), f"backup non cree, fichiers presents : {list(tmp_path.iterdir())}"


def test_log_file_backup_count_capped(tmp_path):
    """backupCount=2 -> au max 2 backups conserves (.1 et .2)."""
    log_path = tmp_path / "capped.log"
    with patch.dict(
        "os.environ",
        {
            "LOG_FILE": str(log_path),
            "LOG_FILE_MAX_BYTES": "150",
            "LOG_FILE_BACKUP_COUNT": "2",
            "LOG_LEVEL": "INFO",
        },
    ):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)
        log = lg.get_logger("test_cap")
        # Beaucoup d'ecritures -> plusieurs rotations, mais cap a 2 backups.
        for i in range(50):
            log.info("ligne longue pour rotation %d %s", i, "x" * 30)
        for h in log.handlers:
            h.flush()

    backup_3 = log_path.with_name(log_path.name + ".3")
    assert not backup_3.exists(), ".3 ne doit pas exister (backupCount=2)"


def test_configure_root_log_file_attaches_handler(tmp_path):
    """configure_root(log_file=...) attache un file handler aux nouveaux loggers."""
    log_path = tmp_path / "configured.log"
    with patch.dict("os.environ", {}, clear=True):
        import importlib

        import src.utils.logger as lg

        importlib.reload(lg)

        # Initialement : pas de file handler
        log_a = lg.get_logger("test_pre_configure")
        from logging.handlers import RotatingFileHandler

        assert not any(isinstance(h, RotatingFileHandler) for h in log_a.handlers)

        # Apres configure_root, les nouveaux loggers ont le file handler
        lg.configure_root(log_file=str(log_path))
        log_b = lg.get_logger("test_post_configure")
        log_b.info("apres configure_root")
        for h in log_b.handlers:
            h.flush()

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "apres configure_root" in content
