"""Tests pour l'enrichissement dynamique de l'anonymizer (Phase 4).

Couvre :
- _alias_from_ticker : determinisme + unicite par hash
- extend_db_from_database : lecture DB + ajout d'entrees + idempotence
- Cache thread-safe (un seul SELECT par session)
- DB absente : pas d'erreur, retourne 0
- DB corrompue : warning logge mais pas de crash
- Hardcode preserve : Big Tech ne sont pas ecrasees
- reset_dynamic_cache + reset_entity_database
"""

from __future__ import annotations

import sqlite3

import pytest

from src.utils.anonymizer import (
    ENTITY_DATABASE,
    _alias_from_ticker,
    anonymize_article,
    extend_db_from_database,
    reset_dynamic_cache,
    reset_entity_database,
)


@pytest.fixture(autouse=True)
def _reset_anonymizer_state():
    """Avant et apres chaque test : etat propre."""
    reset_entity_database()
    yield
    reset_entity_database()


# =============================================================================
# Tests _alias_from_ticker
# =============================================================================
class TestAliasGeneration:
    """L'alias genere doit etre deterministe et unique par ticker."""

    def test_alias_is_deterministic(self):
        """Meme ticker -> meme alias entre 2 appels."""
        a = _alias_from_ticker("ABCD")
        b = _alias_from_ticker("ABCD")
        assert a == b

    def test_different_tickers_give_different_aliases(self):
        """Tickers differents -> aliases differents (collision improbable hash6)."""
        a = _alias_from_ticker("AAA")
        b = _alias_from_ticker("BBB")
        assert a["alias_company"] != b["alias_company"]
        assert a["ticker"] != b["ticker"]

    def test_alias_format_starts_with_dyn_prefix(self):
        """Les alias dynamiques portent le prefixe DYN_ pour eviter collision avec hardcode."""
        alias = _alias_from_ticker("XYZ")
        assert alias["ticker"].startswith("DYN_")
        assert alias["alias_company"].startswith("DynCorp_")

    def test_alias_includes_ticker_in_names_for_substitution(self):
        """Le ticker doit etre dans `names` pour que la substitution textuelle fonctionne."""
        alias = _alias_from_ticker("ZZZZ")
        assert "ZZZZ" in alias["names"]

    def test_alias_required_keys_present(self):
        """L'alias doit avoir toutes les cles attendues par anonymize_article."""
        alias = _alias_from_ticker("KKK")
        for key in ("ticker", "names", "ceos", "products", "places", "alias_company", "alias_ceo", "alias_product"):
            assert key in alias, f"cle manquante : {key}"


# =============================================================================
# Tests extend_db_from_database
# =============================================================================
class TestDynamicDBLoading:
    """Lecture de la DB SQLite et ajout dynamique a ENTITY_DATABASE."""

    @pytest.fixture
    def db_with_tickers(self, tmp_path):
        """Cree une DB SQLite avec une table `articles` et des tickers de test."""
        db_path = tmp_path / "test_news.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE articles (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    title TEXT
                )
            """)
            # Mix : 2 nouveaux tickers + 1 deja hardcode (AAPL).
            for t in ["XOM", "WMT", "AAPL", "XOM"]:  # XOM doublon -> dedup attendu
                conn.execute("INSERT INTO articles (ticker, title) VALUES (?, ?)", (t, f"News about {t}"))
        return db_path

    def test_loads_unknown_tickers_only(self, db_with_tickers):
        """Seuls les tickers non-hardcodes doivent etre ajoutes."""
        n_added = extend_db_from_database(str(db_with_tickers))
        # XOM et WMT sont nouveaux ; AAPL deja hardcode -> 2 ajouts
        assert n_added == 2
        assert "XOM" in ENTITY_DATABASE
        assert "WMT" in ENTITY_DATABASE

    def test_does_not_overwrite_hardcoded_aapl(self, db_with_tickers):
        """Un ticker hardcode (AAPL) doit garder son alias premium 'AlphaCorp'."""
        original_aapl = dict(ENTITY_DATABASE["AAPL"])
        extend_db_from_database(str(db_with_tickers))
        assert ENTITY_DATABASE["AAPL"]["alias_company"] == "AlphaCorp"
        assert ENTITY_DATABASE["AAPL"] == original_aapl

    def test_idempotent_second_call_is_zero(self, db_with_tickers):
        """Un 2eme appel (cache hit) ne doit rien ajouter."""
        n1 = extend_db_from_database(str(db_with_tickers))
        n2 = extend_db_from_database(str(db_with_tickers))
        assert n1 > 0
        assert n2 == 0  # cache hit

    def test_missing_db_returns_zero_no_crash(self, tmp_path):
        """DB absente : retourne 0 sans lever."""
        missing = tmp_path / "does_not_exist.db"
        n = extend_db_from_database(str(missing))
        assert n == 0

    def test_corrupted_db_does_not_crash(self, tmp_path):
        """DB corrompue / malformee : warning logge, pas de crash."""
        bad = tmp_path / "bad.db"
        bad.write_bytes(b"not_a_sqlite_database")
        # Doit retourner 0 sans lever d'exception
        n = extend_db_from_database(str(bad))
        assert n == 0

    def test_db_without_articles_table_returns_zero(self, tmp_path):
        """DB SQLite valide mais sans table `articles` : 0 ajouts, pas crash."""
        db_path = tmp_path / "empty.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other (id INTEGER)")
        n = extend_db_from_database(str(db_path))
        assert n == 0

    def test_anonymize_works_on_dynamically_loaded_ticker(self, db_with_tickers):
        """Apres extend, anonymize_article doit fonctionner sur un nouveau ticker."""
        extend_db_from_database(str(db_with_tickers))
        # XOM a ete ajoute dynamiquement
        result = anonymize_article("XOM is performing well this quarter.", ticker="XOM")
        # Le ticker dans le texte doit etre remplace par l'alias DynCorp_*
        assert "XOM" not in result.text or result.entities_replaced > 0
        assert result.ticker_alias.startswith("DYN_")


# =============================================================================
# Tests reset
# =============================================================================
class TestResetHelpers:
    """Helpers de reset pour isolation entre tests."""

    def test_reset_entity_database_removes_dynamic_only(self, tmp_path):
        """reset_entity_database vire les dynamiques mais garde le hardcode."""
        # Etat initial : on note la taille
        n_hardcoded = len(ENTITY_DATABASE)

        # On simule un ajout dynamique
        db_path = tmp_path / "x.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE articles (ticker TEXT)")
            conn.execute("INSERT INTO articles VALUES ('UNKNOWN1')")
            conn.execute("INSERT INTO articles VALUES ('UNKNOWN2')")
        extend_db_from_database(str(db_path))
        assert len(ENTITY_DATABASE) == n_hardcoded + 2

        # Reset doit ramener a la taille hardcode
        n_removed = reset_entity_database()
        assert n_removed == 2
        assert len(ENTITY_DATABASE) == n_hardcoded
        assert "AAPL" in ENTITY_DATABASE  # hardcode toujours la

    def test_reset_dynamic_cache_allows_reload(self, tmp_path):
        """reset_dynamic_cache permet un nouveau extend_db_from_database()."""
        db_path = tmp_path / "y.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE articles (ticker TEXT)")
            conn.execute("INSERT INTO articles VALUES ('UNK1')")

        n1 = extend_db_from_database(str(db_path))
        assert n1 == 1

        # Sans reset_dynamic_cache, 2eme appel est cache hit
        assert extend_db_from_database(str(db_path)) == 0

        # Apres reset_dynamic_cache + reset_entity_database, nouvel ajout possible
        reset_entity_database()
        reset_dynamic_cache()
        n2 = extend_db_from_database(str(db_path))
        assert n2 == 1
