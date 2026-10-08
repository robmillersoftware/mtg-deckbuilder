"""CardService shortlist pieces: shared SQL filters, full-text names, popular names, FTS index."""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.services import card_service as cs
from app.services.card_service import CardService


def svc(rows=()):
    s = CardService.__new__(CardService)
    result = MagicMock()
    result.all.return_value = list(rows)
    s.db = MagicMock(execute=AsyncMock(return_value=result))
    return s


def sql_of(s):
    stmt, params = s.db.execute.call_args[0]
    return str(stmt), params


class TestFilterSql:
    def test_format_legality_and_colors(self):
        assert CardService._filter_sql("modern", True, ["r", "x"]) == [
            "legalities->>'modern' = 'legal'",
            "(colors <@ ARRAY['R']::varchar[] OR colors = '{}' OR colors IS NULL)",
        ]

    def test_standard_only_without_format_with_alias(self):
        assert CardService._filter_sql(None, True, None, alias="c") == ["c.is_standard_legal = true"]

    def test_view_queries_skip_legality(self):
        assert CardService._filter_sql("standard", True, None, legality=False) == []


class TestTextSearchNames:
    async def test_ors_query_words_and_filters(self):
        s = svc([("Shock",), ("Lightning Strike",)])
        names = await s.text_search_names("Cheap red removal!", format="standard", colors=["R"])
        assert names == ["Shock", "Lightning Strike"]
        sql, params = sql_of(s)
        assert params["q"] == "cheap | removal"
        assert cs.CARD_TSVECTOR in sql and "FROM cards" in sql
        assert "legalities->>'standard' = 'legal'" in sql
        assert "colors <@ ARRAY['R']" in sql

    async def test_colors_and_filler_words_are_dropped(self):
        s = svc([("Shock",)])
        await s.text_search_names("cheap Red cards, blue deck, R U")
        assert sql_of(s)[1]["q"] == "cheap"

    async def test_all_filler_skips_the_query(self):
        s = svc()
        assert await s.text_search_names("red cards", colors=["R"]) == []
        s.db.execute.assert_not_called()

    async def test_no_words_skips_the_query(self):
        s = svc()
        assert await s.text_search_names("?! --") == []
        s.db.execute.assert_not_called()


class TestRoleCardNames:
    async def test_roles_from_query_words_and_filters(self):
        s = svc([("Shock",), ("Bolt",)])
        names = await s.role_card_names("cheap red burn", format="modern", colors=["R"])
        assert names == ["Shock", "Bolt"]
        sql, params = sql_of(s)
        assert params == {"roles": ["burn"], "limit": cs.SHORTLIST_SIZE}
        assert "FROM card_roles r JOIN cards c ON c.id = r.card_id" in sql
        assert "c.legalities->>'modern' = 'legal'" in sql and "c.colors <@ ARRAY['R']" in sql
        assert "MAX(r.efficiency) DESC, MAX(r.confidence) DESC, c.name" in sql

    async def test_singular_matches_plural_key_and_expands(self):
        s = svc()
        await s.role_card_names("Counterspell and removal")
        assert sql_of(s)[1]["roles"] == sorted(
            ["counterspell", "removal_targeted", "removal_mass", "removal_artifact_enchantment"])

    async def test_whole_words_only(self):
        s = svc()
        assert await s.role_card_names("burnout rampage") == []
        s.db.execute.assert_not_called()

    async def test_no_role_skips_the_query(self):
        s = svc()
        assert await s.role_card_names("something spooky") == []
        s.db.execute.assert_not_called()


async def test_popular_names_exclude_basics_and_filter_on_card_alias():
    s = svc([("Burst Lightning", 45)])
    assert await s.popular_card_names(format="standard", colors=["R"]) == ["Burst Lightning"]
    sql, params = sql_of(s)
    assert "NOT LIKE 'Basic Land%'" in sql and "c.legalities->>'standard' = 'legal'" in sql
    assert params == {"format": "standard", "limit": cs.SHORTLIST_SIZE}


async def test_popular_names_default_format_is_standard():
    s = svc()
    await s.popular_card_names()
    assert sql_of(s)[1]["format"] == "standard"


async def test_tournament_frequency_lives_on_card_service():
    s = svc([("shock", 7)])
    assert await s.tournament_frequency(["Shock"], format="modern") == {"shock": 7}
    empty = svc()
    assert await empty.tournament_frequency([]) == {}
    empty.db.execute.assert_not_called()


def test_migration_index_matches_query_expression():
    path = Path(__file__).parents[1] / "alembic" / "versions" / "016_add_cards_fts_index.py"
    spec = importlib.util.spec_from_file_location("migration_016", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.revision == "016" and m.down_revision == "015"
    assert m.CARD_TSVECTOR == cs.CARD_TSVECTOR
