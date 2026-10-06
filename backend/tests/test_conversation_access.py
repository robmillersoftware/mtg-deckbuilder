"""Anonymous conversations can be reloaded; nobody can read or continue another user's."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.routes.conversations import get_conversation
from app.models.conversation import Conversation
from app.services.chat_service import ChatService

ALICE, BOB = uuid4(), uuid4()


def db_with(conv):
    result = MagicMock()
    result.scalar_one_or_none.return_value = conv
    return MagicMock(execute=AsyncMock(return_value=result), add=MagicMock(), flush=AsyncMock())


def conv(owner):
    c = Conversation(user_id=owner, messages=[])
    c.id = uuid4()
    return c


@pytest.mark.parametrize("owner,caller,ok", [
    (None, None, True), (None, ALICE, True), (ALICE, ALICE, True),
    (ALICE, None, False), (ALICE, BOB, False),
])
async def test_get_conversation_access(owner, caller, ok):
    c = conv(owner)
    user = SimpleNamespace(id=caller) if caller else None
    if ok:
        assert await get_conversation(c.id, user, db_with(c)) is c
    else:
        with pytest.raises(HTTPException) as e:
            await get_conversation(c.id, user, db_with(c))
        assert e.value.status_code == 404


async def chat_conversation(existing, user_id):
    svc = ChatService.__new__(ChatService)
    svc.db = db_with(existing)
    return await svc._get_or_create_conversation(existing.id, user_id)


async def test_signing_in_claims_an_anonymous_conversation():
    c = conv(None)
    assert await chat_conversation(c, ALICE) is c
    assert c.user_id == ALICE


async def test_cannot_continue_another_users_conversation():
    c = conv(ALICE)
    got = await chat_conversation(c, BOB)
    assert got is not c and got.user_id == BOB and c.messages == []


def list_db(rows):
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return MagicMock(execute=AsyncMock(return_value=result))


def where_sql(db):
    stmt = db.execute.call_args.args[0]
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


async def test_signed_out_list_is_this_browsers_anonymous_conversations():
    from app.api.routes.conversations import list_conversations
    c = conv(None)
    c.created_at = c.updated_at = __import__("datetime").datetime(2026, 10, 6)
    db = list_db([c])
    got = await list_conversations(ids=f"{c.id},not-a-uuid", limit=20, offset=0, current_user=None, db=db)
    assert [r.id for r in got] == [c.id]
    sql = where_sql(db)
    assert c.id.hex in sql.replace("-", "") and "user_id IS NULL" in sql


async def test_signed_out_list_without_ids_is_empty():
    from app.api.routes.conversations import list_conversations
    db = list_db([])
    assert await list_conversations(ids=None, limit=20, offset=0, current_user=None, db=db) == []
    db.execute.assert_not_called()


async def test_signed_in_list_is_the_users_own():
    from app.api.routes.conversations import list_conversations
    db = list_db([])
    await list_conversations(ids=None, limit=20, offset=0, current_user=SimpleNamespace(id=ALICE), db=db)
    assert ALICE.hex in where_sql(db).replace("-", "")


@pytest.mark.parametrize("owner,caller,ok", [(None, None, True), (ALICE, None, False), (ALICE, BOB, False)])
async def test_delete_access(owner, caller, ok):
    from app.api.routes.conversations import delete_conversation
    c = conv(owner)
    db = db_with(c)
    db.delete, db.commit = AsyncMock(), AsyncMock()
    user = SimpleNamespace(id=caller) if caller else None
    if ok:
        await delete_conversation(c.id, user, db)
        db.delete.assert_awaited_once_with(c)
    else:
        with pytest.raises(HTTPException):
            await delete_conversation(c.id, user, db)
        db.delete.assert_not_awaited()
