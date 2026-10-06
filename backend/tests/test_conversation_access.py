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
