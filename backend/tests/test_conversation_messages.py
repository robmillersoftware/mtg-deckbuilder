"""Messages are saved: SQLAlchemy only writes a JSONB column it sees reassigned."""

from sqlalchemy import inspect
from sqlalchemy.orm.attributes import set_committed_value

from app.models.conversation import Conversation


def test_add_message_marks_messages_changed_on_a_loaded_conversation():
    conv = Conversation()
    set_committed_value(conv, "messages", [{"role": "user", "content": "hi"}])  # as loaded from the DB
    conv.add_message("assistant", "hello")
    history = inspect(conv).attrs.messages.history
    assert history.has_changes()
    assert [m["content"] for m in conv.messages] == ["hi", "hello"]
