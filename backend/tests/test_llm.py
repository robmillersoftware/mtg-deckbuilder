"""OpenRouter adapter: tool-schema conversion and tool-call parsing."""

from types import SimpleNamespace

from app.services import llm


def call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_to_openai_tools():
    tools = [{"name": "suggest_core", "description": "Suggest cards",
              "input_schema": {"type": "object", "properties": {"strategy": {"type": "string"}}}}]
    assert llm.to_openai_tools(tools) == [{
        "type": "function",
        "function": {
            "name": "suggest_core",
            "description": "Suggest cards",
            "parameters": {"type": "object", "properties": {"strategy": {"type": "string"}}},
        },
    }]


def test_parse_tool_reply_text_and_calls():
    msg = SimpleNamespace(content="Here you go", tool_calls=[call("suggest_core", '{"strategy": "burn"}')])
    assert llm.parse_tool_reply(msg) == ("Here you go", [("suggest_core", {"strategy": "burn"})])


def test_parse_tool_reply_no_calls_and_null_content():
    assert llm.parse_tool_reply(SimpleNamespace(content=None, tool_calls=None)) == ("", [])


def test_parse_tool_reply_bad_or_non_object_arguments_become_empty():
    msg = SimpleNamespace(content="", tool_calls=[call("a", "{not json"), call("b", "[1, 2]")])
    assert llm.parse_tool_reply(msg) == ("", [("a", {}), ("b", {})])


def test_calls_disable_reasoning(monkeypatch):
    seen = []

    class FakeCompletions:
        def create(self, **kwargs):
            seen.append(kwargs)
            msg = SimpleNamespace(content="ok", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    assert llm.complete("sys", "hi") == "ok"
    assert llm.chat_with_tools("sys", [{"role": "user", "content": "hi"}], []) == ("ok", [])
    assert all(k["extra_body"] == llm.NO_REASONING for k in seen)
    assert seen[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_empty_system_prompt_is_omitted():
    assert llm._messages("", [{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]
