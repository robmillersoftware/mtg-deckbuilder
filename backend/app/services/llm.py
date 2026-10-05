"""
LLM access through OpenRouter's OpenAI-compatible API.

Every text-generation call in the app goes through here. The model is
settings.LLM_MODEL, so switching models is a config change.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

from app.core.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def is_configured() -> bool:
    return bool(settings.OPENROUTER_API_KEY)


def _client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.OPENROUTER_API_KEY)


def _messages(system: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return ([{"role": "system", "content": system}] if system else []) + messages


def complete(system: str, user: str, max_tokens: int = 4096) -> str:
    """One system + user turn; returns the reply text ('' if the model returned none)."""
    response = _client().chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=max_tokens,
        messages=_messages(system, [{"role": "user", "content": user}]),
    )
    return response.choices[0].message.content or ""


async def stream_text(system: str, user: str, max_tokens: int = 4096) -> AsyncIterator[str]:
    """Yield reply text as it streams."""
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.OPENROUTER_API_KEY)
    try:
        stream = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            max_tokens=max_tokens,
            messages=_messages(system, [{"role": "user", "content": user}]),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    finally:
        await client.close()


def to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic-style tool specs (name/description/input_schema) to OpenAI functions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def chat_with_tools(
    system: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    require_tool: bool = False,
    max_tokens: int = 2048,
) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
    """Returns (reply text, [(tool_name, tool_input), ...])."""
    response = _client().chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=max_tokens,
        messages=_messages(system, messages),
        tools=to_openai_tools(tools),
        tool_choice="required" if require_tool else "auto",
    )
    return parse_tool_reply(response.choices[0].message)


def parse_tool_reply(message: Any) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for call in message.tool_calls or []:
        args: Optional[Dict[str, Any]] = None
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            pass
        calls.append((call.function.name, args if isinstance(args, dict) else {}))
    return message.content or "", calls
