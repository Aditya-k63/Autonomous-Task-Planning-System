"""Tests for the context budget."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.context import trim_messages


def _conversation(pairs: int):
    messages = [SystemMessage(content="system")]
    for i in range(pairs):
        messages.append(AIMessage(content=f"call {i}"))
        messages.append(ToolMessage(content=f"result {i}", tool_call_id=f"c{i}"))
    messages.append(HumanMessage(content="go"))
    return messages


def test_short_conversation_is_untouched():
    messages = _conversation(1)
    trimmed = trim_messages(messages, keep_pairs=3)
    assert len(trimmed) == len(messages)


def test_long_conversation_is_bounded():
    trimmed = trim_messages(_conversation(20), keep_pairs=3)
    assert len(trimmed) < 20
    assert any(isinstance(m, SystemMessage) for m in trimmed)


def test_system_message_is_always_kept():
    trimmed = trim_messages(_conversation(20), keep_pairs=1)
    assert trimmed[0].content == "system"


def test_recent_tool_results_survive():
    trimmed = trim_messages(_conversation(20), keep_pairs=3)
    contents = [m.content for m in trimmed if isinstance(m, ToolMessage)]
    assert contents
    assert contents[-1] == "result 19"


def test_dropped_exchanges_are_summarised():
    trimmed = trim_messages(_conversation(20), keep_pairs=2)
    assert any("were dropped" in str(m.content) for m in trimmed)


def test_empty_list_is_safe():
    assert trim_messages([]) == []
