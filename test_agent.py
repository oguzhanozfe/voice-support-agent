"""The loop against a scripted fake model: no API key, no network, no cost."""
import json
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

import agent
import crm


def call(id, name, arguments):
    return ChatCompletionMessageToolCall(
        id=id, type="function", function={"name": name, "arguments": arguments}
    )


def fake_client(*turns):
    """Each turn is what the model says next: a reply string or a list of tool calls."""
    remaining = list(turns)

    def create(**_):
        turn = remaining.pop(0)
        calls = turn if isinstance(turn, list) else None
        message = ChatCompletionMessage(
            role="assistant", content=None if calls else turn, tool_calls=calls
        )
        return NS(
            model="fake",
            choices=[NS(message=message, finish_reason="tool_calls" if calls else "stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1),
        )

    return NS(chat=NS(completions=NS(create=create)))


@pytest.fixture(autouse=True)
def temp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(crm, "DB_PATH", tmp_path / "crm.db")
    monkeypatch.setattr(agent, "LOG_PATH", tmp_path / "steps.jsonl")
    crm.init_db()


def address():
    return crm.get_customer("deniz@example.com")["address"]


# A read and a write in the same turn: the read runs, the write waits.
ADDRESS_CHANGE = [
    call("r1", "get_order", '{"order_id": "O5001"}'),
    call("w1", "update_address", '{"customer_id": "C1001", "address": "9 New Street"}'),
]


def test_write_waits_for_approval_then_runs():
    before = address()
    client = fake_client(ADDRESS_CHANGE, "Done.")
    session = agent.Session("t")

    reply = agent.send(session, "Change my address to 9 New Street", client)
    assert reply.pending == [
        {"tool": "update_address", "input": {"customer_id": "C1001", "address": "9 New Street"}}
    ]
    assert address() == before

    reply = agent.confirm(session, True, client)
    assert reply.text == "Done." and reply.pending == []
    assert address() == "9 New Street"

    # Every tool call gets a result, matched by id, right after the assistant message.
    assert [m["role"] for m in session.messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert [m["tool_call_id"] for m in session.messages[3:5]] == ["r1", "w1"]

    events = [json.loads(line)["event"] for line in agent.LOG_PATH.read_text().splitlines()]
    assert events == [
        "customer_message",
        "model_call",
        "tool_call",
        "awaiting_confirmation",
        "confirmation",
        "tool_call",
        "model_call",
        "agent_reply",
    ]


def test_declined_write_does_not_run():
    before = address()
    client = fake_client(ADDRESS_CHANGE, "Okay, I left it unchanged.")
    session = agent.Session("t")

    agent.send(session, "Change my address", client)
    agent.confirm(session, False, client)

    assert address() == before
    assert "declined" in session.messages[4]["content"]


def test_tool_error_goes_back_to_the_model():
    client = fake_client([call("r1", "get_order", '{"order_id": "nope"}')], "Which order?")
    session = agent.Session("t")

    assert agent.send(session, "Where is my order?", client).text == "Which order?"
    assert session.messages[3]["content"].startswith("Error:")


def test_malformed_write_is_rejected_without_asking():
    before = address()
    client = fake_client([call("w1", "update_address", '{"customer_id": ')], "Sorry, which address?")
    session = agent.Session("t")

    reply = agent.send(session, "Change my address", client)
    assert reply.pending == [] and address() == before
    assert session.messages[3]["content"].startswith("Error:")


def test_loop_stops_at_max_steps():
    client = fake_client(*[[call(f"r{i}", "get_order", '{"order_id": "O5001"}')] for i in range(agent.MAX_STEPS)])

    assert agent.send(agent.Session("t"), "Loop forever", client).text == agent.SORRY
