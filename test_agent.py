"""The loop against a scripted fake model: no API key, no network, no cost."""
import json
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

import agent
import crm
import voice


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
    call("r1", "get_order", '{"customer_id": "C1001", "order_id": "O5001"}'),
    call("w1", "update_address", '{"customer_id": "C1001", "address": "9 New Street, Newtown"}'),
]


def test_write_waits_for_approval_then_runs():
    before = address()
    client = fake_client(ADDRESS_CHANGE, "Done.")
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Change my address to 9 New Street", client)
    assert reply.pending == [
        {"tool": "update_address", "input": {"customer_id": "C1001", "address": "9 New Street, Newtown"}}
    ]
    assert address() == before

    reply = agent.confirm(session, True, client)
    assert reply.text == "Done." and reply.pending == []
    assert address() == "9 New Street, Newtown"

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
    session = agent.Session("t", customer_id="C1001")

    agent.send(session, "Change my address", client)
    agent.confirm(session, False, client)

    assert address() == before
    assert "declined" in session.messages[4]["content"]


def test_tool_error_goes_back_to_the_model():
    client = fake_client([call("r1", "get_order", '{"customer_id": "C1001", "order_id": "nope"}')], "Which order?")
    session = agent.Session("t", customer_id="C1001")

    assert agent.send(session, "Where is my order?", client).text == "Which order?"
    assert session.messages[3]["content"].startswith("Error:")


def test_malformed_write_is_rejected_without_asking():
    before = address()
    client = fake_client([call("w1", "update_address", '{"customer_id": ')], "Sorry, which address?")
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Change my address", client)
    assert reply.pending == [] and address() == before
    assert session.messages[3]["content"].startswith("Error:")


def test_loop_stops_at_max_steps():
    client = fake_client(*[[call(f"r{i}", "search_kb", '{"query": "returns"}')] for i in range(agent.MAX_STEPS)])

    assert agent.send(agent.Session("t", customer_id="C1001"), "Loop forever", client).text == agent.SORRY


def test_order_of_another_customer_is_refused():
    with pytest.raises(LookupError):
        crm.get_order("C1002", "O5001")


def test_kb_search_finds_section_and_survives_query_syntax():
    assert crm.search_kb("how long do refunds take")[0]["cite"] == "[returns.md: Refund timing]"
    results = crm.search_kb('refund" OR NEAR(* -- ')
    assert any(r["cite"] == "[returns.md: Refund timing]" for r in results)


def test_handoff_stores_summary_and_our_transcript():
    client = fake_client(
        [call("w1", "escalate_to_human", '{"reason": "asked for a human", "summary": "Deniz wants a person"}')],
        "A human will take over.",
    )
    session = agent.Session("t", customer_id="C1001")
    agent.send(session, "I want a human", client)
    agent.confirm(session, True, client)

    handoff = crm.list_handoffs()[0]
    assert handoff["summary"] == "Deniz wants a person"
    assert {"role": "user", "text": "I want a human"} in handoff["transcript"]


def test_view_bounds_history_and_keeps_tool_pairs():
    messages = [{"role": "system", "content": agent.SYSTEM}]
    for i in range(20):
        messages += [
            {"role": "user", "content": f"question {i} " + "x" * 500},
            {
                "role": "assistant",
                "reasoning": "thoughts " * 100,
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "n", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": "y" * 500},
        ]
    view = agent._view(messages)

    assert len(json.dumps(view)) < len(json.dumps(messages)) / 4
    assert view[0] == messages[0] and view[1]["role"] == "user"  # the digest is never a system message
    assert not any("reasoning" in m for m in view)
    assert view[2]["content"].startswith("question 17")
    # Every tool result still follows the assistant message that asked for it.
    for i, m in enumerate(view):
        if m["role"] == "tool":
            assert view[i - 1]["tool_calls"][0]["id"] == m["tool_call_id"]
    assert len(messages) == 61  # the full history is untouched


@pytest.mark.parametrize("bad", ["9 New Street", "New Street, Newtown", "9, Newtown", "   "])
def test_incomplete_address_is_rejected_before_the_customer_is_asked(bad):
    before = address()
    arguments = json.dumps({"customer_id": "C1001", "address": bad})
    client = fake_client([call("w1", "update_address", arguments)], "What is your city?")
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Change my address", client)
    assert reply.pending == [] and reply.text == "What is your city?"
    assert address() == before
    assert "incomplete" in session.messages[3]["content"]


def test_each_write_needs_its_own_answer():
    before = address()
    client = fake_client(
        [
            call("w1", "update_address", '{"customer_id": "C1001", "address": "9 New Street, Newtown"}'),
            call("w2", "create_ticket", '{"customer_id": "C1001", "subject": "s", "body": "b"}'),
        ],
        "One done, one declined.",
    )
    session = agent.Session("t", customer_id="C1001")

    assert len(agent.send(session, "Do both", client).pending) == 2
    reply = agent.confirm(session, False, client)  # "no" to the address, the ticket still waits
    assert [p["tool"] for p in reply.pending] == ["create_ticket"] and address() == before

    reply = agent.confirm(session, True, client)
    assert reply.text == "One done, one declined."
    assert [m["tool_call_id"] for m in session.messages if m["role"] == "tool"] == ["w1", "w2"]


@pytest.mark.parametrize(
    "heard, answer",
    [("Yes.", True), ("Evet!", True), ("No", False), ("Hayır.", False), ("no problem but wait", None), ("yes no", None), ("", None)],
)
def test_spoken_answer_counts_only_when_clear(heard, answer):
    assert voice.parse_answer(heard) is answer


def test_conversation_stays_on_one_customer():
    session = agent.Session("t")
    client = fake_client(
        [call("r1", "get_order", '{"customer_id": "C1002", "order_id": "O5003"}')],  # nobody looked up yet
        [call("r2", "get_customer", '{"email": "deniz@example.com"}')],
        [call("r3", "get_customer", '{"email": "sam@example.com"}')],
        [call("w1", "update_address", '{"customer_id": "C1002", "address": "1 Hack Road, Badtown"}')],
        "Sorry, I cannot do that.",
    )

    for text in ["Order O5003?"]:
        assert agent.send(session, text, client).pending == []
    tool_output = [m["content"] for m in session.messages if m["role"] == "tool"]
    assert tool_output[0].startswith("Error: Not allowed")
    assert '"id": "C1001"' in tool_output[1]
    assert tool_output[2].startswith("Error: Not allowed") and "Fixture" not in tool_output[2]
    assert tool_output[3].startswith("Error: Not allowed")
    assert crm.get_customer("sam@example.com")["address"] == "22 Fixture Road, Mocktown"


def test_policy_answer_gets_its_source_from_code():
    client = fake_client([call("r1", "search_kb", '{"query": "refund timing"}')], "Five business days.")
    reply = agent.send(agent.Session("t"), "How long do refunds take?", client)
    assert reply.text == "Five business days.\n\n[returns.md: Refund timing]"
