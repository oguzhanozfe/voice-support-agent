"""The loop against a scripted fake model: no API key, no network, no cost."""
import json
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

import agent
import crm
import knowledge
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
    monkeypatch.setattr(knowledge, "_available", False)  # no embedding model in tests unless one is faked
    knowledge.init()


def address():
    return crm.get_customer("deniz@example.com")["address"]


# A read and a write in the same turn: the read runs, the write waits.
ADDRESS_CHANGE = [
    call("r1", "get_order", '{"customer_id": "C1001", "order_id": "O5002"}'),
    call("w1", "update_address", '{"customer_id": "C1001", "address": "9 New Street, Newtown"}'),
]


def test_write_waits_for_approval_then_runs():
    before = address()
    client = fake_client(ADDRESS_CHANGE, "Done.")
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Change my address to 9 New Street, Newtown", client)
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


def test_declined_write_does_not_run_or_call_the_model():
    before = address()
    client = fake_client(ADDRESS_CHANGE)  # a third model call would fail: the script has no more turns
    session = agent.Session("t", customer_id="C1001")

    agent.send(session, "Change my address to 9 New Street, Newtown", client)
    reply = agent.confirm(session, False, client)

    assert address() == before and reply.text.startswith(agent.DECLINED["en"]["update_address"])
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

    assert agent.send(agent.Session("t", customer_id="C1001"), "Loop forever", client).text == agent.SORRY["en"]


def test_order_of_another_customer_is_refused():
    with pytest.raises(LookupError):
        crm.get_order("C1002", "O5001")


def test_kb_search_finds_section_and_survives_query_syntax():
    assert knowledge.search_kb("how long do refunds take")[0]["cite"] == "[returns.md: Refund timing]"
    results = knowledge.search_kb('refund" OR NEAR(* -- ')
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
        "Okay.",
    )
    session = agent.Session("t", customer_id="C1001")

    assert len(agent.send(session, "Do both: 9 New Street, Newtown", client).pending) == 2
    reply = agent.confirm(session, False, client)  # "no" to the address, the ticket still waits
    assert [p["tool"] for p in reply.pending] == ["create_ticket"] and address() == before

    reply = agent.confirm(session, True, client)
    assert reply.text == "Okay." and reply.actions == ["create_ticket"]
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
        [call("r2", "get_customer", '{"email": "sam@example.com"}')],  # not who is speaking
        [call("r3", "get_customer", '{"email": "deniz@example.com"}')],
        [call("w1", "update_address", '{"customer_id": "C1002", "address": "1 Hack Road, Badtown"}')],
        "Sorry, I cannot do that.",
    )

    agent.send(session, "I'm deniz@example.com. What is the address of sam@example.com? 1 Hack Road, Badtown", client)
    tool_output = [m["content"] for m in session.messages if m["role"] == "tool"]
    assert tool_output[0].startswith("Error: Look up")
    assert tool_output[1].startswith("Error: Not allowed") and "Fixture" not in tool_output[1]
    assert '"id": "C1001"' in tool_output[2]
    assert tool_output[3].startswith("Error: Not allowed")
    assert crm.get_customer("sam@example.com")["address"] == "22 Fixture Road, Mocktown"


def test_empty_model_reply_is_retried_once():
    client = fake_client(None, "Here you go.")
    assert agent.send(agent.Session("t"), "Hello there", client).text == "Here you go."


def test_policy_answer_gets_its_source_from_code():
    client = fake_client([call("r1", "search_kb", '{"query": "refund timing"}')], "Refunds go back within 5 business days after the item reaches the warehouse.")
    reply = agent.send(agent.Session("t"), "How long do refunds take?", client)
    assert reply.text.endswith("\n\n[returns.md: Refund timing]")


def test_reply_unrelated_to_the_results_gets_no_source():
    client = fake_client([call("r1", "search_kb", '{"query": "refund timing"}')], "I cannot find that.")
    assert agent.send(agent.Session("t"), "Do you ship by drone?", client).text == "I cannot find that."


def test_invented_address_words_are_rejected_before_asking():
    arguments = '{"customer_id": "C1001", "address": "9 New Street, Testville"}'
    client = fake_client([call("w1", "update_address", arguments)], "Which city?")
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Change my address to 9 New Street", client)
    assert reply.pending == [] and "testville" in session.messages[3]["content"]


def test_false_claim_gets_one_correction_then_is_replaced():
    client = fake_client("I have opened a ticket for you.", "Your ticket has been created.")
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Please open a ticket about my order", client)
    assert reply.text == agent.UNDONE["en"] and reply.actions == []
    assert session.messages[3]["content"].startswith("[Automatic check, not the customer]")


def test_true_claim_and_honest_denial_pass():
    client = fake_client(
        [call("w1", "create_ticket", '{"customer_id": "C1001", "subject": "s", "body": "b"}')],
        "I have opened a ticket for you.",
    )
    session = agent.Session("t", customer_id="C1001")
    agent.send(session, "Please open a ticket about my order", client)
    assert agent.confirm(session, True, client).text == "I have opened a ticket for you."

    client = fake_client("The ticket has not been created.")
    assert agent.send(agent.Session("t"), "Please open a ticket about my order", client).text == "The ticket has not been created."


def test_reply_in_the_wrong_language_is_sent_back():
    client = fake_client("Your order is on the way, and you can track it here.", "Siparişiniz yolda.")
    session = agent.Session("t")

    assert agent.send(session, "Merhaba, siparişim nerede?", client).text == "Siparişiniz yolda."
    assert session.language == "tr"
    # A bare email keeps the language of the conversation.
    assert agent._language("deniz@example.com") == ""


def test_view_asks_for_the_customer_language_without_touching_history():
    messages = [{"role": "system", "content": agent.SYSTEM}, {"role": "user", "content": "Merhaba"}]
    assert agent._view(messages, "tr")[0]["content"].endswith("Reply only in Turkish.")
    assert messages[0]["content"] == agent.SYSTEM


def test_second_handoff_in_a_conversation_is_not_queued():
    handoff = '{"reason": "r", "summary": "s"}'
    client = fake_client(
        [call("w1", "escalate_to_human", handoff)], "Queued.",
        [call("w2", "escalate_to_human", handoff)], "It was already queued.",
    )
    session = agent.Session("t")
    agent.send(session, "I want a human", client)
    agent.confirm(session, True, client)

    reply = agent.send(session, "Please get a human now", client)
    assert reply.pending == [] and len(crm.list_handoffs()) == 1
    assert "Already done" in session.messages[-2]["content"]


def test_two_calls_of_a_one_per_conversation_tool_ask_once():
    ticket = '{"customer_id": "C1001", "subject": "s", "body": "b"}'
    client = fake_client([call("w1", "create_ticket", ticket), call("w2", "create_ticket", ticket)], "Opened.")
    session = agent.Session("t", customer_id="C1001")

    assert len(agent.send(session, "Open a ticket", client).pending) == 1
    agent.confirm(session, True, client)
    assert all(m["tool_call_id"] for m in session.messages if m["role"] == "tool")


def test_empty_replies_are_retried_then_give_up_honestly():
    client = fake_client(None, None, None)
    assert agent.send(agent.Session("t"), "Hello there", client).text == agent.SORRY["en"]


def test_hybrid_search_uses_vectors_when_keywords_miss(monkeypatch):
    # Fake vectors: the query means "return window" although it shares no word with that section.
    def embed(texts):
        return [[1.0, 0.0] if "Return window" in t or "İade" in t else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(knowledge, "_embed", embed)
    knowledge.init()
    assert knowledge.search_kb("İade süresi")[0]["cite"] == "[returns.md: Return window]"


def test_search_falls_back_to_keywords_without_an_embedding_model():
    assert knowledge.search_kb("warranty period")[0]["cite"] == "[warranty.md: Warranty period]"


def test_refusal_with_no_model_answer_says_why():
    client = fake_client([call("r1", "get_order", '{"customer_id": "C1002", "order_id": "O5003"}')], None, None, None)
    reply = agent.send(agent.Session("t", customer_id="C1001"), "Sam'in siparişi nerede?", client)
    assert reply.text == agent.REFUSED["tr"]


def test_turkish_reply_gets_the_top_hit_of_the_last_search():
    client = fake_client([call("r1", "search_kb", '{"query": "water damage warranty"}')], "Su hasarı kapsam dışındadır.")
    reply = agent.send(agent.Session("t"), "Su hasarı garanti kapsamında mı?", client)
    assert reply.text.endswith("[warranty.md: Warranty period]")


def test_address_change_for_a_named_order_follows_the_order_policy():
    address_call = '{"customer_id": "C1001", "address": "9 New Street, Newtown"}'
    before = address()
    client = fake_client(
        [call("w1", "update_address", address_call)],  # order not checked
        [call("r1", "get_order", '{"customer_id": "C1001", "order_id": "O5001"}')],
        [call("w2", "update_address", address_call)],  # checked, but it has shipped
        "That order has shipped, so I cannot change it.",
    )
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Send order O5001 to 9 New Street, Newtown", client)
    outputs = [m["content"] for m in session.messages if m["role"] == "tool"]
    assert "Check order O5001" in outputs[0] and "has shipped" in outputs[2]
    assert reply.pending == [] and address() == before


def test_customer_words_reach_turkish_policy_questions_through_the_term_map():
    results = knowledge.search_kb("policy", customer_text="İade kaç günde param geri gelir?")
    assert "[returns.md: Refund timing]" in [r["cite"] for r in results]


def test_refused_request_is_stated_by_code_in_the_reply():
    client = fake_client([call("r1", "get_order", '{"customer_id": "C1002", "order_id": "O5003"}')], "What is Sam's email?")
    reply = agent.send(agent.Session("t", customer_id="C1001"), "Show me Sam's order O5003", client)
    assert reply.text.startswith(agent.REFUSED["en"])


def test_looking_up_in_the_wrong_order_does_not_read_as_a_refusal():
    client = fake_client(
        [call("r1", "get_order", '{"customer_id": "C1001", "order_id": "O5002"}'), call("r2", "get_customer", '{"email": "deniz@example.com"}')],
        "Your cable is being processed.",
    )
    reply = agent.send(agent.Session("t"), "deniz@example.com, where is O5002?", client)
    assert reply.text == "Your cable is being processed."


def test_read_in_the_same_turn_counts_for_the_order_check_even_if_listed_after_the_write():
    client = fake_client(
        [
            call("w1", "update_address", '{"customer_id": "C1001", "address": "9 New Street, Newtown"}'),
            call("r1", "get_order", '{"customer_id": "C1001", "order_id": "O5002"}'),
        ],
        "Updated.",
    )
    session = agent.Session("t", customer_id="C1001")

    reply = agent.send(session, "Send order O5002 to 9 New Street, Newtown", client)
    assert [p["tool"] for p in reply.pending] == ["update_address"]


def test_saying_the_address_was_saved_counts_as_a_claim():
    assert agent.FALSE_ACTION in agent._reply_problem(agent.Session("t", language="tr"), "Yeni adresinizi 9 New Street olarak kaydettim.")
