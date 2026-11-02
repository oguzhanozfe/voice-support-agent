"""Support agent: a hand-written tool-calling loop with a human gate on writes."""
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import crm

MODEL = os.environ.get("LLM_MODEL", "qwen3.5:4b")
MAX_STEPS = 8  # model calls per run, so a confused model cannot spend without limit
LOG_PATH = Path("logs/steps.jsonl")
SORRY = "Sorry, I could not complete that. Please try again or ask for a human agent."

SYSTEM = """You are the customer support agent of Example Shop, an online electronics store.
Answer in the customer's language. Use the tools for facts about customers and orders; never guess them.
Look a customer up by email before acting on their account, and share an order only with its owner.
When you have what a change needs, call the tool directly: the system shows the customer the exact
change and asks them to confirm it, so do not ask for that confirmation yourself.
If a tool says the customer declined, accept that and ask what they would like instead.
Escalate to a human when the customer asks for one or when you cannot resolve the issue."""


def _tool(name, description, **params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS = [
    _tool("get_customer", "Find a customer by email address.", email="Customer email address"),
    _tool("get_order", "Get an order's item, status, total and owner.", order_id="Order id, e.g. O5001"),
    _tool(
        "update_address",
        "Change a customer's address.",
        customer_id="Customer id, e.g. C1001",
        address="The full new address",
    ),
    _tool(
        "create_ticket",
        "Open a support ticket for a customer.",
        customer_id="Customer id, e.g. C1001",
        subject="One-line summary",
        body="What happened and what the customer wants",
    ),
    _tool(
        "escalate_to_human",
        "Hand the conversation to a human agent.",
        reason="Why a human is needed, with the customer id if known",
    ),
]
READ_TOOLS = {"get_customer": crm.get_customer, "get_order": crm.get_order}
WRITE_TOOLS = {
    "update_address": crm.update_address,
    "create_ticket": crm.create_ticket,
    "escalate_to_human": crm.escalate_to_human,
}


@dataclass
class Session:
    id: str
    messages: list = field(default_factory=lambda: [{"role": "system", "content": SYSTEM}])
    pending: list = field(default_factory=list)  # write calls waiting for the customer's answer
    run_id: str = ""  # one run = one customer message until the agent's reply
    step: int = 0


@dataclass
class Reply:
    text: str
    pending: list = field(default_factory=list)  # non-empty means: ask for confirmation


def send(session, text, client):
    """Handle a customer message. The caller must first resolve any pending confirmation."""
    session.run_id, session.step = uuid.uuid4().hex[:8], 0
    _log(session, "customer_message", text=text)
    session.messages.append({"role": "user", "content": text})
    return _run(session, client)


def confirm(session, approved, client):
    """Run or decline the pending writes, then let the model continue."""
    _log(session, "confirmation", approved=approved, tools=[c.function.name for c in session.pending])
    for call in session.pending:
        if approved:
            _execute(session, call)
        else:
            _decline(session, call)
    session.pending = []
    return _run(session, client)


def _run(session, client):
    for _ in range(MAX_STEPS):
        # 1. Ask the model what to do next, given the whole conversation so far.
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL, max_tokens=4096, tools=TOOLS, messages=session.messages
        )
        choice = response.choices[0]
        message = choice.message
        _log(
            session,
            "model_call",
            model=response.model,
            finish_reason=choice.finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            ms=round((time.perf_counter() - started) * 1000),
        )

        # Output that was cut off or filtered cannot be replayed: it may be empty or
        # hold a half-written tool call. Keep it out of the history.
        if choice.finish_reason not in ("stop", "tool_calls") or not (message.content or message.tool_calls):
            return Reply(SORRY)

        # 2. Keep the full message: tool results must point at its tool call ids.
        session.messages.append(message.model_dump(exclude_none=True))
        text = message.content or ""

        # 3. No tool call means the model has answered the customer.
        if not message.tool_calls:
            _log(session, "agent_reply", text=text)
            return Reply(text)

        # 4. Run reads now, hold writes. The gate is here in code, so the model
        # cannot talk its way past it. Malformed arguments fail in _execute, before any write.
        for call in message.tool_calls:
            if call.function.name in WRITE_TOOLS and _is_json_object(call.function.arguments):
                session.pending.append(call)
            else:
                _execute(session, call)

        # 5. Stop and wait for the human. The loop resumes in confirm().
        if session.pending:
            pending = [{"tool": c.function.name, "input": json.loads(c.function.arguments)} for c in session.pending]
            _log(session, "awaiting_confirmation", pending=pending)
            return Reply(text, pending)

        # 6. The results are in the history now, so go around again.

    _log(session, "max_steps_reached")
    return Reply(SORRY)


def _is_json_object(arguments):
    try:
        return isinstance(json.loads(arguments), dict)
    except ValueError:
        return False


def _execute(session, call):
    name, arguments = call.function.name, call.function.arguments
    try:
        output = {**READ_TOOLS, **WRITE_TOOLS}[name](**json.loads(arguments))
        content, is_error = json.dumps(output, ensure_ascii=False), False
    except (LookupError, ValueError, TypeError) as e:
        # The model reads the error and can recover, e.g. by asking for the right id.
        content, is_error = f"Error: {e}", True
    _log(session, "tool_call", tool=name, input=arguments, output=content, is_error=is_error)
    session.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})


def _decline(session, call):
    _log(session, "tool_declined", tool=call.function.name, input=call.function.arguments)
    content = "The customer declined this action. It was not performed."
    session.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})


def _log(session, event, **fields):
    session.step += 1
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "session": session.id,
        "run": session.run_id,
        "step": session.step,
        "event": event,
        **fields,
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
