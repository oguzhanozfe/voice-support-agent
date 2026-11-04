"""Run the scenarios against a real model and write what happened, pass or fail.

Usage: uv run python eval/run.py [--scenarios FILE] [--name NAME] [--only PREFIX]
Each scenario gets its own temporary database. Approvals are scripted per scenario.
Besides each scenario's own checks, every reply is checked for language and for
claims of actions that did not happen. These word lists are separate from agent.py on
purpose, so the agent's own guard does not grade itself.
"""
import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openai  # noqa: E402

import agent  # noqa: E402
import crm  # noqa: E402
import knowledge  # noqa: E402


TR_HINTS = set("ve bir için ile lütfen adresiniz siparişiniz size sizi başka yardımcı olabilir mi mı var yok gün iş".split())
EN_HINTS = set("the your is are you for with can please order has have and of to".split())
CLAIM_PATTERNS = {
    "tickets": r"ticket.{0,60}(open|creat|raise|file|submit|log)|(open|creat|rais|filed|submit|logged).{0,60}ticket|talep.{0,60}(aç|oluştur|kayd)",
    "handoffs": r"(forward|hand|transfer|escalat|connect|pass|submit).{0,60}(human|agent|representative|team)|(temsilci|insan).{0,60}(aktar|ilet|bağla|gönder)",
    "address": r"address.{0,60}(update|chang|set to|now)|(updat|chang).{0,60}address|adres.{0,60}(güncel|değiş|kaydet|kaydedil)|address.{0,60}saved",
}
NEGATED = r"^\s*[-*•]|[ai]lemez|[ae]mez|[ae]maz|\bcannot\b|[ae]bilirim|\brules?\b|\bpolic|declin|reddet|\bhappy\b|\bhelp you\b|\bnot\b|n't|\bunable\b|\bno\b|\bcan\b|\bcould\b|\bwould\b|\bwill\b|\bif\b|\bneed\b|\bwant\b|\blike\b|\bbefore\b|\bto\s+(open|update|change|create|escalate)|\?|değil|edilmedi|yapılmadı|ister|için|lütfen|gerekli"


def reply_language(text):
    text = re.sub(r"\[[^\]]*\]|\S+@\S+", " ", text)
    words = set(re.findall(r"\w+", text.lower()))
    tr = len(words & TR_HINTS) + 3 * len(re.findall(r"[çğıöşü]", text.lower()))
    en = len(words & EN_HINTS)
    return "tr" if tr > en else "en" if en > tr else ""


def false_claims(text, called_ok):
    found = []
    for kind, pattern in CLAIM_PATTERNS.items():
        if called_ok[kind]:
            continue
        # Source tags such as [account.md: Changing a delivery address] are not claims.
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\[[^\]]*\]", "", text)):
            lowered = sentence.lower()
            if re.search(pattern, lowered) and not re.search(NEGATED, lowered):
                found.append(f"false claim ({kind}): {sentence.strip()[:90]!r}")
    return found


def run_scenario(scenario, client, workdir):
    crm.DB_PATH = workdir / f"{scenario['id']}.db"
    agent.LOG_PATH = workdir / f"{scenario['id']}.jsonl"
    crm.init_db()
    knowledge.init()
    before = {c[0]: c[3] for c in crm.CUSTOMERS}
    session = agent.Session(scenario["id"])
    replies, turn_seconds, error = [], [], None
    try:
        for text in scenario["turns"]:
            started = time.perf_counter()
            reply = agent.send(session, text, client)
            while reply.pending:
                reply = agent.confirm(session, scenario.get("approve", True), client)
            turn_seconds.append(time.perf_counter() - started)
            replies.append(reply.text)
    except openai.APIError as e:
        error = type(e).__name__

    called = [
        c["function"]["name"] for m in session.messages if m["role"] == "assistant" for c in m.get("tool_calls", [])
    ]
    failures = _check(scenario, called, "\n".join(replies), before)
    if error:
        failures.append(f"API error: {error}")
    events = [json.loads(line) for line in agent.LOG_PATH.read_text().splitlines()]
    # What reached the customer or the database counts. A write the guard rejected reached neither.
    shown = [json.dumps(e, ensure_ascii=False) for e in events if e["event"] in ("awaiting_confirmation", "tool_call")]
    for word in scenario.get("address_never", []):
        if any(word in e and "update_address" in e for e in shown):
            failures.append(f"{word!r} was offered or written as part of the new address")
    ok = {e["tool"] for e in events if e["event"] == "tool_call" and not e["output"].startswith("Error")}
    called_ok = {
        "tickets": "create_ticket" in ok,
        "handoffs": "escalate_to_human" in ok,
        "address": "update_address" in ok,
    }
    last = replies[-1] if replies else ""
    claims = [c for r in replies for c in false_claims(r, called_ok)]
    failures += claims
    language = reply_language(last)
    wrong_language = bool(language and language != scenario.get("lang", "en"))
    if wrong_language:
        failures.append(f"reply is in {language}, expected {scenario.get('lang', 'en')}")
    model_calls = [e for e in events if e["event"] == "model_call"]
    return {
        "id": scenario["id"],
        "category": scenario["category"],
        "note": scenario.get("note", ""),
        "passed": not failures,
        "failures": failures,
        "tools_ok": not any(f.startswith("tool") for f in failures),
        "false_claims": len(claims),
        "wrong_language": wrong_language,
        "corrections": sum(e["event"] == "reply_rejected" for e in events),
        "replaced": sum(e["event"] == "reply_replaced" for e in events),
        "called": called,
        "reply": "\n".join(replies),
        "turn_seconds": turn_seconds,
        "model_calls": len(model_calls),
        "input_tokens": sum(e["input_tokens"] for e in model_calls),
        "output_tokens": sum(e["output_tokens"] for e in model_calls),
    }


def _check(s, called, text, before):
    failures = []
    for tool in s.get("tools", []):
        if tool not in called:
            failures.append(f"tool not called: {tool}")
    for tool in s.get("forbid", []):
        if tool in called:
            failures.append(f"tool should not be called: {tool}")
    for customer_id, expected in s.get("address", {}).items():
        actual = _address(customer_id)
        if actual != expected:
            failures.append(f"address is {actual!r}, expected {expected!r}")
    for customer_id in s.get("address_unchanged", []):
        if _address(customer_id) != before[customer_id]:
            failures.append(f"address changed to {_address(customer_id)!r}")
    if "tickets" in s and _count("tickets") != s["tickets"]:
        failures.append(f"{_count('tickets')} tickets, expected {s['tickets']}")
    if "handoffs" in s and _count("handoffs") != s["handoffs"]:
        failures.append(f"{_count('handoffs')} handoffs, expected {s['handoffs']}")
    if "cites" in s and s["cites"] not in text:
        failures.append(f"reply does not cite {s['cites']}")
    if "text_any" in s and not any(t in text for t in s["text_any"]):
        failures.append(f"reply has none of {s['text_any']}")
    for t in s.get("text_none", []):
        if t in text:
            failures.append(f"reply contains {t!r}")
    return failures


def _address(customer_id):
    with crm._db() as conn:
        return conn.execute("SELECT address FROM customers WHERE id = ?", (customer_id,)).fetchone()[0]


def _count(table):
    with crm._db() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _percentile(values, p):
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def report(results, scenarios, source):
    by_id = {s["id"]: s for s in scenarios}
    passed = sum(r["passed"] for r in results)
    turns = [t for r in results for t in r["turn_seconds"]]
    handoff = [r for r in results if "handoffs" in by_id[r["id"]]]
    handoff_ok = sum(not any("handoffs" in f for f in r["failures"]) for r in handoff)
    cited = [r for r in results if "cites" in by_id[r["id"]]]
    cited_ok = sum(not any("cite" in f for f in r["failures"]) for r in cited)

    lines = [
        "# Evaluation report",
        "",
        f"Written by `eval/run.py` on {date.today()} from `{source}`. Model: `{agent.MODEL}`, "
        f"reasoning effort: `{agent.REASONING_EFFORT or 'backend default'}`. One run per scenario, all data synthetic.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Task completion (every check passed) | {passed}/{len(results)} ({passed / len(results):.0%}) |",
        f"| Correct tool use | {sum(r['tools_ok'] for r in results)}/{len(results)} |",
        f"| Correct handoff decision | {handoff_ok}/{len(handoff)} |",
        f"| Replies with a false claim of an action | {sum(r['false_claims'] > 0 for r in results)}/{len(results)} |",
        f"| Replies in the wrong language | {sum(r['wrong_language'] for r in results)}/{len(results)} |",
        f"| Right source on the reply (code picks and adds it) | {cited_ok}/{len(cited)} |",
        f"| Replies the agent's guard sent back / replaced | {sum(r['corrections'] for r in results)} / {sum(r['replaced'] for r in results)} |",
        f"| Latency per customer turn, p50 / p95 | {_percentile(turns, 50):.1f} s / {_percentile(turns, 95):.1f} s |",
        f"| Model calls | {sum(r['model_calls'] for r in results)} |",
        f"| Tokens in / out | {sum(r['input_tokens'] for r in results)} / {sum(r['output_tokens'] for r in results)} |",
        "",
        "## By category",
        "",
        "| Category | Passed |",
        "|---|---|",
    ]
    groups = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)
    lines += [f"| {c} | {sum(r['passed'] for r in rs)}/{len(rs)} |" for c, rs in groups.items()]
    lines += ["", "## Failures", ""]
    failed = [r for r in results if not r["passed"]]
    if not failed:
        lines.append("None.")
    for r in failed:
        note = f" ({r['note']})" if r["note"] else ""
        lines.append(f"- **{r['id']}**{note}: {'; '.join(r['failures'])}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="eval/scenarios.json")
    parser.add_argument("--name", default="REPORT")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    client = openai.OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key=os.environ.get("LLM_API_KEY", "unused"),
        timeout=180,
    )
    scenarios = json.loads((ROOT / args.scenarios).read_text(encoding="utf-8"))
    scenarios = [s for s in scenarios if s["id"].startswith(args.only)]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in scenarios:
            results.append(run_scenario(s, client, Path(tmp)))
            print(("pass" if results[-1]["passed"] else "FAIL"), s["id"], "; ".join(results[-1]["failures"]), flush=True)
    out = ROOT / "eval"
    (out / f"{args.name}.results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{args.name}.md").write_text(report(results, scenarios, args.scenarios), encoding="utf-8")
    print(f"{sum(r['passed'] for r in results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
