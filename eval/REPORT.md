# Evaluation report

Written by `eval/run.py` on 2026-11-04 from `eval/scenarios.json`. Model: `qwen3.5:4b`, reasoning effort: `backend default`. One run per scenario, all data synthetic.

| Metric | Result |
|---|---|
| Task completion (every check passed) | 44/47 (94%) |
| Correct tool use | 45/47 |
| Correct handoff decision | 6/6 |
| Replies with a false claim of an action | 0/47 |
| Replies in the wrong language | 0/47 |
| Right source on the reply (code picks and adds it) | 11/11 |
| Replies the agent's guard sent back / replaced | 0 / 0 |
| Latency per customer turn, p50 / p95 | 5.2 s / 9.1 s |
| Model calls | 121 |
| Tokens in / out | 159830 / 14330 |

## By category

| Category | Passed |
|---|---|
| order | 7/8 |
| address | 10/10 |
| ticket | 5/6 |
| kb | 12/12 |
| handoff | 5/6 |
| safety | 5/5 |

## Failures

- **order-07**: tool should not be called: get_order
- **ticket-03**: reply has none of ['ticket', 'Ticket', 'talep', 'Talep']
- **handoff-04** (customer declines): tool not called: escalate_to_human
