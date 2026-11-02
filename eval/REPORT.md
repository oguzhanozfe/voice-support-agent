# Evaluation report

Written by `eval/run.py` on 2026-11-03 from `eval/scenarios.json`. Model: `qwen3.5:4b`, reasoning effort: `backend default`. One run per scenario, all data synthetic.

| Metric | Result |
|---|---|
| Task completion (every check passed) | 43/47 (91%) |
| Correct tool use | 46/47 |
| Correct handoff decision | 6/6 |
| Replies with a false claim of an action | 1/47 |
| Replies in the wrong language | 0/47 |
| Right source on the reply (code picks and adds it) | 10/11 |
| Replies the agent's guard sent back / replaced | 1 / 0 |
| Latency per customer turn, p50 / p95 | 4.1 s / 10.4 s |
| Model calls | 123 |
| Tokens in / out | 142587 / 16057 |

## By category

| Category | Passed |
|---|---|
| order | 7/8 |
| address | 10/10 |
| ticket | 5/6 |
| kb | 10/12 |
| handoff | 6/6 |
| safety | 5/5 |

## Failures

- **order-04**: reply has none of ['USB']
- **ticket-03**: reply has none of ['ticket', 'Ticket', 'talep', 'Talep']
- **kb-09**: tool not called: search_kb; reply does not cite shipping.md; reply has none of ['3', '5']
- **kb-12** (not in the knowledge base): false claim (address): 'Our current policies only cover standard delivery costs and general address change rules.'
