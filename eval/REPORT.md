# Evaluation report

Written by `eval/run.py` on 2026-11-03. Model: `qwen3.5:4b`, reasoning effort: `backend default`. One run per scenario, all data synthetic.

| Metric | Result |
|---|---|
| Task completion (every check passed) | 45/47 (96%) |
| Correct tool use | 46/47 |
| Correct handoff decision | 6/6 |
| Reply cites the right source | 10/11 |
| Latency per customer turn, p50 / p95 | 4.4 s / 8.0 s |
| Model calls | 118 |
| Tokens in / out | 129387 / 14554 |

## By category

| Category | Passed |
|---|---|
| order | 8/8 |
| address | 10/10 |
| ticket | 5/6 |
| kb | 11/12 |
| handoff | 6/6 |
| safety | 5/5 |

## Failures

- **ticket-03**: tool not called: create_ticket; 0 tickets, expected 1
- **kb-10**: reply does not cite returns.md
