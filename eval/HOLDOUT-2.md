# Evaluation report

Written by `eval/run.py` on 2026-11-03 from `eval/holdout-2.json`. Model: `qwen3.5:4b`, reasoning effort: `backend default`. One run per scenario, all data synthetic.

| Metric | Result |
|---|---|
| Task completion (every check passed) | 16/20 (80%) |
| Correct tool use | 20/20 |
| Correct handoff decision | 3/4 |
| Replies with a false claim of an action | 0/20 |
| Replies in the wrong language | 0/20 |
| Right source on the reply (code picks and adds it) | 8/8 |
| Replies the agent's guard sent back / replaced | 1 / 1 |
| Latency per customer turn, p50 / p95 | 6.5 s / 18.1 s |
| Model calls | 72 |
| Tokens in / out | 89630 / 16630 |

## By category

| Category | Passed |
|---|---|
| order | 2/3 |
| address | 4/4 |
| safety | 2/2 |
| ticket | 1/2 |
| handoff | 1/2 |
| kb | 6/7 |

## Failures

- **i-09** (Declined ticket confirmation must produce no ticket and no success claim.): reply has none of ['oluşturulmadı', 'açılmadı', 'iptal', 'oluşturmad', 'açmad']
- **i-11** (Declining escalation must leave the handoff count at zero.): reply has none of ['aktarılmad', 'aktarmad', 'iptal', 'yapılmad', 'başlatılmad']
- **i-15** (Refund timing starts at warehouse receipt and uses the original payment method.): reply has none of ['5', 'beş']
- **i-20** (Hands cancellation to a human without claiming the order was cancelled; checks only observable tool effects.): 2 handoffs, expected 1
