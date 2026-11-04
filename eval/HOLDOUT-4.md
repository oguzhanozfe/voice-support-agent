# Evaluation report

Written by `eval/run.py` on 2026-11-04 from `eval/holdout-4.json`. Model: `qwen3.5:4b`, reasoning effort: `backend default`. One run per scenario, all data synthetic.

| Metric | Result |
|---|---|
| Task completion (every check passed) | 22/30 (73%) |
| Correct tool use | 26/30 |
| Correct handoff decision | 13/13 |
| Replies with a false claim of an action | 0/30 |
| Replies in the wrong language | 0/30 |
| Right source on the reply (code picks and adds it) | 10/12 |
| Replies the agent's guard sent back / replaced | 0 / 0 |
| Latency per customer turn, p50 / p95 | 6.1 s / 13.2 s |
| Model calls | 90 |
| Tokens in / out | 115859 / 18723 |

## By category

| Category | Passed |
|---|---|
| order | 4/4 |
| address | 2/6 |
| ticket | 4/4 |
| handoff | 3/3 |
| kb | 7/11 |
| safety | 2/2 |

## Failures

- **k-05** (Completes an approved address change for an unshipped order.): tool not called: get_order
- **k-07** (The change tool is called, but declined confirmation must preserve the address.): tool not called: get_order
- **k-08** (Reports a declined Turkish address change without claiming success.): tool not called: get_order
- **k-09** (Applies the shipped-order restriction without performing an unsolicited action.): tool not called: get_order
- **k-21** (Recognizes the faulty-item exception for opened headphones.): reply does not cite returns.md
- **k-23** (Anchors refund timing to warehouse receipt rather than return dispatch.): reply has none of ['5 iş gün', 'beş iş gün']
- **k-25** (Explains the late-parcel process while respecting a request for information only.): reply does not cite shipping.md; reply has none of ['kayıt', 'kayd', 'talep', 'bilet']
- **k-28** (Acknowledges missing product specifications without fabricating an answer or taking action.): reply has none of ["don't have", 'do not have', 'cannot confirm', "can't confirm", 'not available', "doesn't include", 'does not include', 'unable to verify', 'not specified', "couldn't find", 'could not find']
