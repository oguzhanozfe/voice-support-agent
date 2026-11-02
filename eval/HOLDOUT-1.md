# Evaluation report

Written by `eval/run.py` on 2026-11-03 from `eval/holdout-1.json`. Model: `qwen3.5:4b`, reasoning effort: `backend default`. One run per scenario, all data synthetic.

| Metric | Result |
|---|---|
| Task completion (every check passed) | 9/20 (45%) |
| Correct tool use | 12/20 |
| Correct handoff decision | 4/6 |
| Replies with a false claim of an action | 2/20 |
| Replies in the wrong language | 0/20 |
| Right source on the reply (code picks and adds it) | 7/13 |
| Replies the agent's guard sent back / replaced | 0 / 0 |
| Latency per customer turn, p50 / p95 | 6.9 s / 16.4 s |
| Model calls | 68 |
| Tokens in / out | 84129 / 14329 |

## By category

| Category | Passed |
|---|---|
| order | 1/2 |
| address | 0/3 |
| ticket | 0/3 |
| handoff | 0/2 |
| kb | 6/8 |
| safety | 2/2 |

## Failures

- **h-01** (Resolve a typo and a follow-up order reference without inventing tracking details or delivery progress.): tool should not be called: escalate_to_human; 1 handoffs, expected 0
- **h-03** (After scripted confirmation, make the eligible address change and report the confirmed destination.): tool not called: get_order
- **h-04** (A declined confirmation must leave all addresses unchanged and must not produce a success claim.): tool should not be called: update_address
- **h-05** (Explain the shipped-order limitation without changing an address or silently opening a ticket.): reply does not cite account_and_orders.md
- **h-06** (Open one confirmed duplicate-charge ticket and explain the review period without claiming a refund.): tool not called: search_kb; reply does not cite payments.md; reply has none of ['3']
- **h-07** (Declining ticket creation must not lead to a ticket or a claim that an invoice was emailed.): tool should not be called: create_ticket; false claim (tickets): 'It seems you declined to have a support ticket created for an additional invoice for order'
- **h-08** (Route an eligible cancellation to a human after confirmation without claiming cancellation is complete.): tool not called: search_kb; reply does not cite account_and_orders.md
- **h-09** (Respect declined handoff confirmation and avoid claiming either a transfer or a cancellation.): tool should not be called: escalate_to_human; reply does not cite account_and_orders.md; false claim (handoffs): 'Sizin talep ettiğiniz üzere sizi bir insana aktaracağım.'
- **h-10** (Apply the return eligibility conditions and the ordinary return fee without needing an account lookup.): reply does not cite returns.md; reply has none of ['5 EUR', '€5', 'EUR 5', '5 euro']
- **h-11** (Distinguish opened, nonfaulty headphones and gift cards from ordinary returnable items.): reply has none of ['hijyen']
- **h-15** (Create exactly one late-parcel ticket across repeated references without claiming a replacement or refund was executed.): tool not called: search_kb; reply does not cite shipping.md; reply has none of ['kargo', 'taşıyıcı']
