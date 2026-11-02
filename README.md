# voice-support-agent

A customer support agent with a hand-written tool-calling loop, no agent framework. It looks up customers and orders in a fake CRM and changes data only after the customer confirms.

All data is synthetic. There is no authentication and sessions live in memory, so run it locally only.

Text chat only for now. Knowledge base search, handoff with context, evals and voice are not built yet.

## Run

You need [uv](https://docs.astral.sh/uv/) and any OpenAI-compatible model that supports tool calling. The default is a local [Ollama](https://ollama.com):

```bash
ollama pull qwen3.5:4b
uv run uvicorn app:app
```

To point it at another OpenAI-compatible endpoint, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in the environment before starting. So far it has only been run against the local default.

## Try it

```bash
curl -s localhost:8000/chat/demo -H 'content-type: application/json' \
  -d '{"message": "My email is deniz@example.com. Change my address to 9 New Street."}'
```

The reply lists the change in `pending` and nothing is written yet. Answer it:

```bash
curl -s localhost:8000/chat/demo/confirm -H 'content-type: application/json' \
  -d '{"approved": true}'
```

Every step of a run is appended to `logs/steps.jsonl`, with tokens and latency for each model call.

## How it works

`agent.py` holds the loop:

1. Send the conversation and the tool list to the model.
2. If the reply has no tool calls, it is the answer.
3. Run read tools and put their results in the history.
4. Hold write tools (`update_address`, `create_ticket`, `escalate_to_human`) and return them to the caller. `confirm()` runs or declines them and resumes the loop.

The gate is in code, so a model cannot skip it. `MAX_STEPS` limits model calls per run.

## Test

No key or network needed; the model is a scripted fake:

```bash
uv run pytest
```
