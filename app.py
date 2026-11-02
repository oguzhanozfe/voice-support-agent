"""HTTP API around the agent. Run with: uv run uvicorn app:app"""
import os
import threading
from dataclasses import asdict

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import agent
import crm

app = FastAPI(title="Support agent")
# Any OpenAI-compatible endpoint works; the default is a local Ollama, which ignores the key.
client = openai.OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "unused"),
    timeout=120,
)
crm.init_db()

# ponytail: in-memory sessions behind one lock, lost on restart and one request at a time.
# Enough for a local demo; move to per-session locks and SQLite if this ever serves real traffic.
sessions: dict[str, agent.Session] = {}
lock = threading.Lock()


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=2000)  # long inputs cost money on paid backends


class Confirmation(BaseModel):
    approved: bool


def _call_model(call, *args):
    try:
        return asdict(call(*args, client))
    except openai.APIError as e:
        raise HTTPException(502, f"Model API error: {type(e).__name__}") from None


@app.post("/chat/{session_id}")
def chat(session_id: str, body: Message):
    with lock:
        session = sessions.setdefault(session_id, agent.Session(session_id))
        if session.pending:
            raise HTTPException(409, "Answer the pending confirmation first")
        return _call_model(agent.send, session, body.message)


@app.post("/chat/{session_id}/confirm")
def confirm(session_id: str, body: Confirmation):
    with lock:
        session = sessions.get(session_id)
        if session is None or not session.pending:
            raise HTTPException(409, "Nothing is waiting for confirmation")
        return _call_model(agent.confirm, session, body.approved)
