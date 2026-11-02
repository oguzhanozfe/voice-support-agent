"""HTTP API and demo page around the agent. Run with: uv run uvicorn app:app"""
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import crm
import voice

MAX_AUDIO_BYTES = 10_000_000

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


@app.get("/")
def page():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/handoffs")
def handoffs():
    """The queue a human agent would work from."""
    return crm.list_handoffs()


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


@app.post("/voice/{session_id}")
async def voice_turn(session_id: str, request: Request):
    """One spoken turn: the request body is the recording, the reply carries speech and timings."""
    audio = await request.body()
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"Send a recording of at most {MAX_AUDIO_BYTES} bytes")
    return await run_in_threadpool(_voice_turn, session_id, audio)


def _voice_turn(session_id, audio):
    started = time.perf_counter()
    try:
        transcript, language = voice.transcribe(audio)
    except ImportError:
        raise HTTPException(501, "Voice needs the extra: uv sync --extra voice") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if not transcript:
        raise HTTPException(422, "No speech was heard")
    heard = time.perf_counter()

    with lock:
        session = sessions.setdefault(session_id, agent.Session(session_id))
        if not session.pending:
            reply = _call_model(agent.send, session, transcript)
        else:
            answer = voice.parse_answer(transcript)
            # Anything but a clear yes or no changes nothing: the question is simply asked again.
            reply = (
                _call_model(agent.confirm, session, answer)
                if answer is not None
                else {"text": "", "pending": agent._describe(session.pending), "actions": []}
            )
    answered = time.perf_counter()

    text = reply["text"]
    if reply["pending"]:
        text = f"{text} {voice.confirmation_prompt(reply['pending'], language)}".strip()
    try:
        speech = voice.speak(text, language)
    except FileNotFoundError:
        raise HTTPException(503, "Voice files are missing, see the README") from None
    done = time.perf_counter()

    timings = {
        "stt_ms": round((heard - started) * 1000),
        "agent_ms": round((answered - heard) * 1000),
        "tts_ms": round((done - answered) * 1000),
        "total_ms": round((done - started) * 1000),
    }
    agent._log(session, "voice_turn", transcript=transcript, language=language, **timings)
    return {**reply, "text": text, "transcript": transcript, "language": language, "audio": speech, "timings": timings}
