"""Speech in and speech out. Needs the `voice` extra; models load on first use."""
import base64
import io
import os
import re
import wave
from functools import lru_cache
from pathlib import Path

VOICE_DIR = Path("voices")
VOICES = {"en": "en_US-lessac-medium", "tr": "tr_TR-dfki-medium"}
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

YES = {"yes", "yeah", "yep", "confirm", "evet", "onayla", "onaylıyorum"}
NO = {"no", "nope", "cancel", "hayır", "iptal"}

PROMPTS = {
    "en": {
        "update_address": "change your address to {address}",
        "create_ticket": "open a support ticket: {subject}",
        "escalate_to_human": "hand you over to a human agent",
        "ask": "I am about to {action}. Please say yes or no.",
    },
    "tr": {
        "update_address": "adresinizi {address} olarak değiştirmek",
        "create_ticket": "destek talebi açmak: {subject}",
        "escalate_to_human": "sizi bir müşteri temsilcisine aktarmak",
        "ask": "Şu işlemi yapmak üzereyim: {action}. Lütfen evet veya hayır deyin.",
    },
}


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


@lru_cache(maxsize=len(VOICES))
def _voice(name):
    from piper import PiperVoice

    return PiperVoice.load(VOICE_DIR / f"{name}.onnx")


def _decode(audio):
    """Any recording to 16 kHz mono floats. Done here because faster-whisper's own decoder breaks on current PyAV."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    try:
        with av.open(io.BytesIO(audio)) as container:
            frames = [f for frame in container.decode(audio=0) for f in resampler.resample(frame)]
    except (av.FFmpegError, IndexError):
        raise ValueError("The recording could not be read") from None
    if not frames:
        raise ValueError("The recording is empty")
    return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32) / 32768


def transcribe(audio):
    segments, info = _whisper().transcribe(_decode(audio))
    return " ".join(s.text.strip() for s in segments).strip(), info.language


def speak(text, language):
    # Source tags and markdown read badly aloud.
    spoken = re.sub(r"\[[^\]]*\]|[*_`#]", "", text)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        _voice(VOICES.get(language, VOICES["en"])).synthesize_wav(spoken, wav)
    return base64.b64encode(buffer.getvalue()).decode()


def parse_answer(transcript):
    """True, False or None. Only a clear answer counts: "no problem, but wait" is not a yes."""
    words = re.sub(r"[^\w\s]", "", transcript.lower()).split()
    if len(words) == 1 and words[0] in YES:
        return True
    if len(words) == 1 and words[0] in NO:
        return False
    return None


def confirmation_prompt(pending, language):
    prompts = PROMPTS.get(language, PROMPTS["en"])
    call = pending[0]
    return prompts["ask"].format(action=prompts[call["tool"]].format_map(_Blank(call["input"])))


class _Blank(dict):
    def __missing__(self, key):
        return ""
