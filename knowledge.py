"""Hybrid search over kb/: SQLite FTS5 keywords and embedding vectors, fused by rank."""
import json
import math
import os
import re
from pathlib import Path

import openai

import crm

KB_DIR = Path(__file__).parent / "kb"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
RRF_K = 60  # standard reciprocal rank fusion constant

# Without this, "how long do refunds take" ranks on "how", "long" and "take".
STOP_WORDS = set("a an and are can do does for how i in is it long many much my of on or take the to what when where why with you your".split())

# The documents are English. A customer's Turkish words reach them through this short map.
# ponytail: a hand list for this store's topics. Translated documents would replace it.
TR_TERMS = {
    "iade": "return refund", "iadeyi": "return refund", "iadesi": "refund return", "param": "refund", "kripto": "cryptocurrency payment", "kapıda": "cash delivery payment", "kargo": "shipping delivery",
    "paket": "parcel delivery", "teslimat": "delivery", "garanti": "warranty", "garantisi": "warranty", "fatura": "invoice",
    "faturamın": "invoice", "ödeme": "payment", "iptal": "cancel", "temsilci": "human agents", "hasar": "damage",
    "hasarı": "water damage", "arızalı": "faulty", "bozuk": "faulty", "açılmış": "opened", "kulaklık": "headphones",
    "kulaklığın": "opened headphones", "ücret": "cost fee", "ücreti": "cost fee", "ücretsiz": "free", "süresi": "window timing",
    "gönderileli": "late parcel", "kayıp": "lost parcel", "gecikti": "late parcel",
}

_client = openai.OpenAI(
    base_url=os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.environ.get("EMBED_API_KEY", "unused"),
    timeout=20,
    max_retries=0,
)
_cache: dict[str, list[float]] = {}
_available = True


def _embed(texts):
    """Vectors for texts, or None if no embedding model answers: search then falls back to keywords."""
    global _available
    missing = [t for t in texts if t not in _cache]
    if missing and _available:
        try:
            response = _client.embeddings.create(model=EMBED_MODEL, input=missing)
            _cache.update(zip(missing, (d.embedding for d in response.data)))
        except openai.APIError:
            _available = False  # asked once per process, so tests and offline runs stay fast
    if any(t not in _cache for t in texts):
        return None
    return [_cache[t] for t in texts]


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


def _sections():
    for path in sorted(KB_DIR.glob("*.md")):
        for section in path.read_text(encoding="utf-8").split("\n## ")[1:]:
            title, _, body = section.partition("\n")
            yield f"{path.name}: {title.strip()}", title.strip(), body.strip()


def init():
    sections = list(_sections())
    vectors = _embed([f"{title}\n{body}" for _, title, body in sections])
    with crm._db() as conn:
        conn.execute("DELETE FROM kb")
        conn.executemany("INSERT INTO kb VALUES (?, ?, ?)", sections)
        conn.execute("DELETE FROM kb_vectors")
        if vectors:
            conn.executemany(
                "INSERT INTO kb_vectors VALUES (?, ?)",
                [(s[0], json.dumps(v)) for s, v in zip(sections, vectors)],
            )


def _turkish_words(text):
    return re.findall(r"\w+", text.replace("İ", "i").replace("I", "ı").lower())


def english_terms(text):
    return " ".join(TR_TERMS.get(word, "") for word in _turkish_words(text))


def search_kb(query, customer_text=""):
    """Top sections for the model's query, also ranked by what the customer actually said."""
    # Quote every word so customer text cannot be read as FTS5 query syntax.
    said = _turkish_words(customer_text)
    words = re.findall(r"\w+", query) + said + english_terms(customer_text).split()
    if not words:
        raise ValueError("Search query must contain words")
    words = [w for w in words if w.lower() not in STOP_WORDS] or words
    match = " OR ".join(f'"{w}"' for w in words)
    with crm._db() as conn:
        keyword = [r["source"] for r in conn.execute(
            "SELECT source FROM kb WHERE kb MATCH ? ORDER BY bm25(kb, 0, 5, 1) LIMIT 5", (match,)
        )]
        stored = conn.execute("SELECT source, vector FROM kb_vectors").fetchall()
        bodies = {r["source"]: r["body"] for r in conn.execute("SELECT source, body FROM kb")}

    rankings = [keyword]
    queries = [q for q in (query, customer_text) if q.strip()]
    vectors = _embed(queries) if stored else None
    for vector in vectors or []:
        scored = [(_cosine(vector, json.loads(r["vector"])), r["source"]) for r in stored]
        rankings.append([source for _, source in sorted(scored, reverse=True)[:5]])

    # A section ranked well by several lists rises most.
    scores = {}
    for ranking in rankings:
        for rank, source in enumerate(ranking):
            scores[source] = scores.get(source, 0) + 1 / (RRF_K + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:4]
    return [{"cite": f"[{source}]", "text": bodies[source]} for source in best]
