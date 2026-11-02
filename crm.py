"""Fake CRM and knowledge base on SQLite. Every record here is synthetic."""
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("crm.db")
KB_DIR = Path(__file__).parent / "kb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, address TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
    item TEXT NOT NULL, status TEXT NOT NULL, total_eur REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
    subject TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY, reason TEXT NOT NULL, summary TEXT NOT NULL, transcript TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS kb USING fts5(source UNINDEXED, title, body, tokenize='porter');
"""

CUSTOMERS = [
    ("C1001", "Deniz Ornek", "deniz@example.com", "1 Sample Street, Testville"),
    ("C1002", "Sam Placeholder", "sam@example.com", "22 Fixture Road, Mocktown"),
    ("C1003", "Mira Demo", "mira@example.com", "3 Dummy Avenue, Stubford"),
]
ORDERS = [
    ("O5001", "C1001", "Wireless headphones", "shipped", 89.90),
    ("O5002", "C1001", "USB-C cable", "processing", 12.50),
    ("O5003", "C1002", "Desk lamp", "delivered", 34.00),
    ("O5004", "C1003", "Mechanical keyboard", "processing", 120.00),
]


# Without this, "how long do refunds take" ranks on "how", "long" and "take".
STOP_WORDS = set("a an and are can do does for how i in is it long many much my of on or take the to what when where why with you your".split())


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:  # commits on success, rolls back on an exception
            yield conn
    finally:
        conn.close()


def init_db():
    with _db() as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
        conn.executemany("INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?)", ORDERS)
        conn.execute("DELETE FROM kb")
        conn.executemany("INSERT INTO kb VALUES (?, ?, ?)", _kb_sections())


def _kb_sections():
    for path in sorted(KB_DIR.glob("*.md")):
        for section in path.read_text(encoding="utf-8").split("\n## ")[1:]:
            title, _, body = section.partition("\n")
            yield f"{path.name}: {title.strip()}", title.strip(), body.strip()


def search_kb(query):
    # Quote every word so customer text cannot be read as FTS5 query syntax.
    words = re.findall(r"\w+", query)
    if not words:
        raise ValueError("Search query must contain words")
    words = [w for w in words if w.lower() not in STOP_WORDS] or words
    match = " OR ".join(f'"{w}"' for w in words)
    with _db() as conn:
        rows = conn.execute(
            "SELECT source, body FROM kb WHERE kb MATCH ? ORDER BY bm25(kb, 0, 5, 1) LIMIT 3", (match,)
        ).fetchall()
    return [{"cite": f"[{r['source']}]", "text": r["body"]} for r in rows]


def get_customer(email):
    with _db() as conn:
        row = conn.execute("SELECT * FROM customers WHERE email = ?", (email.strip().lower(),)).fetchone()
    if row is None:
        raise LookupError(f"No customer with email {email}")
    return dict(row)


def get_order(customer_id, order_id):
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?", (order_id, customer_id)
        ).fetchone()
    if row is None:
        # The same answer for "missing" and "someone else's", so ids cannot be probed.
        raise LookupError(f"Customer {customer_id} has no order {order_id}")
    return dict(row)


def check_address(address):
    # ponytail: a shape check, not real validation. A postal address API would replace it.
    street, _, city = address.partition(",")
    if not (re.search(r"\d", street) and len(street.split()) >= 2 and city.strip()):
        raise ValueError("Address is incomplete: ask the customer for the street, house number and city")


def update_address(customer_id, address):
    check_address(address)
    with _db() as conn:
        updated = conn.execute(
            "UPDATE customers SET address = ? WHERE id = ?", (address.strip(), customer_id)
        ).rowcount
    if not updated:
        raise LookupError(f"No customer with id {customer_id}")
    return {"customer_id": customer_id, "address": address.strip()}


def create_ticket(customer_id, subject, body):
    try:
        with _db() as conn:
            ticket_id = conn.execute(
                "INSERT INTO tickets (customer_id, subject, body) VALUES (?, ?, ?)",
                (customer_id, subject, body),
            ).lastrowid
    except sqlite3.IntegrityError:
        raise LookupError(f"No customer with id {customer_id}") from None
    return {"ticket_id": ticket_id}


def escalate_to_human(reason, summary, transcript):
    with _db() as conn:
        handoff_id = conn.execute(
            "INSERT INTO handoffs (reason, summary, transcript) VALUES (?, ?, ?)",
            (reason, summary, json.dumps(transcript, ensure_ascii=False)),
        ).lastrowid
    return {"handoff_id": handoff_id, "status": "queued for a human agent"}


def list_handoffs():
    with _db() as conn:
        rows = conn.execute("SELECT * FROM handoffs ORDER BY id DESC").fetchall()
    return [{**dict(r), "transcript": json.loads(r["transcript"])} for r in rows]
