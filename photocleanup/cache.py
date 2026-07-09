import hashlib
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "data" / "cache.db"


def _conn():
    _DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS eyes (key TEXT PRIMARY KEY, value REAL)")
    return conn


def content_key(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()  # keyed by bytes, so a re-upload to a new temp path still hits


def get_eyes(key):
    with _conn() as conn:
        row = conn.execute("SELECT value FROM eyes WHERE key = ?", (key,)).fetchone()
    return (True, row[0]) if row else (False, None)


def put_eyes(key, value):
    with _conn() as conn:
        conn.execute("INSERT OR REPLACE INTO eyes (key, value) VALUES (?, ?)", (key, value))
