import hashlib
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "data" / "cache.db"


def _conn():
    _DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS faces (key TEXT PRIMARY KEY, eyes REAL, face_sharp REAL)")
    return conn


def content_key(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()  # keyed by bytes, so a re-upload to a new temp path still hits


def get_faces(key):
    with _conn() as conn:
        return conn.execute("SELECT eyes, face_sharp FROM faces WHERE key = ?", (key,)).fetchone()


def put_faces(key, eyes, face_sharp):
    with _conn() as conn:
        conn.execute("INSERT OR REPLACE INTO faces (key, eyes, face_sharp) VALUES (?, ?, ?)", (key, eyes, face_sharp))
