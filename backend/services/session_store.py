"""
In-memory session store for pyramid indexes.

Each document upload creates a session with a unique ID.
The pyramid index is stored here and retrieved by session_id
on subsequent queries. Sessions expire after TTL_SECONDS.
"""

import time
import uuid
from typing import Dict, Any, Optional


TTL_SECONDS = 1800  # 30 minutes


class _Session:
    """A single session holding a pyramid index and metadata."""

    __slots__ = ("session_id", "pyramid_index", "chunks", "metadata", "created_at", "last_accessed")

    def __init__(self, session_id: str, pyramid_index: dict, chunks: list, metadata: dict):
        self.session_id = session_id
        self.pyramid_index = pyramid_index
        self.chunks = chunks
        self.metadata = metadata
        self.created_at = time.time()
        self.last_accessed = time.time()

    def touch(self):
        self.last_accessed = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > TTL_SECONDS


class SessionStore:
    """Thread-safe in-memory store for pyramid sessions."""

    def __init__(self):
        self._sessions: Dict[str, _Session] = {}

    def create(self, pyramid_index: dict, chunks: list, metadata: dict) -> str:
        """Store a new pyramid index and return the session ID."""
        self._cleanup_expired()
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = _Session(session_id, pyramid_index, chunks, metadata)
        return session_id

    def get(self, session_id: str) -> Optional[_Session]:
        """Retrieve a session by ID. Returns None if expired or missing."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            del self._sessions[session_id]
            return None
        session.touch()
        return session

    def _cleanup_expired(self):
        """Remove all expired sessions. Called lazily on create."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]


# Singleton instance shared across the app
store = SessionStore()
