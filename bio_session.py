"""
Server-side Biometric Authentication Session Store (Demo / Prototype).

After /api/recognize produces a successful 1:N match, the matched identity is
stored HERE on the server — keyed by an opaque random token held in the
browser's Flask session cookie. The frontend never sends user_id; payment
requests resolve the payer strictly from this store within a short TTL window.
"""

import secrets
import threading
import time
from typing import Dict, Optional

import config


class BiometricSessionStore:
    """
    Thread-safe in-memory store of fresh biometric recognitions.

    Each entry: token -> {user_id, name, usn, score, created_at}
    Entries expire after BIOMETRIC_SESSION_TTL_SECONDS (30s by default).
    """

    def __init__(self, ttl_seconds: int = config.BIOMETRIC_SESSION_TTL_SECONDS):
        self.ttl_seconds = int(ttl_seconds)
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now() -> float:
        return time.time()

    def create(self, user_id: int, name: str, usn: str, score: float) -> str:
        """Registers a fresh biometric match and returns its opaque session token."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {
                "user_id": int(user_id),
                "name": name,
                "usn": usn,
                "score": float(score),
                "created_at": self._now(),
            }
            self._purge_expired_locked()
        return token

    def get(self, token: Optional[str], consume: bool = False) -> Optional[Dict]:
        """
        Returns the biometric session payload if the token exists AND is fresh,
        otherwise None. With consume=True a successful read burns the token so a
        recognition can authorize exactly one payment.
        """
        if not token:
            return None
        with self._lock:
            entry = self._sessions.get(token)
            if entry is None:
                return None
            if self._now() - entry["created_at"] > self.ttl_seconds:
                del self._sessions[token]
                return None
            if consume:
                del self._sessions[token]
            return dict(entry)

    def invalidate(self, token: Optional[str]) -> None:
        """Removes a token (e.g. after a failed re-recognition attempt)."""
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _purge_expired_locked(self) -> None:
        now = self._now()
        expired = [t for t, s in self._sessions.items() if now - s["created_at"] > self.ttl_seconds]
        for t in expired:
            del self._sessions[t]

    def purge_expired(self) -> None:
        with self._lock:
            self._purge_expired_locked()
