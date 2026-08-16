"""
SQLite Database Layer for PalmPay Biometric Templates.
Stores user identity metadata and serialized normalized feature vectors.
Raw palm biometric images are NOT stored to protect user privacy.
"""

from datetime import datetime, timezone
import os
import sqlite3
from typing import Dict, List, Optional
import numpy as np

import config


class Database:
    """
    Thread-safe SQLite database manager for PalmPay.
    """

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with Row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def init_db(self):
        """Initializes tables for users and biometric templates."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    usn TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    sample_index INTEGER NOT NULL,
                    feature_blob BLOB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_templates_user_id ON templates(user_id);
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def serialize_vector(vector: np.ndarray) -> bytes:
        """Serializes a float32 numpy array into raw binary bytes."""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_vector(blob: bytes, dim: int = config.FEATURE_VECTOR_DIM) -> np.ndarray:
        """Deserializes raw binary bytes back into a float32 numpy array."""
        arr = np.frombuffer(blob, dtype=np.float32)
        if len(arr) != dim:
            arr = arr[:dim]
        return arr

    def add_user(self, name: str, usn: str) -> Optional[int]:
        """
        Creates a new user record.

        Args:
            name: Full name of user.
            usn: Unique University Seat Number / Student ID.

        Returns:
            Inserted user_id if successful, None if USN already exists.
        """
        name = name.strip()
        usn = usn.strip().upper()
        if not name or not usn:
            return None

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, usn, created_at) VALUES (?, ?, ?)",
                (name, usn, self._now_iso()),
            )
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def add_templates(
        self,
        user_id: int,
        embeddings: List[np.ndarray],
        model_name: str = config.MODEL_NAME,
    ) -> bool:
        """
        Stores a batch of sample feature embeddings for an enrolled user.

        Args:
            user_id: ID of the user.
            embeddings: List of 1D float32 numpy arrays.
            model_name: Model identifier string.

        Returns:
            True if all templates were successfully stored.
        """
        if not embeddings:
            return False

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            for idx, emb in enumerate(embeddings):
                blob = self.serialize_vector(emb)
                cursor.execute(
                    """
                    INSERT INTO templates (user_id, sample_index, feature_blob, feature_dim, model_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        idx + 1,
                        blob,
                        len(emb),
                        model_name,
                        self._now_iso(),
                    ),
                )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_all_users_with_templates(self) -> List[Dict]:
        """
        Retrieves all enrolled users along with their deserialized template embeddings.
        """
        users_map = {}
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, usn, created_at FROM users ORDER BY id ASC")
            for row in cursor.fetchall():
                users_map[row["id"]] = {
                    "id": row["id"],
                    "name": row["name"],
                    "usn": row["usn"],
                    "created_at": row["created_at"],
                    "templates": [],
                    "sample_count": 0,
                }

            cursor.execute(
                "SELECT user_id, sample_index, feature_blob, feature_dim, model_name FROM templates ORDER BY user_id, sample_index ASC"
            )
            for row in cursor.fetchall():
                uid = row["user_id"]
                if uid in users_map:
                    vec = self.deserialize_vector(row["feature_blob"], row["feature_dim"])
                    users_map[uid]["templates"].append(vec)
                    users_map[uid]["sample_count"] += 1
        finally:
            conn.close()

        return list(users_map.values())

    def get_user_by_usn(self, usn: str) -> Optional[Dict]:
        """Looks up a user by USN."""
        usn = usn.strip().upper()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, usn, created_at FROM users WHERE usn = ?", (usn,))
            row = cursor.fetchone()
            if row:
                return {"id": row["id"], "name": row["name"], "usn": row["usn"], "created_at": row["created_at"]}
            return None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Looks up a user by ID."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, usn, created_at FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {"id": row["id"], "name": row["name"], "usn": row["usn"], "created_at": row["created_at"]}
            return None
        finally:
            conn.close()

    def delete_user(self, user_id: int) -> bool:
        """Deletes a user and all their stored templates."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM templates WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """Returns summary statistics for the database."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as user_count FROM users")
            user_count = cursor.fetchone()["user_count"]
            cursor.execute("SELECT COUNT(*) as template_count FROM templates")
            template_count = cursor.fetchone()["template_count"]
            return {
                "total_users": user_count,
                "total_templates": template_count,
                "database_path": self.db_path,
            }
        finally:
            conn.close()
