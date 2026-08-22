"""
SQLite Database Layer for PalmPay Biometric Templates.
Stores user identity metadata and serialized normalized feature vectors.
Raw palm biometric images are NOT stored to protect user privacy.
"""

from datetime import datetime, timedelta, timezone
import os
import sqlite3
import uuid
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
            # ---- Payment System (Demo / Prototype) Schema Additions ----
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('CONSUMER', 'MERCHANT')),
                    balance REAL NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                """
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_wallets_user_id ON wallets(user_id);
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    sender_user_id INTEGER,
                    merchant_id INTEGER,
                    merchant_name TEXT,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('SUCCESS', 'FAILED_BALANCE', 'FAILED_AUTH', 'FAILED_AMOUNT')),
                    biometric_score REAL,
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sender_user_id) REFERENCES users (id),
                    FOREIGN KEY (merchant_id) REFERENCES users (id)
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_txn_sender ON transactions(sender_user_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transactions(merchant_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_txn_created ON transactions(created_at);"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cashout_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (merchant_id) REFERENCES users (id)
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS topup_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
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
                # Demo merchant accounts are payment-layer entities, not biometric enrollees.
                if str(row["usn"]).startswith(config.MERCHANT_USN_PREFIX):
                    continue
                users_map[row["id"]] = {
                    "id": row["id"],
                    "name": row["name"],
                    "usn": row["usn"],
                    "created_at": row["created_at"],
                    "templates": [],
                    "sample_count": 0,
                }

            cursor.execute(
                """
                SELECT user_id, sample_index, feature_blob, feature_dim, model_name FROM templates
                WHERE feature_dim = ? AND model_name = ?
                ORDER BY user_id, sample_index ASC
                """,
                (config.FEATURE_VECTOR_DIM, config.MODEL_NAME),
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
        """Deletes a user and all their stored templates plus payment-layer records."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE sender_user_id = ? OR merchant_id = ?", (user_id, user_id))
            cursor.execute("DELETE FROM cashout_requests WHERE merchant_id = ?", (user_id,))
            cursor.execute("DELETE FROM topup_requests WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM wallets WHERE user_id = ?", (user_id,))
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

    # ---------------------------------------------------------
    # Payment System (Demo / Prototype) Data Layer
    # ---------------------------------------------------------

    @staticmethod
    def new_transaction_id() -> str:
        """Generates a UUID4 string used as the public transaction id."""
        return str(uuid.uuid4())

    def ensure_wallet(self, user_id: int, role: str) -> Dict:
        """
        Creates a wallet for a user if it does not already exist.
        Role must be CONSUMER or MERCHANT. Returns the wallet row dict.
        """
        existing = self.get_wallet(user_id)
        if existing:
            return existing

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO wallets (user_id, role, balance, created_at, updated_at) VALUES (?, ?, 0.0, ?, ?)",
                (user_id, role, self._now_iso(), self._now_iso()),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_wallet(user_id)

    def get_wallet(self, user_id) -> Optional[Dict]:
        """Returns the wallet row for a user, or None."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, role, balance, created_at, updated_at FROM wallets WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "role": row["role"],
                "balance": float(row["balance"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def credit_wallet(self, user_id: int, amount: float) -> bool:
        """Adds amount to a user's wallet balance atomically."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE wallets SET balance = balance + ?, updated_at = ? WHERE user_id = ?",
                (float(amount), self._now_iso(), user_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def debit_wallet(self, user_id: int, amount: float) -> bool:
        """
        Subtracts amount from a user's wallet only if sufficient balance exists.
        Returns True when the debit was applied.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE wallets SET balance = balance - ?, updated_at = ? WHERE user_id = ? AND balance >= ?",
                (float(amount), self._now_iso(), user_id, float(amount)),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_or_create_merchant_account(self, merchant_config: Dict) -> Dict:
        """
        Resolves a hardcoded config merchant entry ({id, name, category}) to a
        wallet-backed users row so foreign keys stay meaningful. Idempotent.
        Returns {"merchant_id": <config id>, "name", "category", "user_id"}.
        """
        usn = f"{config.MERCHANT_USN_PREFIX}{str(merchant_config['id']).upper()}"
        user = self.get_user_by_usn(usn)
        if not user:
            uid = self.add_user(merchant_config["name"], usn)
            if uid is None:
                # Race or duplicate — re-fetch
                user = self.get_user_by_usn(usn)
                uid = user["id"] if user else None
            if uid:
                self.ensure_wallet(uid, "MERCHANT")
            user = self.get_user_by_usn(usn)
        else:
            self.ensure_wallet(user["id"], "MERCHANT")

        return {
            "merchant_id": merchant_config["id"],
            "name": merchant_config["name"],
            "category": merchant_config.get("category", ""),
            "user_id": user["id"] if user else None,
        }

    def seed_merchant_accounts(self) -> List[Dict]:
        """Ensures every configured demo merchant has a users row + MERCHANT wallet."""
        accounts = []
        for m in config.MERCHANTS:
            accounts.append(self.get_or_create_merchant_account(m))
        return accounts

    # ---- Transaction primitives (Demo / Prototype) ----

    def add_transaction(
        self,
        amount: float,
        status: str,
        sender_user_id: Optional[int] = None,
        merchant_id: Optional[int] = None,
        merchant_name: Optional[str] = None,
        biometric_score: Optional[float] = None,
        note: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> str:
        """Inserts a transaction row and returns its UUID id."""
        txn_id = transaction_id or self.new_transaction_id()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transactions (id, sender_user_id, merchant_id, merchant_name, amount, status, biometric_score, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    txn_id,
                    sender_user_id,
                    merchant_id,
                    merchant_name,
                    float(amount),
                    status,
                    biometric_score,
                    note,
                    self._now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return txn_id

    @staticmethod
    def _txn_row_to_dict(row) -> Dict:
        """Maps a joined transactions row to a plain dict with derived type."""
        note = row["note"] or ""
        if note == "TOPUP":
            txn_type = "TOPUP"
        elif note.startswith("CASHOUT"):
            txn_type = "CASHOUT"
        else:
            txn_type = "PAYMENT"
        return {
            "id": row["id"],
            "sender_user_id": row["sender_user_id"],
            "merchant_id": row["merchant_id"],
            "merchant_name": row["merchant_name"],
            "amount": float(row["amount"]),
            "status": row["status"],
            "biometric_score": row["biometric_score"],
            "note": note,
            "type": txn_type,
            "created_at": row["created_at"],
        }

    def get_transaction(self, txn_id: str) -> Optional[Dict]:
        """Fetches a single transaction (with consumer name) by UUID id."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT t.*, u.name AS sender_name, u.usn AS sender_usn
                FROM transactions t
                LEFT JOIN users u ON u.id = t.sender_user_id
                WHERE t.id = ?
                """,
                (txn_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            txn = self._txn_row_to_dict(row)
            txn["sender_name"] = row["sender_name"]
            txn["sender_usn"] = row["sender_usn"]
            return txn
        finally:
            conn.close()

    def list_transactions(
        self,
        user_id=None,
        role: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Filtered + paginated transaction list.
        role='consumer' filters on sender_user_id=user_id; role='merchant' on merchant_id=user_id.
        Returns {"transactions": [...], "total": int, "limit": int, "offset": int}.
        """
        where_clauses = []
        params: List = []

        if user_id is not None and role == "consumer":
            where_clauses.append("t.sender_user_id = ?")
            params.append(user_id)
        elif user_id is not None and role == "merchant":
            where_clauses.append("t.merchant_id = ?")
            params.append(user_id)

        if status:
            where_clauses.append("t.status = ?")
            params.append(status)

        if date_from:
            where_clauses.append("t.created_at >= ?")
            params.append(f"{date_from}T00:00:00")

        if date_to:
            # Inclusive upper bound: advance one day and compare exclusively.
            try:
                end_dt = datetime.fromisoformat(date_to) + timedelta(days=1)
                upper_bound = end_dt.date().isoformat()
            except ValueError:
                upper_bound = date_to
            where_clauses.append("t.created_at < ?")
            params.append(upper_bound)

        if types:
            type_clauses = []
            for txn_type in types:
                if txn_type == "TOPUP":
                    type_clauses.append("t.note = 'TOPUP'")
                elif txn_type == "CASHOUT":
                    type_clauses.append("t.note LIKE 'CASHOUT%'")
                else:  # PAYMENT
                    type_clauses.append(
                        "(t.note IS NULL OR (t.note != 'TOPUP' AND t.note NOT LIKE 'CASHOUT%'))"
                    )
            where_clauses.append("(" + " OR ".join(type_clauses) + ")")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            count_sql = f"SELECT COUNT(*) AS total FROM transactions t {where_sql}"
            cursor.execute(count_sql, params)
            total = cursor.fetchone()["total"]

            limit = max(1, min(int(limit), 200))
            offset = max(0, int(offset))

            data_sql = f"""
                SELECT t.*, u.name AS sender_name, u.usn AS sender_usn
                FROM transactions t
                LEFT JOIN users u ON u.id = t.sender_user_id
                {where_sql}
                ORDER BY t.created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = cursor.execute(data_sql, params + [limit, offset]).fetchall()
        finally:
            conn.close()

        txns = []
        for row in rows:
            txn = self._txn_row_to_dict(row)
            txn["sender_name"] = row["sender_name"]
            txn["sender_usn"] = row["sender_usn"]
            txns.append(txn)

        return {"transactions": txns, "total": total, "limit": limit, "offset": offset}

    def get_consumer_stats(self, user_id: int) -> Dict:
        """Quick stats for the consumer dashboard."""
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COALESCE(SUM(amount), 0.0) AS spent
                FROM transactions
                WHERE sender_user_id = ? AND status = 'SUCCESS' AND note != 'TOPUP'
                  AND created_at >= ?
                """,
                (user_id, month_start),
            )
            spent_month = float(cursor.fetchone()["spent"])

            cursor.execute("SELECT COUNT(*) AS c FROM transactions WHERE sender_user_id = ?", (user_id,))
            txn_count = int(cursor.fetchone()["c"])

            cursor.execute(
                """
                SELECT merchant_name, amount, created_at FROM transactions
                WHERE sender_user_id = ? AND status = 'SUCCESS' AND note != 'TOPUP'
                ORDER BY created_at DESC LIMIT 1
                """,
                (user_id,),
            )
            last_row = cursor.fetchone()
        finally:
            conn.close()

        last_payment = None
        if last_row:
            last_payment = {
                "merchant_name": last_row["merchant_name"],
                "amount": float(last_row["amount"]),
                "created_at": last_row["created_at"],
            }

        return {
            "spent_this_month": round(spent_month, 2),
            "transaction_count": txn_count,
            "last_payment": last_payment,
        }

    def get_merchant_stats(self, merchant_user_id: int) -> Dict:
        """
        Revenue stats for the merchant dashboard:
        today, this week (7 days incl today), all-time txn count, 7-day daily chart series.
        """
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = day_start - timedelta(days=6)

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COALESCE(SUM(amount), 0.0) AS rev FROM transactions
                WHERE merchant_id = ? AND status = 'SUCCESS' AND note != 'TOPUP' AND created_at >= ?
                """,
                (merchant_user_id, day_start.isoformat()),
            )
            today_revenue = float(cursor.fetchone()["rev"])

            cursor.execute(
                """
                SELECT COALESCE(SUM(amount), 0.0) AS rev FROM transactions
                WHERE merchant_id = ? AND status = 'SUCCESS' AND note != 'TOPUP' AND created_at >= ?
                """,
                (merchant_user_id, week_start.isoformat()),
            )
            week_revenue = float(cursor.fetchone()["rev"])

            cursor.execute(
                "SELECT COUNT(*) AS c FROM transactions WHERE merchant_id = ?",
                (merchant_user_id,),
            )
            total_txn_count = int(cursor.fetchone()["c"])

            chart_rows = cursor.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, SUM(amount) AS total
                FROM transactions
                WHERE merchant_id = ? AND status = 'SUCCESS' AND note != 'TOPUP'
                  AND created_at >= ?
                GROUP BY day
                """,
                (merchant_user_id, week_start.isoformat()),
            ).fetchall()
        finally:
            conn.close()

        totals_by_day = {r["day"]: float(r["total"]) for r in chart_rows}
        daily_chart_data = []
        for offset_days in range(6, -1, -1):
            d = (day_start - timedelta(days=offset_days)).date()
            daily_chart_data.append(
                {
                    "date": d.isoformat(),
                    "label": d.strftime("%a"),
                    "amount": round(totals_by_day.get(d.isoformat(), 0.0), 2),
                }
            )

        return {
            "today_revenue": round(today_revenue, 2),
            "week_revenue": round(week_revenue, 2),
            "total_txn_count": total_txn_count,
            "daily_chart_data": daily_chart_data,
        }

    # ---- Cash out requests (logged only — no real transfer) ----

    def create_cashout_request(self, merchant_user_id: int, amount: float) -> int:
        """Creates a PENDING cashout request for a merchant wallet owner."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cashout_requests (merchant_id, amount, status, created_at) VALUES (?, ?, 'PENDING', ?)",
                (merchant_user_id, float(amount), self._now_iso()),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def list_cashout_requests(self, status: Optional[str] = None, merchant_user_id: Optional[int] = None) -> List[Dict]:
        """Cashout requests with merchant names, newest first."""
        query = """
            SELECT c.*, u.name AS merchant_name
            FROM cashout_requests c
            LEFT JOIN users u ON u.id = c.merchant_id
        """
        clauses, params = [], []
        if status:
            clauses.append("c.status = ?")
            params.append(status)
        if merchant_user_id is not None:
            clauses.append("c.merchant_id = ?")
            params.append(merchant_user_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY c.created_at DESC"

        conn = self.get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": r["id"],
                "merchant_id": r["merchant_id"],
                "merchant_name": r["merchant_name"],
                "amount": float(r["amount"]),
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def approve_cashout_request(self, request_id: int) -> Dict:
        """
        Approves a PENDING cashout: debits the merchant wallet atomically and logs a CASHOUT transaction.
        No real transfer occurs (demo only).
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cashout_requests WHERE id = ?", (request_id,))
            req = cursor.fetchone()
            if not req:
                return {"success": False, "message": "Cashout request not found."}
            if req["status"] != "PENDING":
                return {"success": False, "message": f"Request #{request_id} is already {req['status']}."}

            cursor.execute("SELECT balance FROM wallets WHERE user_id = ?", (req["merchant_id"],))
            wallet = cursor.fetchone()
            balance = float(wallet["balance"]) if wallet else 0.0
            amount = float(req["amount"])
            if balance < amount:
                return {
                    "success": False,
                    "message": f"Insufficient merchant balance (₹{balance:.2f}) to cash out ₹{amount:.2f}.",
                }

            cursor.execute(
                "UPDATE wallets SET balance = balance - ?, updated_at = ? WHERE user_id = ?",
                (amount, self._now_iso(), req["merchant_id"]),
            )
            cursor.execute(
                "UPDATE cashout_requests SET status = 'APPROVED' WHERE id = ?",
                (request_id,),
            )

            txn_id = self.new_transaction_id()
            # Cashout ledger row: money leaves the merchant wallet (simulated).
            # merchant_id is also set so the row appears in the merchant's own history.
            cursor.execute(
                """
                INSERT INTO transactions (id, sender_user_id, merchant_id, merchant_name, amount, status, biometric_score, note, created_at)
                VALUES (?, ?, ?, NULL, ?, 'SUCCESS', NULL, 'CASHOUT', ?)
                """,
                (txn_id, req["merchant_id"], req["merchant_id"], amount, self._now_iso()),
            )
            new_balance = balance - amount
            conn.commit()
        except Exception as exc:
            conn.rollback()
            return {"success": False, "message": f"Cashout approval failed: {exc}"}
        finally:
            conn.close()

        return {"success": True, "message": "Cashout approved (simulated).", "new_balance": round(new_balance, 2)}

    def get_all_wallets_with_users(self) -> List[Dict]:
        """All wallets joined with user identity (admin views)."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT w.id AS wallet_id, w.user_id, w.role, w.balance, w.created_at AS wallet_created_at,
                       u.name, u.usn, u.created_at AS user_created_at
                FROM wallets w
                JOIN users u ON u.id = w.user_id
                ORDER BY CASE w.role WHEN 'MERCHANT' THEN 0 ELSE 1 END, u.name ASC
                """
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "wallet_id": r["wallet_id"],
                "user_id": r["user_id"],
                "role": r["role"],
                "balance": float(r["balance"]),
                "name": r["name"],
                "usn": r["usn"],
                "created_at": r["user_created_at"],
            }
            for r in rows
        ]

    def get_payment_system_stats(self) -> Dict:
        """System-wide transaction stats for the admin dashboard."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS c FROM transactions")
            total_transactions = int(cursor.fetchone()["c"])
            cursor.execute("SELECT COALESCE(SUM(amount), 0.0) AS v FROM transactions WHERE status = 'SUCCESS'")
            total_volume = float(cursor.fetchone()["v"])
        finally:
            conn.close()
        return {"total_transactions": total_transactions, "total_volume": round(total_volume, 2)}

    # ---- Consumer top-up requests (admin-approved) ----

    def create_topup_request(self, user_id: int, amount: float) -> int:
        """Creates a PENDING top-up request from a consumer."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO topup_requests (user_id, amount, status, created_at) VALUES (?, ?, 'PENDING', ?)",
                (user_id, float(amount), self._now_iso()),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def list_topup_requests(self, status: Optional[str] = None, user_id: Optional[int] = None) -> List[Dict]:
        """Top-up requests with consumer names, newest first."""
        query = """
            SELECT tr.*, u.name AS consumer_name, u.usn AS consumer_usn
            FROM topup_requests tr
            LEFT JOIN users u ON u.id = tr.user_id
        """
        clauses, params = [], []
        if status:
            clauses.append("tr.status = ?")
            params.append(status)
        if user_id is not None:
            clauses.append("tr.user_id = ?")
            params.append(user_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY tr.created_at DESC"

        conn = self.get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "consumer_name": r["consumer_name"],
                "consumer_usn": r["consumer_usn"],
                "amount": float(r["amount"]),
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def approve_topup_request(self, request_id: int) -> Dict:
        """Approves a PENDING top-up request: credits wallet + logs TOPUP txn."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM topup_requests WHERE id = ?", (request_id,))
            req = cursor.fetchone()
        finally:
            conn.close()

        if not req:
            return {"success": False, "message": "Top-up request not found."}
        if req["status"] != "PENDING":
            return {"success": False, "message": f"Request #{request_id} is already {req['status']}."}

        result = self.add_topup(req["user_id"], float(req["amount"]), note="TOPUP")
        if result.get("success"):
            conn = self.get_connection()
            try:
                conn.execute("UPDATE topup_requests SET status = 'APPROVED' WHERE id = ?", (request_id,))
                conn.commit()
            finally:
                conn.close()
            result["message"] = f"Top-up request #{request_id} approved."
        return result

    def process_payment(
        self,
        sender_user_id: int,
        merchant_user_id: int,
        merchant_name: str,
        amount: float,
        biometric_score: Optional[float] = None,
    ) -> Dict:
        """
        Atomically debits the consumer wallet, credits the merchant wallet, and
        records a SUCCESS transaction — all within one SQLite transaction.

        Returns:
            {"status": "SUCCESS", "transaction_id", "new_balance", "amount"} or
            {"status": "FAILED_BALANCE", "transaction_id", "balance"} (logged, no money moved).
        """
        amount = float(amount)
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")

            cursor.execute("SELECT balance FROM wallets WHERE user_id = ?", (sender_user_id,))
            row = cursor.fetchone()
            if not row:
                conn.rollback()
                return {"status": "FAILED_AUTH", "message": "Consumer wallet not found."}
            balance = float(row["balance"])

            cursor.execute("SELECT balance FROM wallets WHERE user_id = ?", (merchant_user_id,))
            mrow = cursor.fetchone()
            if not mrow:
                conn.rollback()
                return {"status": "FAILED_AUTH", "message": "Merchant wallet not found."}

            txn_id = self.new_transaction_id()

            if balance < amount:
                # Log the failed attempt for audit; no balances change.
                cursor.execute(
                    """
                    INSERT INTO transactions (id, sender_user_id, merchant_id, merchant_name, amount, status, biometric_score, note, created_at)
                    VALUES (?, ?, ?, ?, ?, 'FAILED_BALANCE', ?, 'Insufficient wallet balance', ?)
                    """,
                    (txn_id, sender_user_id, merchant_user_id, merchant_name, amount, biometric_score, self._now_iso()),
                )
                conn.commit()
                return {"status": "FAILED_BALANCE", "transaction_id": txn_id, "balance": round(balance, 2)}

            new_balance = balance - amount
            cursor.execute(
                "UPDATE wallets SET balance = balance - ?, updated_at = ? WHERE user_id = ?",
                (amount, self._now_iso(), sender_user_id),
            )
            cursor.execute(
                "UPDATE wallets SET balance = balance + ?, updated_at = ? WHERE user_id = ?",
                (amount, self._now_iso(), merchant_user_id),
            )
            cursor.execute(
                """
                INSERT INTO transactions (id, sender_user_id, merchant_id, merchant_name, amount, status, biometric_score, note, created_at)
                VALUES (?, ?, ?, ?, ?, 'SUCCESS', ?, 'Palm-verified payment', ?)
                """,
                (txn_id, sender_user_id, merchant_user_id, merchant_name, amount, biometric_score, self._now_iso()),
            )
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            finally:
                conn.close()
            return {"status": "FAILED_BALANCE", "message": f"Payment processing error: {exc}"}
        finally:
            try:
                conn.close()
            except Exception:
                pass

        return {
            "status": "SUCCESS",
            "transaction_id": txn_id,
            "new_balance": round(new_balance, 2),
            "amount": round(amount, 2),
        }

    def add_topup(self, user_id: int, amount: float, note: str = "TOPUP") -> Dict:
        """
        Credits a consumer wallet and logs a TOPUP transaction atomically.
        Returns {"success", "new_balance", "transaction_id"} or {"success": False, ...}.
        """
        if amount is None or float(amount) <= 0:
            return {"success": False, "message": "Top-up amount must be greater than zero."}

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT balance FROM wallets WHERE user_id = ?", (user_id,))
            wallet = cursor.fetchone()
            if not wallet:
                return {"success": False, "message": "Wallet not found for this user."}

            start_balance = float(wallet["balance"])
            new_balance = start_balance + float(amount)
            cursor.execute(
                "UPDATE wallets SET balance = ?, updated_at = ? WHERE user_id = ?",
                (new_balance, self._now_iso(), user_id),
            )
            txn_id = self.new_transaction_id()
            cursor.execute(
                """
                INSERT INTO transactions (id, sender_user_id, merchant_id, merchant_name, amount, status, biometric_score, note, created_at)
                VALUES (?, ?, NULL, NULL, ?, 'SUCCESS', NULL, ?, ?)
                """,
                (txn_id, user_id, float(amount), note, self._now_iso()),
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            return {"success": False, "message": f"Top-up failed: {exc}"}
        finally:
            conn.close()

        return {"success": True, "new_balance": round(new_balance, 2), "transaction_id": txn_id}
