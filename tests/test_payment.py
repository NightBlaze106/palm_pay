"""
Integration Tests for the PalmPay Payment System (Demo / Prototype).

Covers:
- Demo merchant accounts + MERCHANT wallets
- Wallet auto-creation on enrollment
- Biometric session gating of POST /api/payment (server-side payer identity,
  30-second freshness window, single-use consumption)
- Amount validation and insufficient-balance handling
- Admin authentication + top-up (direct and consumer-request flows)
- Merchant cash out request/approval flow
- Transaction listing/filtering, merchant stats, wallet API, receipt page

The biometric pipeline boundary is mocked (app.pipeline.process_frame) —
no changes to core/ are required or made.
"""

import base64
import json
import unittest
import uuid
from unittest.mock import patch

import cv2
import numpy as np

import config
import app as flask_app
from core.geometry import GEO_FEATURE_DIM
from database.db import Database


def _b64_image():
    """Tiny valid JPEG as a data URI (pipeline is mocked anyway)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(config.FEATURE_VECTOR_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _variant(vec: np.ndarray, seed: int, w: float = 0.05) -> np.ndarray:
    """Intra-class variation of an identity vector."""
    noise = _unit_vec(seed)
    out = (1.0 - w) * vec + w * noise
    return out / np.linalg.norm(out)


def _pipeline_result(embedding, geometry=None):
    """Full result dict shaped exactly like core.pipeline.BiometricPipeline.process_frame output."""
    emb = np.asarray(embedding, dtype=np.float32)
    if geometry is None:
        # Fixed synthetic hand-geometry vector (20 dims) so the enrollment and
        # matching paths see a consistent shape across all mocked frames.
        geometry = np.linspace(0.4, 1.6, GEO_FEATURE_DIM, dtype=np.float32)
    return {
        "success": True,
        "hand_detected": True,
        "landmarks_detected": True,
        "landmarks_count": 21,
        "roi_extracted": True,
        "roi_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "embedding": emb,
        "geometry": np.asarray(geometry, dtype=np.float32),
        "geometry_ok": True,
        "model_name": config.MODEL_NAME,
        "feature_dim": config.FEATURE_VECTOR_DIM,
        "latency_ms": 1.23,
        "preview_roi_bgr_b64": "",
        "preview_enhanced_b64": "",
        "preview_annotated_b64": "",
        "message": "Palm biometric features extracted successfully.",
    }


class PaymentSystemTests(unittest.TestCase):
    """End-to-end tests for the demo payment layer."""

    @classmethod
    def setUpClass(cls):
        cls.app = flask_app.app
        cls.db = flask_app.db
        cls.sessions = flask_app.biometric_sessions
        # Snapshot merchant wallet balances so shared demo data can be restored.
        cls.merchant_balance_snapshot = {
            acc["user_id"]: cls.db.get_wallet(acc["user_id"])["balance"]
            for acc in flask_app.MERCHANT_ACCOUNTS.values()
        }

    def setUp(self):
        self.client = self.app.test_client()
        self.admin_client = self.app.test_client()
        self.admin_client.post(
            "/admin/login",
            data={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD},
        )
        self.created_user_ids = []
        self._img = _b64_image()
        self.sessions._sessions.clear()

    def tearDown(self):
        for uid in self.created_user_ids:
            self.db.delete_user(uid)
        self.sessions._sessions.clear()

    @classmethod
    def tearDownClass(cls):
        # Restore merchant wallet balances altered by test payments/cashouts.
        for uid, snapshot in cls.merchant_balance_snapshot.items():
            current = cls.db.get_wallet(uid)["balance"]
            delta = round(current - snapshot, 6)
            if delta > 0:
                cls.db.debit_wallet(uid, delta)
            elif delta < 0:
                cls.db.credit_wallet(uid, -delta)

    # ---- helpers ----

    def _unique_usn(self) -> str:
        return f"PT{uuid.uuid4().hex[:8].upper()}"

    def _track_user(self, usn: str) -> int:
        user = self.db.get_user_by_usn(usn)
        self.assertIsNotNone(user)
        self.created_user_ids.append(user["id"])
        return user["id"]

    def _enroll(self, name: str, embedding) -> int:
        """Enrolls a consumer through the real /api/enroll route with mocked pipeline."""
        usn = self._unique_usn()
        with patch("app.pipeline.process_frame", return_value=_pipeline_result(embedding)):
            resp = self.client.post(
                "/api/enroll",
                data=json.dumps({"name": name, "usn": usn, "samples": [self._img] * 4}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        uid = data["user_id"]
        self.created_user_ids.append(uid)
        return uid

    def _inject_bio_session(self, uid: int, name: str, usn: str, score: float):
        token = self.sessions.create(uid, name, usn, score)
        with self.client.session_transaction() as s:
            s["bio_token"] = token
        return token


    # ---- Stage 1: merchants + wallets ----

    def test_merchant_accounts_and_wallets_seeded(self):
        for mid, acc in flask_app.MERCHANT_ACCOUNTS.items():
            self.assertIn(mid, ["m1", "m2", "m3", "m4"])
            wallet = self.db.get_wallet(acc["user_id"])
            self.assertIsNotNone(wallet)
            self.assertEqual(wallet["role"], "MERCHANT")
        # Merchant accounts must not leak into the biometric user listing.
        biometric_names = [u["usn"] for u in self.db.get_all_users_with_templates()]
        self.assertFalse(any(u.startswith(config.MERCHANT_USN_PREFIX) for u in biometric_names))

    def test_enrollment_geometry_consistency_gate(self):
        """Wild-geometry samples are dropped; all-wild enrollments are rejected."""
        name = "Consistency Carol"
        usn = self._unique_usn()
        base_geo = np.linspace(0.4, 1.6, GEO_FEATURE_DIM, dtype=np.float32)
        wild_geo = base_geo + 5.0  # far outside any sane pose envelope

        results = [
            _pipeline_result(_variant(_unit_vec(21), i)) for i in range(4)
        ]
        results[2]["geometry"] = wild_geo.copy()

        with patch("app.pipeline.process_frame", side_effect=results):
            resp = self.client.post(
                "/api/enroll",
                data=json.dumps({"name": name, "usn": usn, "samples": [self._img] * 4}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertEqual(data["sample_count"], 3)  # wild sample dropped
        self.assertEqual(data["geometry_count"], 3)
        uid = self._track_user(usn)
        stored = self.db.get_all_users_with_templates()
        geo_counts = {u["id"]: len(u["geometry"]) for u in stored}
        self.assertEqual(geo_counts.get(uid), 3)

    def test_enrollment_rejected_when_all_samples_inconsistent(self):
        name = "Inconsistent Ivan"
        usn = self._unique_usn()
        base_geo = np.linspace(0.4, 1.6, GEO_FEATURE_DIM, dtype=np.float32)
        results = [
            _pipeline_result(_variant(_unit_vec(31), i), geometry=base_geo + (i * 7.0))
            for i in range(4)
        ]
        with patch("app.pipeline.process_frame", side_effect=results):
            resp = self.client.post(
                "/api/enroll",
                data=json.dumps({"name": name, "usn": usn, "samples": [self._img] * 4}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        self.assertFalse(data["success"])
        self.assertIn("inconsistent", data["message"])

    def test_wallet_auto_created_on_enrollment(self):
        vec = _unit_vec(101)
        uid = self._enroll("Auto Wallet Consumer", vec)
        wallet = self.db.get_wallet(uid)
        self.assertIsNotNone(wallet)
        self.assertEqual(wallet["role"], "CONSUMER")
        self.assertAlmostEqual(wallet["balance"], 0.0)

    # ---- Stage 4: payment security model ----

    def test_payment_requires_biometric_session(self):
        before = self.db.get_payment_system_stats()["total_transactions"]
        resp = self.client.post(
            "/api/payment",
            data=json.dumps({"merchant_id": "m1", "amount": 100}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 401)
        data = json.loads(resp.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["reason"], "FAILED_AUTH")
        after = self.db.get_payment_system_stats()["total_transactions"]
        self.assertEqual(before, after, "No transaction row should be logged without a session.")

    def test_recognize_creates_server_side_session_on_match(self):
        alice_vec = _unit_vec(11)
        uid = self._enroll("Session Alice", alice_vec)
        user = self.db.get_user_by_id(uid)

        with patch("app.pipeline.process_frame", return_value=_pipeline_result(_variant(alice_vec, 12))):
            resp = self.client.post("/api/recognize", data=json.dumps({"image": self._img}), content_type="application/json")

        # Existing /api/recognize response contract must be unchanged.
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertTrue(data["recognized"])
        self.assertEqual(data["matched_user"]["id"], uid)
        self.assertIn("similarity_score", data)
        self.assertIn("threshold_used", data)

        # Server-side session created for THIS browser.
        with self.client.session_transaction() as s:
            token = s.get("bio_token")
        self.assertIsNotNone(token)
        bio = self.sessions.get(token)
        self.assertIsNotNone(bio)
        self.assertEqual(bio["user_id"], uid)
        self.assertEqual(bio["name"], user["name"])
        self.assertGreater(bio["score"], 0.9)

    def test_full_payment_success_flow_single_use_session(self):
        alice_vec = _unit_vec(21)
        uid = self._enroll("Payment Alice", alice_vec)
        merchant_acc = flask_app.MERCHANT_ACCOUNTS["m2"]
        self.db.credit_wallet(uid, 500.0)

        bal_before = self.db.get_wallet(uid)["balance"]
        merch_before = self.db.get_wallet(merchant_acc["user_id"])["balance"]

        with patch("app.pipeline.process_frame", return_value=_pipeline_result(alice_vec)):
            rec = json.loads(
                self.client.post("/api/recognize", data=json.dumps({"image": self._img}), content_type="application/json").data
            )
        self.assertTrue(rec["recognized"])
        score = rec["similarity_score"]

        # Frontend-sent user_id must be ignored — identity comes from the session.
        resp = self.client.post(
            "/api/payment",
            data=json.dumps({"merchant_id": "m2", "amount": 100, "user_id": 999999}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertEqual(data["consumer"]["id"], uid)
        self.assertEqual(data["merchant"]["id"], "m2")
        self.assertAlmostEqual(data["new_balance"], bal_before - 100.0)
        self.assertTrue(data["receipt_url"].startswith("/receipt/"))

        # Atomic wallet movement.
        self.assertAlmostEqual(self.db.get_wallet(uid)["balance"], bal_before - 100.0)
        self.assertAlmostEqual(self.db.get_wallet(merchant_acc["user_id"])["balance"], merch_before + 100.0)

        # Transaction record with the biometric score.
        txn = self.db.get_transaction(data["transaction_id"])
        self.assertEqual(txn["status"], "SUCCESS")
        self.assertEqual(txn["sender_user_id"], uid)
        self.assertEqual(txn["merchant_id"], merchant_acc["user_id"])
        self.assertEqual(txn["merchant_name"], "Photocopy")
        self.assertAlmostEqual(txn["biometric_score"], score)
        self.assertEqual(txn["type"], "PAYMENT")

        # Session consumed: one scan authorizes exactly one payment.
        resp2 = self.client.post(
            "/api/payment",
            data=json.dumps({"merchant_id": "m2", "amount": 10}),
            content_type="application/json",
        )
        self.assertEqual(resp2.status_code, 401)

    def test_failed_recognition_invalidates_previous_session(self):
        alice_vec = _unit_vec(31)
        uid = self._enroll("Invalidation Alice", alice_vec)
        stranger_vec = _unit_vec(77)  # orthogonal to every enrolled template
        self._inject_bio_session(uid, "Invalidation Alice", "X", 0.99)

        with patch("app.pipeline.process_frame", return_value=_pipeline_result(stranger_vec)):
            data = json.loads(
                self.client.post("/api/recognize", data=json.dumps({"image": self._img}), content_type="application/json").data
            )
        self.assertFalse(data["recognized"])

        with self.client.session_transaction() as s:
            self.assertIsNone(s.get("bio_token"), "A failed scan must clear any pending biometric session.")
        resp = self.client.post(
            "/api/payment", data=json.dumps({"merchant_id": "m1", "amount": 50}), content_type="application/json"
        )
        self.assertEqual(resp.status_code, 401)

    def test_amount_validation_logs_failed_amount(self):
        uid = self.db.add_user("Amount Tester", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")
        self._inject_bio_session(uid, "Amount Tester", "X", 0.9)

        for bad_amount in [0, -5, 5001, "abc"]:
            resp = self.client.post(
                "/api/payment",
                data=json.dumps({"merchant_id": "m1", "amount": bad_amount}),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 400)
            data = json.loads(resp.data)
            self.assertEqual(data["reason"], "FAILED_AMOUNT")

        failed_rows = [
            t for t in self.db.list_transactions(user_id=uid, role="consumer", limit=50)["transactions"]
            if t["status"] == "FAILED_AMOUNT"
        ]
        self.assertEqual(len(failed_rows), 4, "Each rejected amount should be audit-logged.")

    def test_insufficient_balance_logged_and_no_debit(self):
        uid = self.db.add_user("Broke Tester", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")
        self.db.credit_wallet(uid, 40.0)
        merchant_acc = flask_app.MERCHANT_ACCOUNTS["m3"]
        merch_before = self.db.get_wallet(merchant_acc["user_id"])["balance"]

        self._inject_bio_session(uid, "Broke Tester", "X", 0.88)
        resp = self.client.post(
            "/api/payment",
            data=json.dumps({"merchant_id": "m3", "amount": 100}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 402)
        data = json.loads(resp.data)
        self.assertEqual(data["reason"], "FAILED_BALANCE")
        self.assertAlmostEqual(data["balance"], 40.0)

        # No money moved.
        self.assertAlmostEqual(self.db.get_wallet(uid)["balance"], 40.0)
        self.assertAlmostEqual(self.db.get_wallet(merchant_acc["user_id"])["balance"], merch_before)

        failed = [
            t for t in self.db.list_transactions(user_id=uid, role="consumer", limit=50)["transactions"]
            if t["status"] == "FAILED_BALANCE"
        ]
        self.assertTrue(failed)

    def test_expired_biometric_session_rejected(self):
        uid = self.db.add_user("Expiry Tester", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")

        token = self._inject_bio_session(uid, "Expiry Tester", "X", 0.9)
        # Force-expire beyond the 30-second window.
        entry = self.sessions._sessions[token]
        entry["created_at"] -= (self.sessions.ttl_seconds + 1)

        resp = self.client.post(
            "/api/payment",
            data=json.dumps({"merchant_id": "m1", "amount": 25}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(json.loads(resp.data)["reason"], "FAILED_AUTH")

    def test_unknown_merchant_rejected(self):
        resp = self.client.post(
            "/api/payment",
            data=json.dumps({"merchant_id": "hacker_bank", "amount": 10}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(json.loads(resp.data)["reason"], "INVALID_MERCHANT")


    # ---- Stage 2/3/7: admin, top-ups, cash out ----

    def test_admin_pages_require_login(self):
        anon = self.app.test_client()
        for path in ["/admin/dashboard", "/admin/topup", "/admin/cashouts"]:
            resp = anon.get(path)
            self.assertEqual(resp.status_code, 302)
            self.assertIn("/admin/login", resp.headers["Location"])
        resp = anon.post("/api/topup", data=json.dumps({"user_id": 1, "amount": 5}), content_type="application/json")
        self.assertEqual(resp.status_code, 401)

    def test_admin_login_rejects_bad_password(self):
        resp = self.admin_client.post("/admin/login", data={"username": "admin", "password": "wrong"})
        self.assertIn(b"Invalid admin credentials", resp.data)

    def test_direct_topup_flow(self):
        uid = self.db.add_user("Direct Topup", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")

        resp = self.admin_client.post(
            "/api/topup",
            data=json.dumps({"user_id": uid, "amount": 500}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertAlmostEqual(data["new_balance"], 500.0)

        txn = self.db.get_transaction(data["transaction_id"])
        self.assertEqual(txn["type"], "TOPUP")

        # Only consumers may be topped up.
        m_uid = flask_app.MERCHANT_ACCOUNTS["m1"]["user_id"]
        resp = self.admin_client.post("/api/topup", data=json.dumps({"user_id": m_uid, "amount": 5}), content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_consumer_topup_request_approval_flow(self):
        uid = self.db.add_user("Request Topup", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")

        resp = self.client.post(
            "/api/topup/request",
            data=json.dumps({"user_id": uid, "amount": 250}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        request_id = data["request_id"]

        pending = self.db.list_topup_requests(status="PENDING")
        self.assertTrue(any(p["id"] == request_id for p in pending))
        self.assertAlmostEqual(self.db.get_wallet(uid)["balance"], 0.0, "Balance must not change until approval.")

        # Non-admin cannot approve.
        resp = self.client.post(f"/api/topup_request/{request_id}/approve")
        self.assertEqual(resp.status_code, 401)

        resp = self.admin_client.post(f"/api/topup_request/{request_id}/approve")
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertAlmostEqual(self.db.get_wallet(uid)["balance"], 250.0)

        approved = self.db.list_topup_requests()
        target = next(r for r in approved if r["id"] == request_id)
        self.assertEqual(target["status"], "APPROVED")

        topups = [t for t in self.db.list_transactions(user_id=uid, role="consumer", limit=10)["transactions"] if t["type"] == "TOPUP"]
        self.assertTrue(topups)

    def test_cashout_request_approval_flow(self):
        merchant_acc = flask_app.MERCHANT_ACCOUNTS["m4"]
        self.db.credit_wallet(merchant_acc["user_id"], 300.0)
        balance_before = self.db.get_wallet(merchant_acc["user_id"])["balance"]

        # Over-balance request rejected at creation.
        resp = self.client.post(
            "/api/cashout",
            data=json.dumps({"merchant_id": "m4", "amount": 99999}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

        resp = self.client.post(
            "/api/cashout",
            data=json.dumps({"merchant_id": "m4", "amount": 200}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        request_id = data["request_id"]

        pending = self.db.list_cashout_requests(status="PENDING")
        self.assertTrue(any(p["id"] == request_id for p in pending))

        # Non-admin blocked; double approval rejected.
        resp = self.client.post(f"/api/cashout/{request_id}/approve")
        self.assertEqual(resp.status_code, 401)

        resp = self.admin_client.post(f"/api/cashout/{request_id}/approve")
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertAlmostEqual(self.db.get_wallet(merchant_acc["user_id"])["balance"], balance_before - 200.0)

        cashouts = [
            t for t in self.db.list_transactions(user_id=merchant_acc["user_id"], role="merchant", limit=50)["transactions"]
            if t["type"] == "CASHOUT"
        ]
        self.assertTrue(cashouts and cashouts[0]["status"] == "SUCCESS")

        resp = self.admin_client.post(f"/api/cashout/{request_id}/approve")
        self.assertEqual(resp.status_code, 400)


    # ---- Stages 5/6: stats, wallets, history, receipt ----

    def test_merchant_stats_endpoint_shape(self):
        resp = self.client.get("/api/merchant/stats/m1")
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        for key in ["today", "this_week", "total", "daily_chart_data"]:
            self.assertIn(key, data)
        self.assertEqual(len(data["daily_chart_data"]), 7)

        resp = self.client.get("/api/merchant/stats/nope")
        self.assertEqual(resp.status_code, 404)

    def test_wallet_endpoint_both_forms(self):
        uid = self.db.add_user("Wallet API", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")

        data = json.loads(self.client.get(f"/api/wallet/{uid}").data)
        self.assertTrue(data["success"])
        self.assertEqual(data["role"], "CONSUMER")
        self.assertIn("recent_transactions", data)
        self.assertIn("balance", data)

        data = json.loads(self.client.get("/api/wallet/m1").data)
        self.assertTrue(data["success"])
        self.assertEqual(data["role"], "MERCHANT")

        resp = self.client.get("/api/wallet/not-a-user")
        self.assertEqual(resp.status_code, 404)

    def test_transactions_listing_filters_pagination(self):
        uid = self.db.add_user("History Tester", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")
        self.db.credit_wallet(uid, 600.0)
        for _ in range(3):
            self._inject_bio_session(uid, "History Tester", "X", 0.8)
            resp = self.client.post(
                "/api/payment",
                data=json.dumps({"merchant_id": "m1", "amount": 50}),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200)

        page1 = json.loads(
            self.client.get(f"/api/transactions?role=consumer&user_id={uid}&limit=2&offset=0").data
        )
        page2 = json.loads(
            self.client.get(f"/api/transactions?role=consumer&user_id={uid}&limit=2&offset=2").data
        )
        self.assertEqual(page1["total"], 3)
        self.assertEqual(len(page1["transactions"]), 2)
        self.assertEqual(len(page2["transactions"]), 1)
        ids1 = {t["id"] for t in page1["transactions"]}
        ids2 = {t["id"] for t in page2["transactions"]}
        self.assertFalse(ids1 & ids2)

        only_success = json.loads(
            self.client.get(f"/api/transactions?role=consumer&user_id={uid}&status=SUCCESS").data
        )
        self.assertTrue(all(t["status"] == "SUCCESS" for t in only_success["transactions"]))
        self.assertEqual(only_success["total"], 3)

    def test_transactions_csv_export(self):
        resp = self.client.get("/api/transactions/export?role=merchant&user_id=m1")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Transaction ID", resp.data)
        self.assertIn(resp.content_type.split(";")[0], ["text/csv"])

    def test_receipt_page_found_and_missing(self):
        uid = self.db.add_user("Receipt Tester", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")
        txn_id = self.db.add_transaction(amount=42.0, status="SUCCESS", sender_user_id=uid,
                                         merchant_id=flask_app.MERCHANT_ACCOUNTS["m1"]["user_id"],
                                         merchant_name="Canteen", biometric_score=0.91, note="test")

        resp = self.client.get(f"/receipt/{txn_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"DEMO RECEIPT", resp.data)
        self.assertIn(txn_id[:8].encode(), resp.data)
        self.assertIn(b"simulated transaction", resp.data)

        resp = self.client.get("/receipt/does-not-exist")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Receipt not found", resp.data)

    def test_key_pages_render(self):
        uid = self.db.add_user("Page Tester", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")

        for path, marker in [
            ("/pay", b"MERCHANT ZONE"),
            ("/merchant?merchant_id=m1", b"Merchant Console"),
            # POS terminal is embedded in the merchant console (everything-in-one-page)
            ("/merchant?merchant_id=m1&view=pos", b"MERCHANT ZONE"),
            ("/merchant?merchant_id=m1&view=pos", b"CUSTOMER ZONE"),
            (f"/transactions?role=consumer&user_id={uid}", b"Transaction History"),
            ("/transactions?role=merchant&user_id=m1", b"Transaction History"),
            (f"/consumer?user_id={uid}", b"PalmPay"),
        ]:
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, path)
            self.assertIn(marker, resp.data)

    def test_consumer_dashboard_has_no_pay_action(self):
        """Consumer dashboard must be read-only: no Pay button/tab anywhere."""
        uid = self.db.add_user("ReadOnly Consumer", self._unique_usn())
        self.created_user_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")

        resp = self.client.get(f"/consumer?user_id={uid}")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertNotIn('href="/pay"', html)
        self.assertNotIn("Pay ₹", html)
        # Top-up request stays available.
        self.assertIn("Top Up", html)
        self.assertIn("merchant counter", html)

        # History page tab bar likewise has no Pay entry.
        resp = self.client.get(f"/transactions?role=consumer&user_id={uid}")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn('href="/pay"', resp.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
