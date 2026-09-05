"""
Integration tests for Payment Flow with Phase-2 Liveness Gate.
"""

import unittest
from flask import session

import config
from app import app, db, biometric_sessions, liveness_sessions


class TestPaymentLiveness(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

        self.orig_requires_liveness = getattr(config, "PAYMENT_REQUIRES_LIVENESS", False)
        config.PAYMENT_REQUIRES_LIVENESS = False

        # Ensure demo consumer user and wallet exist
        self.user = db.get_user_by_usn("TEST_LIVENESS_USER")
        if not self.user:
            uid = db.add_user("Test Liveness User", "TEST_LIVENESS_USER")
            self.user = db.get_user_by_id(uid)
        db.ensure_wallet(self.user["id"], "CONSUMER")
        db.credit_wallet(self.user["id"], 500.0)

    def tearDown(self):
        config.PAYMENT_REQUIRES_LIVENESS = self.orig_requires_liveness

    def _create_auth_session(self):
        token = biometric_sessions.create(
            user_id=self.user["id"],
            name=self.user["name"],
            usn=self.user["usn"],
            score=0.98,
        )
        with self.client.session_transaction() as sess:
            sess["bio_token"] = token
        return token

    def test_payment_succeeds_without_liveness_when_flag_disabled(self):
        """When PAYMENT_REQUIRES_LIVENESS=False, standard payment flow succeeds."""
        config.PAYMENT_REQUIRES_LIVENESS = False
        self._create_auth_session()

        resp = self.client.post(
            "/api/payment",
            json={"merchant_id": "m1", "amount": 25.0},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["amount"], 25.0)

    def test_payment_rejected_when_liveness_missing_and_flag_enabled(self):
        """When PAYMENT_REQUIRES_LIVENESS=True, payment without liveness token is 403."""
        config.PAYMENT_REQUIRES_LIVENESS = True
        self._create_auth_session()

        resp = self.client.post(
            "/api/payment",
            json={"merchant_id": "m1", "amount": 20.0},
        )
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["reason"], "LIVENESS_MISSING")
        self.assertTrue(data.get("liveness_missing"))

    def test_payment_rejected_with_invalid_or_consumed_token(self):
        """When PAYMENT_REQUIRES_LIVENESS=True, invalid or reused token is rejected."""
        config.PAYMENT_REQUIRES_LIVENESS = True
        self._create_auth_session()

        resp = self.client.post(
            "/api/payment",
            json={"merchant_id": "m1", "amount": 15.0, "liveness_token": "fake_or_consumed_token"},
        )
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data["reason"], "LIVENESS_INVALID")

    def test_payment_succeeds_with_valid_liveness_token(self):
        """When PAYMENT_REQUIRES_LIVENESS=True, payment with valid token succeeds."""
        config.PAYMENT_REQUIRES_LIVENESS = True
        self._create_auth_session()

        live_token = liveness_sessions.issue_token(score=0.92)

        resp = self.client.post(
            "/api/payment",
            json={"merchant_id": "m1", "amount": 30.0, "liveness_token": live_token},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["amount"], 30.0)
        self.assertTrue(data.get("liveness_passed"))
        self.assertEqual(data.get("liveness_score"), 0.92)

        # Confirm token was consumed (cannot be reused)
        self._create_auth_session()
        resp_reused = self.client.post(
            "/api/payment",
            json={"merchant_id": "m1", "amount": 10.0, "liveness_token": live_token},
        )
        self.assertEqual(resp_reused.status_code, 403)


if __name__ == "__main__":
    unittest.main()
