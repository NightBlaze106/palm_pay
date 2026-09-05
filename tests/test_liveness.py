"""
Unit tests for Phase-2 Challenge-Response Liveness Detection.
"""

import time
import unittest
import numpy as np

import config
from core.liveness import LivenessSessionStore, LandmarkLivenessDetector


def make_synthetic_hand(fist: bool = False, wrist_x: float = 100.0, wrist_y: float = 200.0):
    """Generates 21 synthetic landmarks for tests."""
    lms = [[0.0, 0.0, 0.0] for _ in range(21)]
    # Wrist
    lms[0] = [wrist_x, wrist_y, 0.0]
    # Middle MCP
    lms[9] = [wrist_x, wrist_y - 100.0, 0.0]

    # Other MCPs
    lms[2] = [wrist_x - 30.0, wrist_y - 80.0, 0.0]
    lms[5] = [wrist_x - 15.0, wrist_y - 95.0, 0.0]
    lms[13] = [wrist_x + 15.0, wrist_y - 95.0, 0.0]
    lms[17] = [wrist_x + 30.0, wrist_y - 85.0, 0.0]

    if fist:
        lms[4] = [wrist_x - 25.0, wrist_y - 70.0, 0.0]
        lms[8] = [wrist_x - 15.0, wrist_y - 90.0, 0.0]
        lms[12] = [wrist_x, wrist_y - 90.0, 0.0]
        lms[16] = [wrist_x + 15.0, wrist_y - 90.0, 0.0]
        lms[20] = [wrist_x + 25.0, wrist_y - 80.0, 0.0]
    else:
        lms[4] = [wrist_x - 50.0, wrist_y - 110.0, 0.0]
        lms[8] = [wrist_x - 20.0, wrist_y - 180.0, 0.0]
        lms[12] = [wrist_x, wrist_y - 190.0, 0.0]
        lms[16] = [wrist_x + 20.0, wrist_y - 180.0, 0.0]
        lms[20] = [wrist_x + 40.0, wrist_y - 160.0, 0.0]
    return lms


class TestLiveness(unittest.TestCase):

    def setUp(self):
        self.store = LivenessSessionStore(ttl_seconds=1)

    def test_challenge_start_and_expiry(self):
        """Verify challenge creation, fields, and expiration."""
        ch = self.store.create_challenge(length=2)
        self.assertIn("challenge_id", ch)
        self.assertEqual(len(ch["sequence"]), 2)
        self.assertIn("instructions", ch)

        cid = ch["challenge_id"]
        self.assertIsNotNone(self.store.get_challenge(cid))

        # Manually expire
        self.store._challenges[cid]["expires_at"] = time.time() - 1.0
        self.assertIsNone(self.store.get_challenge(cid))

    def test_challenge_consume_atomicity(self):
        """Verify challenge can be consumed only once."""
        ch = self.store.create_challenge()
        cid = ch["challenge_id"]
        c1 = self.store.consume_challenge(cid)
        self.assertIsNotNone(c1)
        c2 = self.store.consume_challenge(cid)
        self.assertIsNone(c2)

    def test_verify_rejects_too_few_frames(self):
        """Verify that a sequence shorter than LIVENESS_MIN_FRAMES is rejected."""
        frames = [{"ts_ms": 0, "landmarks": make_synthetic_hand(False)}]
        res = LandmarkLivenessDetector.verify_sequence(frames, ["OPEN_PALM", "FIST"])
        self.assertFalse(res["passed"])
        self.assertTrue(any("insufficient_frames" in r for r in res["reasons"]))

    def test_verify_rejects_static_frames(self):
        """Verify anti-static photo check."""
        hand = make_synthetic_hand(False)
        frames = [{"ts_ms": i * 100, "landmarks": hand} for i in range(10)]
        res = LandmarkLivenessDetector.verify_sequence(frames, ["OPEN_PALM", "FIST"])
        self.assertFalse(res["passed"])
        self.assertIn("static_image_detected", res["reasons"])

    def test_verify_rejects_wrong_order(self):
        """Verify challenge order mismatch rejection."""
        frames = []
        for i in range(5):
            frames.append({"ts_ms": i * 100, "landmarks": make_synthetic_hand(False)})
        for i in range(5):
            frames.append({"ts_ms": (5 + i) * 100, "landmarks": make_synthetic_hand(True)})

        res = LandmarkLivenessDetector.verify_sequence(frames, ["FIST", "OPEN_PALM"])
        self.assertFalse(res["passed"])
        self.assertTrue(any("challenge_order_mismatch" in r for r in res["reasons"]))

    def test_verify_accepts_valid_sequence(self):
        """Verify valid OPEN_PALM -> FIST articulation sequence passes."""
        frames = []
        for i in range(5):
            frames.append({"ts_ms": i * 100, "landmarks": make_synthetic_hand(False)})
        for i in range(5):
            frames.append({"ts_ms": (5 + i) * 100, "landmarks": make_synthetic_hand(True)})

        res = LandmarkLivenessDetector.verify_sequence(frames, ["OPEN_PALM", "FIST"])
        self.assertTrue(res["passed"])
        self.assertGreater(res["score"], 0.5)

    def test_token_single_use_and_expiry(self):
        """Verify liveness token is burned on consumption and expires."""
        tok = self.store.issue_token(score=0.95)
        ok, score, _ = self.store.verify_and_consume_token(tok)
        self.assertTrue(ok)
        self.assertEqual(score, 0.95)

        # Attempt reuse
        ok_reused, _, reason = self.store.verify_and_consume_token(tok)
        self.assertFalse(ok_reused)
        self.assertEqual(reason, "liveness_consumed_or_invalid")

        # Expired token
        tok_exp = self.store.issue_token(score=0.88)
        self.store._tokens[tok_exp]["expires_at"] = time.time() - 1.0
        ok_exp, _, exp_reason = self.store.verify_and_consume_token(tok_exp)
        self.assertFalse(ok_exp)
        self.assertEqual(exp_reason, "liveness_expired")

    def test_liveness_min_score_thresholding(self):
        """Verify that LIVENESS_MIN_SCORE enforces a score floor when configured > 0.0."""
        orig_min = config.LIVENESS_MIN_SCORE
        try:
            frames = []
            for i in range(5):
                frames.append({"ts_ms": i * 100, "landmarks": make_synthetic_hand(False)})
            for i in range(5):
                frames.append({"ts_ms": (5 + i) * 100, "landmarks": make_synthetic_hand(True)})

            # With default placeholder (0.0), sequence passes
            config.LIVENESS_MIN_SCORE = 0.0
            res_pass = LandmarkLivenessDetector.verify_sequence(frames, ["OPEN_PALM", "FIST"])
            self.assertTrue(res_pass["passed"])

            # With floor higher than score (e.g. 1.5), sequence is rejected for score_below_minimum
            config.LIVENESS_MIN_SCORE = 1.5
            res_fail = LandmarkLivenessDetector.verify_sequence(frames, ["OPEN_PALM", "FIST"])
            self.assertFalse(res_fail["passed"])
            self.assertTrue(any("score_below_minimum" in r for r in res_fail["reasons"]))
        finally:
            config.LIVENESS_MIN_SCORE = orig_min


if __name__ == "__main__":
    unittest.main()
