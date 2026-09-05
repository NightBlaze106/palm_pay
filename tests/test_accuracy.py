"""
Accuracy Sprint Tests (Demo / Prototype).

Covers:
- FrameQualityGate: blur / glare / brightness / palm-span rejection branches
- Hardened PalmMatcher: mean-template + best-sample dual pass, ambiguity guard,
  frame_pass_score helper
- Strict evaluation policy: suggested threshold must have ZERO observed
  impostor accepts while keeping GAR >= EVAL_SUGGESTION_MIN_GAR
- Multi-frame fusion on POST /api/recognize:
    * >= FUSION_MIN_PASS_FRAMES agreeing frames  -> recognized (+ biometric session)
    * fused match but too few agreeing frames     -> 'unstable_capture', no session
    * legacy single-image body still supported

Biometric extraction is mocked at app.pipeline.process_frame — core biometrics
are exercised through their own unit tests here.
"""

import json
import unittest
import unittest.mock
from unittest.mock import patch

import numpy as np

import config
import app as flask_app
from core.pipeline import FrameQualityGate, apply_finger_mask
from core.matcher import PalmMatcher
from core.evaluation import EvaluationEngine
from tests.test_payment import _pipeline_result, _unit_vec, _variant, _b64_image


class FingerMaskTests(unittest.TestCase):
    """apply_finger_mask — pure-function behavior."""

    def setUp(self):
        # Vertical-gradient image (each column differs; rows identical).
        self.img = np.tile(np.linspace(60, 200, 128).astype(np.uint8), (128, 1))

    def test_applied_top_band_flat_and_palm_untouched(self):
        out, info = apply_finger_mask(self.img, 48.0)
        self.assertTrue(info["applied"])
        self.assertAlmostEqual(info["cut_y"], 48, delta=2)

        # Rows well above the cut must equal the neutral fill.
        self.assertTrue(np.allclose(out[:40, :], out[0, 0]))
        # Rows well below the cut must be byte-exact untouched.
        self.assertTrue(np.array_equal(out[70:, :], self.img[70:, :]))

    def test_feathered_no_harsh_edge(self):
        out, _ = apply_finger_mask(self.img, 48.0)
        col = out[:, 5].astype(int)
        self.assertLessEqual(int(np.abs(np.diff(col)).max()), 12)

    def test_disabled_returns_original(self):
        with patch.object(config, "FINGER_MASK_ENABLED", False):
            out, info = apply_finger_mask(self.img, 48.0)
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, self.img))

    def test_none_knuckle_line_skips(self):
        out, info = apply_finger_mask(self.img, None)
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, self.img))

    def test_out_of_bounds_cut_skipped(self):
        _, info = apply_finger_mask(self.img, 5.0)
        self.assertFalse(info["applied"])
        self.assertIn("cut_out_of_bounds", info["skipped"])

    def test_embedding_on_masked_roi_stays_valid(self):
        from core.palm_model import get_palm_model
        rng = np.random.default_rng(0)
        base = np.tile(np.linspace(80, 180, 128), (128, 1)).astype(np.int16)
        roi = np.clip(base + rng.integers(-10, 10, (128, 128), dtype=np.int16), 0, 255).astype(np.uint8)
        masked, info = apply_finger_mask(roi, 50.0)
        self.assertTrue(info["applied"])

        vec = get_palm_model().extract_embedding(masked)
        self.assertEqual(vec.shape, (config.FEATURE_VECTOR_DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)


class FrameQualityGateTests(unittest.TestCase):

    def setUp(self):
        self.gate = FrameQualityGate()

    def test_sharp_well_lit_frame_passes(self):
        sharp = np.random.default_rng(1).integers(80, 180, (128, 128), dtype=np.uint8)
        res = self.gate.check(sharp, palm_span_px=90.0)
        self.assertTrue(res["passed"])
        self.assertIsNone(res["reason"])

    def test_blurry_frame_rejected(self):
        flat = np.full((128, 128), 120, dtype=np.uint8)
        flat[10:20, :] = 121  # tiny gradient keeps it off degenerate zero-variance edge
        res = self.gate.check(flat, 90.0)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reason"], "blurry")

    def test_glare_rejected(self):
        img = np.random.default_rng(2).integers(60, 160, (128, 128), dtype=np.uint8)
        img[:20, :] = 255
        res = self.gate.check(img, 90.0)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reason"], "glare")

    def test_dark_and_bright_frames_rejected(self):
        dark = np.full((128, 128), 25, dtype=np.uint8)
        self.assertEqual(self.gate.check(dark, 90.0)["reason"], "too_dark")

        bright = np.full((128, 128), 250, dtype=np.uint8)
        self.assertIn(self.gate.check(bright, 90.0)["reason"], ("too_bright", "glare"))

    def test_palm_too_far_rejected_before_scoring(self):
        sharp = np.random.default_rng(3).integers(80, 180, (128, 128), dtype=np.uint8)
        res = self.gate.check(sharp, palm_span_px=self.gate.min_palm_span - 10)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reason"], "too_far")


class HardenedMatcherTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self._rng = rng
        # Keep the shared matcher threshold deterministic inside this suite.
        self._original_threshold = flask_app.matcher.threshold
        self.matcher = PalmMatcher(threshold=0.78)

    def tearDown(self):
        flask_app.matcher.set_threshold(self._original_threshold)

    def _unit(self):
        v = self._rng.standard_normal(config.FEATURE_VECTOR_DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def _enrolled_pair(self):
        alice, bob = self._unit(), self._unit()
        return [
            {"id": 1, "name": "Alice", "usn": "A", "templates": [_variant(alice, i * 10 + 100) for i in range(4)]},
            {"id": 2, "name": "Bob", "usn": "B", "templates": [_variant(bob, i * 10 + 200) for i in range(4)]},
        ], alice, bob

    def test_genuine_recognized_via_mean_template(self):
        users, alice, _ = self._enrolled_pair()
        res = self.matcher.match_1_to_n(_variant(alice, 999), users)
        self.assertTrue(res["recognized"])
        self.assertEqual(res["matched_user"]["name"], "Alice")
        self.assertEqual(res["reason"], "ok")

    def test_half_correlated_stranger_rejected(self):
        """A palm sharing half its signal with an enrolled identity must NOT pass."""
        users, alice, _ = self._enrolled_pair()
        rejected = 0
        for seed in range(30):
            stranger = self._unit()
            mix = 0.5 * alice + 0.5 * stranger
            mix /= np.linalg.norm(mix)
            if not self.matcher.match_1_to_n(mix, users)["recognized"]:
                rejected += 1
        self.assertEqual(rejected, 30, "Dual mean+best rule must block inflated single-sample matches.")

    def test_ambiguous_identity_rejected(self):
        users, alice, bob = self._enrolled_pair()
        mid = (alice + bob) / 2
        mid /= np.linalg.norm(mid)
        low_t_matcher = PalmMatcher(threshold=0.60)  # force both users past threshold
        res = low_t_matcher.match_1_to_n(mid, users)
        self.assertFalse(res["recognized"])
        self.assertEqual(res["reason"], "ambiguous")
        self.assertIsNone(res["matched_user"])

    def test_frame_pass_score_helper(self):
        users, alice, _ = self._enrolled_pair()
        self.assertGreater(self.matcher.frame_pass_score(_variant(alice, 555), users), 0.9)
        self.assertLess(self.matcher.frame_pass_score(self._unit(), users), 0.5)


class StrictEvaluationPolicyTests(unittest.TestCase):

    def test_suggested_threshold_zero_far_with_gar_floor(self):
        rng = np.random.default_rng(42)
        users = []
        global_bias = rng.standard_normal(config.FEATURE_VECTOR_DIM).astype(np.float32)
        global_bias /= np.linalg.norm(global_bias)

        for i in range(3):
            base = rng.standard_normal(config.FEATURE_VECTOR_DIM).astype(np.float32)
            base /= np.linalg.norm(base)
            # Add small shared bias so impostor scores are positive across dimensions
            base = 0.85 * base + 0.15 * global_bias
            base /= np.linalg.norm(base)
            samples = []
            for j in range(4):
                n = rng.standard_normal(config.FEATURE_VECTOR_DIM).astype(np.float32)
                n /= np.linalg.norm(n)
                s = 0.96 * base + 0.04 * n
                samples.append(s / np.linalg.norm(s))
            users.append({"id": i + 1, "name": f"U{i}", "usn": f"X{i}", "templates": samples})

        res = EvaluationEngine().evaluate_enrolled_dataset(users)
        suggested = res["suggested_threshold"]
        self.assertEqual(res["suggestion_policy"].startswith("zero_far"), True)

        point = next(p for p in res["threshold_sweep"] if p["threshold"] == suggested)
        self.assertEqual(point["imp_accept_count"], 0, "Suggested point must have zero impostor accepts.")
        self.assertGreaterEqual(point["gar"], config.EVAL_SUGGESTION_MIN_GAR)
        self.assertGreater(res["impostor_stats"]["max"], 0)
        self.assertLess(res["impostor_stats"]["max"] + 0.05, suggested + 1.0)  # sanity band


class FusionEndpointTests(unittest.TestCase):
    """POST /api/recognize with {images: [...]} — mocked pipeline boundary."""

    def setUp(self):
        self.client = flask_app.app.test_client()
        self.db = flask_app.db
        self.sessions = flask_app.biometric_sessions
        self._orig_threshold = flask_app.matcher.threshold
        flask_app.matcher.set_threshold(0.78)
        self.sessions._sessions.clear()
        self.created_ids = []

    def tearDown(self):
        for uid in self.created_ids:
            self.db.delete_user(uid)
        self.sessions._sessions.clear()
        flask_app.matcher.set_threshold(self._orig_threshold)

    def _enroll_direct(self, name, embedding):
        uid = self.db.add_user(name, f"FA{np.random.default_rng().integers(1e6)}")
        self.assertIsNotNone(uid)
        self.created_ids.append(uid)
        self.db.ensure_wallet(uid, "CONSUMER")
        ok = self.db.add_templates(uid, [embedding], model_name=config.MODEL_NAME)
        self.assertTrue(ok)
        return uid

    def test_multi_frame_recognized_when_min_frames_agree(self):
        alice = _unit_vec(11)
        self._enroll_direct("Fusion Alice", alice)
        frames = [_variant(alice, 100 + i) for i in range(4)]  # 4/4 clear the bar

        with patch_pipeline(frames):
            resp = self.client.post(
                "/api/recognize",
                data=json.dumps({"images": [_b64_image() for _ in range(4)]}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertTrue(data["recognized"], data.get("message"))
        self.assertEqual(data["matched_user"]["name"], "Fusion Alice")
        self.assertGreaterEqual(data["frames_quality_passed"], 3)
        self.assertGreaterEqual(data["frames_above_threshold"], config.FUSION_MIN_PASS_FRAMES)

        # Successful fusion opens the server-side biometric session.
        with self.client.session_transaction() as s:
            token = s.get("bio_token")
        self.assertIsNotNone(token)
        bio = flask_app.biometric_sessions.get(token)
        self.assertIsNotNone(bio)
        self.assertEqual(bio["name"], "Fusion Alice")

    def test_unstable_burst_rejected_even_if_fused_match_exists(self):
        """
        Each frame individually sits just UNDER threshold (noisy capture), but
        averaging cancels the noise so the fused query still matches. The fusion
        gate must reject this as 'unstable_capture' instead of authenticating.
        """
        alice = _unit_vec(22)
        self._enroll_direct("Stability Alice", alice)
        w = 0.47  # individual frames land ~0.75 (< 0.78); their mean clears it
        frames = []
        for i in range(4):
            noise = _unit_vec(300 + i)
            mixed = (1 - w) * alice + w * noise
            frames.append(mixed / np.linalg.norm(mixed))

        with patch_pipeline(frames):
            resp = self.client.post(
                "/api/recognize",
                data=json.dumps({"images": [_b64_image() for _ in range(4)]}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertFalse(data["recognized"], "Fused match without frame agreement must be rejected.")
        self.assertEqual(data["match_reason"], "unstable_capture")
        self.assertIsNone(data["matched_user"])

        with self.client.session_transaction() as s:
            self.assertIsNone(s.get("bio_token"), "Rejected scans must not open a payment session.")

    def test_legacy_single_image_body_still_works(self):
        alice = _unit_vec(33)
        self._enroll_direct("Legacy Alice", alice)

        with patch_pipeline([_variant(alice, 7)]):
            resp = self.client.post(
                "/api/recognize",
                data=json.dumps({"image": _b64_image()}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertTrue(data["recognized"])
        self.assertEqual(data["matched_user"]["name"], "Legacy Alice")
        self.assertEqual(data["frames_received"], 1)


# ---- helpers ----

def patch_pipeline(embedding_sequence):
    """Mock app.pipeline.process_frame to yield the given embeddings in order."""
    seq = list(embedding_sequence)
    calls = {"n": 0}

    def fake_process_frame(_frame_bgr, include_visuals=True):
        emb = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        result = _pipeline_result(emb)
        if include_visuals is False:
            result["preview_roi_bgr_b64"] = ""
            result["preview_enhanced_b64"] = ""
            result["preview_annotated_b64"] = ""
        return result

    return unittest.mock.patch("app.pipeline.process_frame", side_effect=fake_process_frame)


if __name__ == "__main__":
    unittest.main()
