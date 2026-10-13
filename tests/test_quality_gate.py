"""
Unit tests for PalmPay Capture Quality Gate and Distance Estimation.
Verifies:
- Monocular pinhole distance estimation (Z_cm = K / W_px) and zone categorization.
- Dual-metric sharpness evaluation (Tenengrad gradient energy + 3x3 patch Laplacian).
- Boundary edge clipping detection.
- Proximity coaching (too close / too far / optimal).
"""

import unittest
import cv2
import numpy as np

import config
from core.pipeline import FrameQualityGate, evaluate_palm_sharpness, estimate_palm_distance, BiometricPipeline


class TestDistanceAndSharpnessQuality(unittest.TestCase):

    def setUp(self):
        self.gate = FrameQualityGate()

    def test_distance_estimation_zones(self):
        """Verify pinhole distance formula categorizes too_far, optimal, and too_close correctly."""
        # Too far: W_px = 100 px -> Z ~ 43.2 cm
        dist_far, zone_far, msg_far = estimate_palm_distance(100.0)
        self.assertGreater(dist_far, 33.0)
        self.assertEqual(zone_far, "too_far")
        self.assertIn("closer", msg_far.lower())

        # Optimal: W_px = 165 px -> Z ~ 26.2 cm (Sweet spot: 20-33 cm)
        dist_opt, zone_opt, msg_opt = estimate_palm_distance(165.0)
        self.assertGreaterEqual(dist_opt, 20.0)
        self.assertLessEqual(dist_opt, 33.0)
        self.assertEqual(zone_opt, "optimal")
        self.assertIn("ideal", msg_opt.lower())

        # Too close: W_px = 250 px -> Z ~ 17.3 cm (< 20 cm)
        dist_close, zone_close, msg_close = estimate_palm_distance(250.0)
        self.assertLess(dist_close, 20.0)
        self.assertEqual(zone_close, "too_close")
        self.assertIn("back", msg_close.lower())

        # Degenerate input
        dist_none, zone_none, _ = estimate_palm_distance(0.0)
        self.assertEqual(zone_none, "too_far")

    def test_evaluate_palm_sharpness_detects_blur(self):
        """Verify Tenengrad + patch Laplacian correctly accepts sharp textures and rejects blurred ones."""
        rng = np.random.default_rng(42)
        # Sharp high-frequency texture (crease-like patterns)
        sharp_roi = rng.integers(60, 200, (128, 128), dtype=np.uint8)

        passed, metrics, msg = evaluate_palm_sharpness(sharp_roi)
        self.assertTrue(passed)
        self.assertGreater(metrics["tenengrad"], config.QUALITY_MIN_TENENGRAD)
        self.assertGreater(metrics["patch_laplacian_q25"], config.QUALITY_MIN_PATCH_LAPLACIAN)

        # Heavily Gaussian blurred texture (simulating camera defocus or motion blur)
        blurred_roi = cv2.GaussianBlur(sharp_roi, (15, 15), 5.0)
        passed_blur, metrics_blur, msg_blur = evaluate_palm_sharpness(blurred_roi)
        self.assertFalse(passed_blur)
        self.assertIn("blurry", msg_blur.lower())
        self.assertLess(metrics_blur["tenengrad"], config.QUALITY_MIN_TENENGRAD)

    def test_quality_gate_rejects_edge_clipped_hands(self):
        """Verify boundary clipping fails with edge_clipped reason."""
        sharp = np.random.default_rng(1).integers(80, 180, (128, 128), dtype=np.uint8)
        res = self.gate.check(sharp, palm_span_px=100.0, palm_breadth_px=160.0, is_edge_clipped=True)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reason"], "edge_clipped")
        self.assertIn("edge", res["message"].lower())

    def test_quality_gate_rejects_too_close_distance(self):
        """Verify palm held too close (< 20 cm) is coached to move back."""
        sharp = np.random.default_rng(2).integers(80, 180, (128, 128), dtype=np.uint8)
        # Breadth > 215 px indicates hand is too close
        res = self.gate.check(sharp, palm_span_px=180.0, palm_breadth_px=240.0, is_edge_clipped=False)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reason"], "too_close")
        self.assertIn("close", res["message"].lower())

    def test_process_frame_includes_guidance_metadata(self):
        """Verify process_frame exposes distance and HUD guidance fields in response."""
        pipeline = BiometricPipeline()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(dummy_frame, include_visuals=False)

        # Required fields for frontend HUD
        self.assertIn("distance_cm", result)
        self.assertIn("distance_zone", result)
        self.assertIn("palm_center", result)
    def test_api_recognize_includes_guidance_metadata(self):
        """Verify /api/recognize includes distance_cm, distance_zone, palm_center, in_sweet_spot."""
        from app import app
        client = app.test_client()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        b64 = BiometricPipeline.encode_image_to_base64(dummy_frame)
        res = client.post("/api/recognize", json={"image": b64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("distance_cm", data)
        self.assertIn("distance_zone", data)
        self.assertIn("palm_center", data)
        self.assertIn("palm_breadth_px", data)
        self.assertIn("in_sweet_spot", data)


if __name__ == "__main__":
    unittest.main()
