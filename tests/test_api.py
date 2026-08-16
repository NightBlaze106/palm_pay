"""
Integration Tests for Flask Web Application & REST API Routes.
"""

import json
import os
import unittest
import numpy as np

import config
from app import app, db
from core.pipeline import BiometricPipeline


class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_page_routes_status_200(self):
        """Verify all GUI web pages render successfully."""
        routes = ["/", "/enroll", "/recognize", "/users", "/test"]
        for r in routes:
            response = self.app.get(r)
            self.assertEqual(response.status_code, 200, f"Route {r} failed with status {response.status_code}")
            self.assertIn(b"PalmPay", response.data)

    def test_api_users_list(self):
        """Verify /api/users returns JSON list of enrolled subjects."""
        response = self.app.get("/api/users")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertIsInstance(data["users"], list)

    def test_api_config_get_and_post(self):
        """Verify threshold read and update via /api/config."""
        # GET config
        res_get = self.app.get("/api/config")
        self.assertEqual(res_get.status_code, 200)
        cfg = json.loads(res_get.data)
        self.assertIn("threshold", cfg)
        self.assertIn("model_name", cfg)

        # POST update threshold
        res_post = self.app.post(
            "/api/config",
            data=json.dumps({"threshold": 0.82}),
            content_type="application/json",
        )
        self.assertEqual(res_post.status_code, 200)
        updated = json.loads(res_post.data)
        self.assertTrue(updated["success"])
        self.assertAlmostEqual(updated["threshold"], 0.82)

    def test_api_evaluate_and_csv_export(self):
        """Verify empirical evaluation engine and CSV export endpoints."""
        # POST evaluate
        res_eval = self.app.post("/api/evaluate")
        self.assertEqual(res_eval.status_code, 200)
        eval_data = json.loads(res_eval.data)
        self.assertIn("status", eval_data)

        # GET export CSV
        res_csv = self.app.get("/api/export_csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertEqual(res_csv.content_type, "text/csv; charset=utf-8")
        self.assertIn(b"PalmPay Biometric Evaluation Report", res_csv.data)

    def test_api_process_frame_with_synthetic_data(self):
        """Verify /api/process_frame handles base64 image data."""
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        b64 = BiometricPipeline.encode_image_to_base64(dummy_img)

        response = self.app.post(
            "/api/process_frame",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("hand_detected", data)
        self.assertIn("latency_ms", data)


if __name__ == "__main__":
    unittest.main()
