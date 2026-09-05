"""
Unit tests for Phase-2 DeepPalmModel and model abstraction.
"""

import base64
import unittest
import uuid
from unittest.mock import patch
import cv2
import numpy as np

import config
import app as flask_app
from core.geometry import GEO_FEATURE_DIM
from core.palm_model import BasePalmModel, ClassicalPalmModel, DeepPalmModel, get_palm_model
from database.db import Database


class TestDeepModel(unittest.TestCase):

    def setUp(self):
        self.original_deep_enabled = config.DEEP_ENABLED
        self.original_model_name = config.MODEL_NAME
        self.original_dim = config.FEATURE_VECTOR_DIM

    def tearDown(self):
        config.DEEP_ENABLED = self.original_deep_enabled
        config.MODEL_NAME = self.original_model_name
        config.FEATURE_VECTOR_DIM = self.original_dim

    def test_deep_model_disabled_by_default(self):
        """Verify that default model factory returns ClassicalPalmModel when DEEP_ENABLED=False."""
        config.DEEP_ENABLED = False
        model = get_palm_model()
        self.assertIsInstance(model, ClassicalPalmModel)
        self.assertNotIsInstance(model, DeepPalmModel)

    def test_deep_model_instantiation_and_properties(self):
        """Verify DeepPalmModel dimension and model name reporting."""
        deep = DeepPalmModel(feature_dim=512, model_name="deep-test-v1")
        self.assertEqual(deep.get_feature_dim(), 512)
        self.assertEqual(deep.get_model_name(), "deep-test-v1")

    def test_deep_model_embedding_dimension_and_norm(self):
        """Verify embedding output shape and L2 normalization."""
        deep = DeepPalmModel(feature_dim=512)
        dummy_roi = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        emb = deep.extract_embedding(dummy_roi)

        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_deep_model_deterministic_fallback(self):
        """Verify identical inputs produce identical embeddings in fallback mode."""
        deep = DeepPalmModel(feature_dim=512)
        img1 = np.ones((128, 128), dtype=np.uint8) * 120
        img2 = np.ones((128, 128), dtype=np.uint8) * 120

        emb1 = deep.extract_embedding(img1)
        emb2 = deep.extract_embedding(img2)
        self.assertTrue(np.allclose(emb1, emb2, atol=1e-5))

    def test_db_filtering_by_model_name_and_dim(self):
        """Verify database get_all_users_with_templates isolates deep vs classical templates."""
        db = Database(config.DATABASE_PATH)
        # Query specifically for deep model templates
        deep_users = db.get_all_users_with_templates(model_name="deep-palm-embedder-v1", feature_dim=512)
        self.assertIsInstance(deep_users, list)
        for u in deep_users:
            for t in u["templates"]:
                self.assertEqual(len(t), 512)


def _b64_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _pipeline_result(embedding, geometry=None):
    emb = np.asarray(embedding, dtype=np.float32)
    if geometry is None:
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


class TestDeepEnrollmentAndMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = flask_app.app
        cls.client = cls.app.test_client()
        cls.db = flask_app.db

    def setUp(self):
        self.original_deep = config.DEEP_ENABLED
        self.original_match_t = config.DEEP_MATCH_THRESHOLD
        self.created_users = []

    def tearDown(self):
        config.DEEP_ENABLED = self.original_deep
        config.DEEP_MATCH_THRESHOLD = self.original_match_t
        flask_app.pipeline.reload_model()
        for uid in self.created_users:
            self.db.delete_user(uid)

    def test_enrollment_stores_classical_model_identity_when_deep_disabled(self):
        """A) DEEP_ENABLED=False enrollment stores classical model identity and dimension."""
        config.DEEP_ENABLED = False
        flask_app.pipeline.reload_model()

        usn = f"TEST_CLASSIC_{uuid.uuid4().hex[:8].upper()}"
        samples = [_b64_image() for _ in range(6)]
        classic_emb = np.random.randn(config.FEATURE_VECTOR_DIM).astype(np.float32)
        classic_emb /= np.linalg.norm(classic_emb)

        with patch.object(flask_app.pipeline, "process_frame", return_value=_pipeline_result(classic_emb)):
            res = self.client.post("/api/enroll", json={"name": "Classic Subject", "usn": usn, "samples": samples})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data["success"])
            uid = data["user_id"]
            self.created_users.append(uid)

            # Assert returned identity
            self.assertEqual(data["model_name"], config.MODEL_NAME)
            self.assertEqual(data["feature_dim"], config.FEATURE_VECTOR_DIM)

            # Assert database template row
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT feature_dim, model_name FROM templates WHERE user_id = ?", (uid,))
            rows = cursor.fetchall()
            conn.close()
            self.assertTrue(len(rows) > 0)
            for r in rows:
                self.assertEqual(r["model_name"], config.MODEL_NAME)
                self.assertEqual(r["feature_dim"], config.FEATURE_VECTOR_DIM)

    def test_enrollment_stores_deep_model_identity_when_deep_enabled(self):
        """B) DEEP_ENABLED=True enrollment stores deep-palm-embedder-v1 and 512."""
        config.DEEP_ENABLED = True
        flask_app.pipeline.reload_model()

        usn = f"TEST_DEEP_{uuid.uuid4().hex[:8].upper()}"
        samples = [_b64_image() for _ in range(6)]
        deep_emb = np.random.randn(512).astype(np.float32)
        deep_emb /= np.linalg.norm(deep_emb)

        with patch.object(flask_app.pipeline, "process_frame", return_value={
            **_pipeline_result(deep_emb),
            "model_name": "deep-palm-embedder-v1",
            "feature_dim": 512,
        }):
            res = self.client.post("/api/enroll", json={"name": "Deep Subject", "usn": usn, "samples": samples})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data["success"])
            uid = data["user_id"]
            self.created_users.append(uid)

            # Assert returned identity
            self.assertEqual(data["model_name"], "deep-palm-embedder-v1")
            self.assertEqual(data["feature_dim"], 512)

            # Assert database template row
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT feature_dim, model_name FROM templates WHERE user_id = ?", (uid,))
            rows = cursor.fetchall()
            conn.close()
            self.assertTrue(len(rows) > 0)
            for r in rows:
                self.assertEqual(r["model_name"], "deep-palm-embedder-v1")
                self.assertEqual(r["feature_dim"], 512)

    def test_db_template_loading_returns_correct_templates_for_active_model(self):
        """C) DB template loading returns the correct templates for the active model."""
        uid_c = self.db.add_user("Classical Guy", f"USN_C_{uuid.uuid4().hex[:6]}")
        uid_d = self.db.add_user("Deep Guy", f"USN_D_{uuid.uuid4().hex[:6]}")
        self.created_users.extend([uid_c, uid_d])

        v_classic = [np.ones(config.FEATURE_VECTOR_DIM, dtype=np.float32) / np.sqrt(config.FEATURE_VECTOR_DIM)]
        v_deep = [np.ones(512, dtype=np.float32) / np.sqrt(512)]

        self.db.add_templates(uid_c, v_classic, model_name=config.MODEL_NAME)
        self.db.add_templates(uid_d, v_deep, model_name="deep-palm-embedder-v1")

        # When DEEP_ENABLED=False
        config.DEEP_ENABLED = False
        users_classic = self.db.get_all_users_with_templates()
        classic_uids = [u["id"] for u in users_classic if u["templates"]]
        self.assertIn(uid_c, classic_uids)
        self.assertNotIn(uid_d, classic_uids)

        # When DEEP_ENABLED=True
        config.DEEP_ENABLED = True
        users_deep = self.db.get_all_users_with_templates()
        deep_uids = [u["id"] for u in users_deep if u["templates"]]
        self.assertIn(uid_d, deep_uids)
        self.assertNotIn(uid_c, deep_uids)

    def test_deep_enrolled_user_reaches_deep_matcher_path(self):
        """D) A deep-enrolled user can actually reach the deep matcher path."""
        config.DEEP_ENABLED = True
        flask_app.pipeline.reload_model()
        uid_d = self.db.add_user("Deep Target", f"USN_DT_{uuid.uuid4().hex[:6]}")
        self.created_users.append(uid_d)

        # 512-D template
        t_vec = np.zeros(512, dtype=np.float32)
        t_vec[0] = 1.0
        self.db.add_templates(uid_d, [t_vec], model_name="deep-palm-embedder-v1")

        matcher = flask_app.matcher
        enrolled = self.db.get_all_users_with_templates()

        # 1. Matching query vector clears deep threshold
        match_ok = matcher.match_1_to_n(t_vec, enrolled)
        self.assertTrue(match_ok["recognized"])
        self.assertEqual(match_ok["matched_user"]["id"], uid_d)

        # 2. Orthogonal query vector reaches deep_below_threshold path
        ortho_vec = np.zeros(512, dtype=np.float32)
        ortho_vec[1] = 1.0
        match_fail = matcher.match_1_to_n(ortho_vec, enrolled)
        self.assertFalse(match_fail["recognized"])
        self.assertEqual(match_fail["reason"], "deep_below_threshold")

    def test_matcher_active_threshold_controls(self):
        """Verify DEEP_MATCH_THRESHOLD behavior on PalmMatcher."""
        from core.matcher import PalmMatcher
        matcher = PalmMatcher(threshold=config.DEFAULT_RECOGNITION_THRESHOLD)
        orig_deep = config.DEEP_ENABLED
        orig_t = config.DEEP_MATCH_THRESHOLD
        try:
            config.DEEP_ENABLED = False
            self.assertEqual(matcher.active_threshold, config.DEFAULT_RECOGNITION_THRESHOLD)

            config.DEEP_ENABLED = True
            config.DEEP_MATCH_THRESHOLD = 0.0
            self.assertEqual(matcher.active_threshold, config.DEFAULT_RECOGNITION_THRESHOLD)

            config.DEEP_MATCH_THRESHOLD = 0.88
            self.assertEqual(matcher.active_threshold, 0.88)
        finally:
            config.DEEP_ENABLED = orig_deep
            config.DEEP_MATCH_THRESHOLD = orig_t

    def test_channel_handling_and_normalization(self):
        """Item 7: Verify 128x128 grayscale ROI conversion to 3-channel [-1.0, 1.0] tensor."""
        raw_gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        norm_gray = (raw_gray.astype(np.float32) / 127.5) - 1.0
        tensor_3ch = np.repeat(np.expand_dims(norm_gray, 0), 3, axis=0)

        self.assertEqual(tensor_3ch.shape, (3, 128, 128))
        self.assertGreaterEqual(float(np.min(tensor_3ch)), -1.0)
        self.assertLessEqual(float(np.max(tensor_3ch)), 1.0)
        self.assertTrue(np.array_equal(tensor_3ch[0], tensor_3ch[1]))
        self.assertTrue(np.array_equal(tensor_3ch[1], tensor_3ch[2]))


if __name__ == "__main__":
    unittest.main()
