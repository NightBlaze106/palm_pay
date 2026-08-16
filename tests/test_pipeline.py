"""
Unit and Integration Tests for PalmPay Biometric Pipeline.
Tests:
- HandDetector landmark extraction
- PalmROIExtractor affine normalization and geometry
- PalmPreprocessor grayscale & CLAHE enhancement
- MBuLBPFeatureExtractor uniform LBP mapping and 944-D output shape
- PalmMatcher cosine similarity and 1:N decision rules
- Database SQLite operations and vector blob serialization
"""

import os
import unittest
import numpy as np
import cv2

import config
from core.hand_detector import HandDetector
from core.roi import PalmROIExtractor
from core.preprocessing import PalmPreprocessor
from core.features import UniformLBPTable, MBuLBPFeatureExtractor
from core.palm_model import ClassicalPalmModel, get_palm_model
from core.matcher import PalmMatcher
from core.pipeline import BiometricPipeline
from database.db import Database


class TestBiometricPipeline(unittest.TestCase):

    def setUp(self):
        self.roi_extractor = PalmROIExtractor()
        self.preprocessor = PalmPreprocessor()
        self.feature_extractor = MBuLBPFeatureExtractor()
        self.model = ClassicalPalmModel()
        self.matcher = PalmMatcher()

    def test_uniform_lbp_table_properties(self):
        """Verify that uniform LBP table correctly identifies 58 uniform patterns + 1 non-uniform bin (59 total)."""
        table = UniformLBPTable.get_table()
        self.assertEqual(len(table), 256)
        unique_bins = np.unique(table)
        self.assertEqual(len(unique_bins), 59)
        self.assertEqual(np.min(unique_bins), 0)
        self.assertEqual(np.max(unique_bins), 58)

        # 0 (00000000) and 255 (11111111) must be uniform
        self.assertNotEqual(table[0], 58)
        self.assertNotEqual(table[255], 58)
        # 170 (10101010) has 8 transitions -> must map to non-uniform bin 58
        self.assertEqual(table[170], 58)

    def test_roi_extraction_geometry(self):
        """Test affine ROI extraction with synthetic hand landmark coordinates."""
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for y in range(480):
            dummy_frame[y, :, :] = (y % 255, (y * 2) % 255, (y * 3) % 255)

        # Realistic palm landmarks
        synthetic_landmarks = [(320, 360, 0.0)] * 21
        synthetic_landmarks[0] = (320, 360, 0.0)    # Wrist
        synthetic_landmarks[5] = (260, 220, 0.0)    # Index MCP
        synthetic_landmarks[9] = (320, 200, 0.0)    # Middle MCP
        synthetic_landmarks[13] = (350, 215, 0.0)   # Ring MCP
        synthetic_landmarks[17] = (380, 230, 0.0)   # Pinky MCP

        res = self.roi_extractor.extract_roi(dummy_frame, synthetic_landmarks)
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["roi_bgr"])
        self.assertEqual(res["roi_bgr"].shape, (config.ROI_HEIGHT, config.ROI_WIDTH, 3))
        self.assertEqual(len(res["roi_polygon"]), 4)

    def test_preprocessing_clahe(self):
        """Verify grayscale conversion and CLAHE local contrast enhancement."""
        dummy_roi = np.random.randint(50, 200, (config.ROI_HEIGHT, config.ROI_WIDTH, 3), dtype=np.uint8)
        prep = self.preprocessor.preprocess(dummy_roi)
        self.assertTrue(prep["success"])
        self.assertEqual(prep["enhanced"].shape, (config.ROI_HEIGHT, config.ROI_WIDTH))
        self.assertEqual(prep["enhanced"].dtype, np.uint8)
        self.assertEqual(prep["normalized"].shape, (config.ROI_HEIGHT, config.ROI_WIDTH))
        self.assertTrue(np.all(prep["normalized"] >= 0.0) and np.all(prep["normalized"] <= 1.0))

    def test_feature_extraction_dimension_and_norm(self):
        """Verify that MB-uLBP produces exactly a 944-dimensional unit-norm vector."""
        dummy_enhanced = np.random.randint(0, 256, (config.ROI_HEIGHT, config.ROI_WIDTH), dtype=np.uint8)
        vec = self.feature_extractor.extract_features(dummy_enhanced)

        self.assertEqual(vec.shape, (config.FEATURE_VECTOR_DIM,))
        self.assertEqual(len(vec), 944)
        self.assertEqual(vec.dtype, np.float32)

        # L2 Norm of non-zero feature vector must be 1.0
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_cosine_similarity_matcher(self):
        """Verify cosine similarity and 1:N matching decision logic."""
        vec1 = np.random.randn(944).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        # Exact match similarity must equal 1.0
        self.assertAlmostEqual(PalmMatcher.cosine_similarity(vec1, vec1), 1.0, places=5)

        # Genuine-like sample (high correlation)
        noise = np.random.randn(944).astype(np.float32)
        noise /= np.linalg.norm(noise)
        vec1_perturbed = 0.95 * vec1 + 0.05 * noise
        vec1_perturbed /= np.linalg.norm(vec1_perturbed)
        sim_genuine = PalmMatcher.cosine_similarity(vec1, vec1_perturbed)
        self.assertGreater(sim_genuine, 0.90)

        # Impostor-like sample (independent random vector)
        vec2 = np.random.randn(944).astype(np.float32)
        vec2 /= np.linalg.norm(vec2)
        sim_impostor = PalmMatcher.cosine_similarity(vec1, vec2)
        self.assertLess(sim_impostor, 0.50)

        # 1:N Identification Test
        enrolled = [
            {"id": 1, "name": "Alice", "usn": "1MS21CS001", "templates": [vec1]},
            {"id": 2, "name": "Bob", "usn": "1MS21CS002", "templates": [vec2]},
        ]
        matcher = PalmMatcher(threshold=0.75)

        # Query matching Alice
        res_alice = matcher.match_1_to_n(vec1_perturbed, enrolled)
        self.assertTrue(res_alice["recognized"])
        self.assertEqual(res_alice["matched_user"]["name"], "Alice")

        # Query of unknown person
        vec_unknown = np.random.randn(944).astype(np.float32)
        vec_unknown /= np.linalg.norm(vec_unknown)
        res_unknown = matcher.match_1_to_n(vec_unknown, enrolled)
        self.assertFalse(res_unknown["recognized"])

    def test_sqlite_database_roundtrip(self):
        """Verify SQLite user creation, template serialization, and retrieval."""
        test_db_path = os.path.join(config.BASE_DIR, "database", "test_palmpay.db")
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        db = Database(test_db_path)
        uid = db.add_user("Test User", "1MS21CS999")
        self.assertIsNotNone(uid)

        # Duplicate USN must be rejected
        uid_dup = db.add_user("Test User 2", "1MS21CS999")
        self.assertIsNone(uid_dup)

        # Add 3 templates
        templates = [
            np.random.randn(944).astype(np.float32) for _ in range(3)
        ]
        templates = [t / np.linalg.norm(t) for t in templates]
        success = db.add_templates(uid, templates, "MB-uLBP-CLAHE-v1")
        self.assertTrue(success)

        # Retrieve and verify
        all_users = db.get_all_users_with_templates()
        self.assertEqual(len(all_users), 1)
        self.assertEqual(all_users[0]["name"], "Test User")
        self.assertEqual(all_users[0]["sample_count"], 3)
        self.assertEqual(len(all_users[0]["templates"]), 3)
        self.assertEqual(all_users[0]["templates"][0].shape, (944,))

        # Verify exact vector deserialization fidelity
        np.testing.assert_allclose(all_users[0]["templates"][0], templates[0], rtol=1e-5)

        # Cleanup test DB
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
