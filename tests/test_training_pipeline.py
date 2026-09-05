"""
Tests for PalmPay DeepPalm Phase-2 Training & Data Flywheel Pipeline.

Covers:
- Consent gating in /api/enroll:
    * DATASET_CAPTURE_ENABLED=False -> no files written
    * DATASET_CAPTURE_ENABLED=True without consent -> no files written
    * DATASET_CAPTURE_ENABLED=True with consent -> 128x128 PNG + JSON saved
- Subject-disjoint split integrity:
    * Train and validation sets share zero overlapping subject identities
- Dataset purge utility (scripts/purge_dataset.py)
- Head-to-head evaluation pipeline (scripts/eval_deep_model.py)
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import uuid
import numpy as np

import config
import app as flask_app
from training.dataset import PalmDataset
from scripts.purge_dataset import purge_dataset
from scripts.eval_deep_model import (
    generate_synthetic_evaluation_rois,
    extract_subject_templates,
    compute_fine_metrics,
    run_model_eval,
)
from core.palm_model import ClassicalPalmModel, DeepPalmModel
from tests.test_payment import _pipeline_result, _unit_vec, _b64_image


class DatasetCaptureConsentTests(unittest.TestCase):
    """
    Verifies that pilot ROI capture is strictly opt-in and consent-gated.
    Raw images are never stored in SQLite or on disk unless both config and user consent pass.
    """

    def setUp(self):
        self.client = flask_app.app.test_client()
        self.db = flask_app.db
        self.tmp_data_dir = tempfile.mkdtemp(prefix="palmpay_test_pilot_")
        self.created_user_ids = []
        self._img = _b64_image()

    def tearDown(self):
        for uid in self.created_user_ids:
            try:
                self.db.delete_user(uid)
            except Exception:
                pass
        self.created_user_ids.clear()

        # Clean up temp test directory
        if os.path.exists(self.tmp_data_dir):
            shutil.rmtree(self.tmp_data_dir, ignore_errors=True)

    def _unique_usn(self) -> str:
        return f"TC{uuid.uuid4().hex[:8].upper()}"

    def test_capture_disabled_by_default_saves_no_files(self):
        """When DATASET_CAPTURE_ENABLED is False, no files are written even if consent is True."""
        usn = self._unique_usn()
        mock_roi = np.full((128, 128), 120, dtype=np.uint8)
        pipe_res = _pipeline_result(_unit_vec(10))
        pipe_res["roi_enhanced"] = mock_roi

        with patch.object(config, "DATASET_CAPTURE_ENABLED", False), \
             patch.object(config, "DATASET_DIR", self.tmp_data_dir), \
             patch("app.pipeline.process_frame", return_value=pipe_res):

            resp = self.client.post(
                "/api/enroll",
                data=json.dumps({
                    "name": "Consent Test Alice",
                    "usn": usn,
                    "samples": [self._img] * 4,
                    "dataset_consent": True,
                }),
                content_type="application/json",
            )

        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.created_user_ids.append(data["user_id"])

        # Assert no directories or files created
        files_saved = os.listdir(self.tmp_data_dir)
        self.assertEqual(len(files_saved), 0, "No files should be saved when capture flag is OFF")

    def test_capture_enabled_without_consent_saves_no_files(self):
        """When DATASET_CAPTURE_ENABLED is True but user does NOT consent, no files are saved."""
        usn = self._unique_usn()
        mock_roi = np.full((128, 128), 120, dtype=np.uint8)
        pipe_res = _pipeline_result(_unit_vec(11))
        pipe_res["roi_enhanced"] = mock_roi

        with patch.object(config, "DATASET_CAPTURE_ENABLED", True), \
             patch.object(config, "DATASET_DIR", self.tmp_data_dir), \
             patch("app.pipeline.process_frame", return_value=pipe_res):

            resp = self.client.post(
                "/api/enroll",
                data=json.dumps({
                    "name": "Consent Test Bob",
                    "usn": usn,
                    "samples": [self._img] * 4,
                    "dataset_consent": False,  # No consent
                }),
                content_type="application/json",
            )

        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.created_user_ids.append(data["user_id"])

        files_saved = os.listdir(self.tmp_data_dir)
        self.assertEqual(len(files_saved), 0, "No files should be saved without explicit consent")

    def test_capture_enabled_with_consent_saves_roi_and_metadata(self):
        """When DATASET_CAPTURE_ENABLED is True AND user consents, 128x128 PNG and JSON meta are saved."""
        usn = self._unique_usn()
        mock_roi = np.random.default_rng(42).integers(50, 200, (128, 128), dtype=np.uint8)
        pipe_res = _pipeline_result(_unit_vec(12))
        pipe_res["roi_enhanced"] = mock_roi

        with patch.object(config, "DATASET_CAPTURE_ENABLED", True), \
             patch.object(config, "DATASET_DIR", self.tmp_data_dir), \
             patch("app.pipeline.process_frame", return_value=pipe_res):

            resp = self.client.post(
                "/api/enroll",
                data=json.dumps({
                    "name": "Consent Test Charlie",
                    "usn": usn,
                    "samples": [self._img] * 4,
                    "dataset_consent": True,
                }),
                content_type="application/json",
            )

        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        uid = data["user_id"]
        self.created_user_ids.append(uid)

        # Verify folder structure
        user_folder = os.path.join(self.tmp_data_dir, f"user_{uid}_{usn}")
        self.assertTrue(os.path.isdir(user_folder), f"Expected folder {user_folder}")

        files = sorted(os.listdir(user_folder))
        png_files = [f for f in files if f.endswith(".png")]
        json_files = [f for f in files if f.endswith(".json")]

        # 4 samples enrolled -> 4 PNGs + 4 JSONs
        self.assertEqual(len(png_files), 4)
        self.assertEqual(len(json_files), 4)

        # Inspect first JSON metadata
        meta_path = os.path.join(user_folder, json_files[0])
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["user_id"], uid)
        self.assertEqual(meta["usn"], usn)
        self.assertEqual(meta["roi_size"], [128, 128])
        self.assertIn("condition_prompt", meta)
        self.assertIn("timestamp", meta)

        # Verify DB privacy: SQLite has NO raw images or paths
        user = self.db.get_user_by_id(uid)
        self.assertNotIn("image", user)
        self.assertNotIn("roi", user)


class SubjectDisjointSplitTests(unittest.TestCase):
    """
    Verifies biometric training integrity: no subject identities bleed between train and validation.
    """

    def test_zero_subject_leakage_between_train_and_val(self):
        ds_mgr = PalmDataset(data_dir="/dev/null")
        mock_subjects = {
            f"user_{i:03d}": [
                {"image_path": f"user_{i}/sample_{j}.png", "meta_path": f"user_{i}/sample_{j}.json", "subject_id": f"user_{i:03d}"}
                for j in range(1, 5)
            ]
            for i in range(1, 11)  # 10 subjects
        }

        train_samples, val_samples = ds_mgr.split_subject_disjoint(mock_subjects, val_ratio=0.3, seed=42)

        train_subj_ids = {s["subject_id"] for s in train_samples}
        val_subj_ids = {s["subject_id"] for s in val_samples}

        # Subject disjoint assertion
        overlap = train_subj_ids.intersection(val_subj_ids)
        self.assertEqual(len(overlap), 0, f"Subject leakage detected between train and val: {overlap}")

        # Total count integrity
        self.assertEqual(len(train_samples) + len(val_samples), 40)
        self.assertEqual(len(train_subj_ids) + len(val_subj_ids), 10)


class PurgeDatasetUtilityTests(unittest.TestCase):
    """
    Verifies that scripts/purge_dataset.py cleanly wipes ROI files.
    """

    def test_purge_dataset_removes_all_files(self):
        tmp_dir = tempfile.mkdtemp(prefix="palmpay_purge_test_")
        sub_dir = os.path.join(tmp_dir, "user_01_TEST")
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "sample_01.png"), "wb") as f:
            f.write(b"dummy_png")
        with open(os.path.join(sub_dir, "sample_01.json"), "w") as f:
            f.write('{"test": true}')

        # Run purge
        count = purge_dataset(target_dir=tmp_dir, force=True)
        self.assertEqual(count, 1)
        self.assertTrue(os.path.isdir(tmp_dir))
        self.assertEqual(len(os.listdir(tmp_dir)), 0)

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)


class EvalScriptMetricsTests(unittest.TestCase):
    """
    Verifies evaluation metrics computation, EER, and zero-FAR logic.
    """

    def test_synthetic_evaluation_generation(self):
        data = generate_synthetic_evaluation_rois(num_subjects=3, samples_per_subject=4, seed=123)
        self.assertEqual(len(data), 3)
        for subj, rois in data.items():
            self.assertEqual(len(rois), 4)
            self.assertEqual(rois[0].shape, (128, 128))
            self.assertEqual(rois[0].dtype, np.uint8)

    def test_compute_fine_metrics_eer_and_zero_far(self):
        # Genuine scores well separated from impostors
        gen = [0.95, 0.96, 0.97, 0.98, 0.99]
        imp = [0.80, 0.82, 0.84, 0.85, 0.86]

        metrics = compute_fine_metrics(gen, imp)
        self.assertIsNotNone(metrics["eer"])
        self.assertAlmostEqual(metrics["eer"]["eer_value"], 0.0, places=2)
        self.assertEqual(metrics["zero_far_gar"], 1.0)
        self.assertGreater(metrics["zero_far_threshold"], 0.86)

    def test_run_model_eval_classical_and_deep(self):
        rois_dict = generate_synthetic_evaluation_rois(num_subjects=2, samples_per_subject=2, seed=99)
        c_model = ClassicalPalmModel()
        d_model = DeepPalmModel()

        c_res = run_model_eval("Classical", c_model, rois_dict)
        d_res = run_model_eval("Deep", d_model, rois_dict)

        self.assertEqual(c_res["status"], "success")
        self.assertEqual(d_res["status"], "success")
        self.assertEqual(c_res["genuine_comparisons"], 2)  # 2 subjects * 1 pair each = 2
        self.assertEqual(c_res["impostor_comparisons"], 4) # 2 * 2 = 4 cross pairs


if __name__ == "__main__":
    unittest.main()
