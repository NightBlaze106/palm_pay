#!/usr/bin/env python3
"""
Controlled End-to-End Deep Model Smoke Test.
Verifies the DeepPalm integration:
1. Enrollment with DEEP_ENABLED=True stores model_name='deep-palm-embedder-v1', feature_dim=512
2. Template loader retrieves 512-D templates
3. Recognition reaches DeepPalm matcher and verifies match / rejection paths
4. Cleans up all test state and confirms DEEP_ENABLED remains False
"""

import base64
import os
import sys
import uuid
from unittest.mock import patch
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import app as flask_app
from core.geometry import GEO_FEATURE_DIM
from database.db import Database


def _make_dummy_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def run_smoke_test():
    print("=" * 72)
    print("PALMPAY PHASE-2 DEEP MODEL CONTROLLED END-TO-END SMOKE TEST")
    print("=" * 72)
    print("Integration verification: tests deep embedding pipeline routing,")
    print("database schema isolation, and matcher threshold decision path.")
    print("-" * 72)

    client = flask_app.app.test_client()
    db = flask_app.db

    orig_deep = config.DEEP_ENABLED
    orig_match_t = config.DEEP_MATCH_THRESHOLD

    test_uid = None
    try:
        # Step 1: Temporarily enable DeepPalm in controlled test context
        print("[Step 1] Enabling DEEP_ENABLED=True for test...")
        config.DEEP_ENABLED = True
        config.DEEP_MATCH_THRESHOLD = 0.85
        flask_app.pipeline.reload_model()
        assert flask_app.pipeline.model.get_model_name() == "deep-palm-embedder-v1"
        assert flask_app.pipeline.model.get_feature_dim() == 512
        print(f"         Active model: {flask_app.pipeline.model.get_model_name()} ({flask_app.pipeline.model.get_feature_dim()}-D) ✓")

        # Step 2: Enroll ONE test subject
        usn = f"SMOKE_{uuid.uuid4().hex[:8].upper()}"
        print(f"[Step 2] Enrolling test subject (USN: {usn})...")
        samples = [_make_dummy_image() for _ in range(6)]

        # Synthetic 512-D embedding
        rng = np.random.default_rng(12345)
        raw_emb = rng.standard_normal(512).astype(np.float32)
        target_emb = raw_emb / np.linalg.norm(raw_emb)
        geo_vec = np.linspace(0.5, 1.5, GEO_FEATURE_DIM, dtype=np.float32)

        mock_res = {
            "success": True,
            "hand_detected": True,
            "landmarks_detected": True,
            "landmarks_count": 21,
            "roi_extracted": True,
            "roi_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "embedding": target_emb,
            "geometry": geo_vec,
            "geometry_ok": True,
            "model_name": "deep-palm-embedder-v1",
            "feature_dim": 512,
            "latency_ms": 2.5,
            "preview_roi_bgr_b64": "",
            "preview_enhanced_b64": "",
            "preview_annotated_b64": "",
            "message": "Palm biometric features extracted successfully.",
        }

        with patch.object(flask_app.pipeline, "process_frame", return_value=mock_res):
            enroll_res = client.post("/api/enroll", json={
                "name": "Deep Smoke Subject",
                "usn": usn,
                "samples": samples,
            })
            assert enroll_res.status_code == 200, f"Enrollment failed: {enroll_res.get_json()}"
            enroll_data = enroll_res.get_json()
            test_uid = enroll_data["user_id"]
            print(f"         Enrolled user_id={test_uid}, response model_name={enroll_data['model_name']}, feature_dim={enroll_data['feature_dim']} ✓")

        # Step 3: Verify DB template row
        print("[Step 3] Verifying database templates table schema...")
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT feature_dim, model_name FROM templates WHERE user_id = ?", (test_uid,))
        rows = cursor.fetchall()
        conn.close()
        assert len(rows) > 0, "No templates saved in DB"
        for r in rows:
            assert r["model_name"] == "deep-palm-embedder-v1", f"Expected deep-palm-embedder-v1, got {r['model_name']}"
            assert r["feature_dim"] == 512, f"Expected 512, got {r['feature_dim']}"
        print(f"         Verified {len(rows)} templates in DB: model_name='deep-palm-embedder-v1', feature_dim=512 ✓")

        # Step 4: Verify active template loader
        print("[Step 4] Verifying db.get_all_users_with_templates() filtering...")
        enrolled_users = db.get_all_users_with_templates()
        found_target = next((u for u in enrolled_users if u["id"] == test_uid), None)
        assert found_target is not None, "Template loader did not return deep-enrolled user"
        assert len(found_target["templates"]) == len(rows)
        assert all(len(t) == 512 for t in found_target["templates"])
        print(f"         Template loader found target user with {len(found_target['templates'])} x 512-D templates ✓")

        # Step 5: Run recognition for same subject
        print("[Step 5] Running recognition with genuine query...")
        with patch.object(flask_app.pipeline, "process_frame", return_value=mock_res):
            rec_res = client.post("/api/recognize", json={"images": samples[:4]})
            rec_data = rec_res.get_json()
            assert rec_data["recognized"] is True, f"Recognition failed: {rec_data}"
            assert rec_data["matched_user"]["id"] == test_uid
            print(f"         Recognized: {rec_data['matched_user']['name']} (score: {rec_data['similarity_score']:.4f} >= threshold: {rec_data['threshold_used']}) ✓")

        # Step 6 & 7: Confirm deep threshold/rejection path is used
        print("[Step 6 & 7] Running recognition with impostor/orthogonal query...")
        impostor_emb = rng.standard_normal(512).astype(np.float32)
        impostor_emb = impostor_emb - np.dot(impostor_emb, target_emb) * target_emb  # Orthogonalize
        impostor_emb /= np.linalg.norm(impostor_emb)
        mock_impostor = dict(mock_res, embedding=impostor_emb)

        with patch.object(flask_app.pipeline, "process_frame", return_value=mock_impostor):
            rec_res2 = client.post("/api/recognize", json={"images": samples[:4]})
            rec_data2 = rec_res2.get_json()
            assert rec_data2["recognized"] is False
            assert rec_data2.get("match_reason") == "deep_below_threshold", f"Expected deep_below_threshold, got {rec_data2.get('match_reason')}"
            assert rec_data2["threshold_used"] == 0.85
            print(f"         Rejection verified: match_reason='{rec_data2.get('match_reason')}', threshold_used={rec_data2['threshold_used']} ✓")

    finally:
        # Step 8: Clean up and revert DEEP_ENABLED back OFF
        print("[Step 8] Cleaning up test data and reverting flags...")
        if test_uid:
            db.delete_user(test_uid)
            print(f"         Deleted test user {test_uid} ✓")

        config.DEEP_ENABLED = orig_deep
        config.DEEP_MATCH_THRESHOLD = orig_match_t
        flask_app.pipeline.reload_model()
        print(f"         DEEP_ENABLED reverted to: {config.DEEP_ENABLED} ✓")
        print(f"         Active model reverted to: {flask_app.pipeline.model.get_model_name()} ({flask_app.pipeline.model.get_feature_dim()}-D) ✓")

    print("-" * 72)
    print("RESULT: DEEP MODEL ARCHITECTURE & INTEGRATION SMOKE TEST PASSED.")
    print("NOTE: This test verifies the end-to-end software integration, database")
    print("isolation, and matcher routing. It does NOT assert real-world neural")
    print("recognition accuracy. Production biometric evaluation requires real palm")
    print("ROIs collected via consent-gated pilot enrollment.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
