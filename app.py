"""
PalmPay — Contactless Palm Biometric Authentication System (Phase 1 Prototype)
Flask Backend Application and REST API.
"""

from datetime import datetime
import io
import os
import time
from flask import Flask, Response, jsonify, render_template, request

import config
from database.db import Database
from core.pipeline import BiometricPipeline
from core.matcher import PalmMatcher
from core.evaluation import EvaluationEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = "palmpay-dev-secret-key"

# Initialize global components
db = Database(config.DATABASE_PATH)
pipeline = BiometricPipeline()
matcher = PalmMatcher(threshold=config.DEFAULT_RECOGNITION_THRESHOLD)
evaluation_engine = EvaluationEngine(matcher=matcher)


# ---------------------------------------------------------
# Page Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    """Home / Overview Page."""
    stats = db.get_stats()
    return render_template(
        "index.html",
        stats=stats,
        threshold=matcher.threshold,
        model_name=config.MODEL_NAME,
        feature_dim=config.FEATURE_VECTOR_DIM,
    )


@app.route("/enroll")
def enroll_page():
    """Palm Enrollment Page."""
    return render_template(
        "enroll.html",
        required_samples=config.DEFAULT_ENROLLMENT_SAMPLES,
        min_samples=config.MIN_ENROLLMENT_SAMPLES,
        max_samples=config.MAX_ENROLLMENT_SAMPLES,
    )


@app.route("/recognize")
def recognize_page():
    """Real-Time 1:N Palm Recognition Page."""
    stats = db.get_stats()
    return render_template(
        "recognize.html",
        threshold=matcher.threshold,
        user_count=stats["total_users"],
        model_name=config.MODEL_NAME,
    )


@app.route("/users")
def users_page():
    """Enrolled Users Management Page."""
    users = db.get_all_users_with_templates()
    stats = db.get_stats()
    return render_template(
        "users.html",
        users=users,
        stats=stats,
    )


@app.route("/test")
def test_page():
    """Evaluation & Threshold Calibration Page."""
    stats = db.get_stats()
    return render_template(
        "test.html",
        stats=stats,
        threshold=matcher.threshold,
    )


# ---------------------------------------------------------
# Biometric REST API Endpoints
# ---------------------------------------------------------

@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Real-time frame probe for live camera HUD visualizer.
    Receives single base64 frame -> returns hand detection, landmarks, and ROI polygon.
    """
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"success": False, "message": "No image data received."}), 400

    frame_bgr = pipeline.decode_base64_to_image(image_b64)
    if frame_bgr is None:
        return jsonify({"success": False, "message": "Failed to decode frame."}), 400

    # Run biometric pipeline on incoming frame
    result = pipeline.process_frame(frame_bgr, include_visuals=True)

    # Return serializable summary
    return jsonify(
        {
            "success": result["success"],
            "hand_detected": result["hand_detected"],
            "landmarks_detected": result["landmarks_detected"],
            "roi_extracted": result["roi_extracted"],
            "roi_polygon": result["roi_polygon"],
            "latency_ms": result["latency_ms"],
            "preview_roi_b64": result["preview_roi_bgr_b64"],
            "preview_enhanced_b64": result["preview_enhanced_b64"],
            "preview_annotated_b64": result["preview_annotated_b64"],
            "message": result["message"],
        }
    )


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """
    Enrolls a new user with 3-5 palm sample frames.
    Validates identity, extracts 944-D embeddings for all samples, and stores in SQLite.
    """
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    usn = str(data.get("usn", "")).strip().upper()
    samples_b64 = data.get("samples", [])

    if not name or not usn:
        return jsonify({"success": False, "message": "Name and USN / Student ID are required."}), 400

    if not isinstance(samples_b64, list) or len(samples_b64) < config.MIN_ENROLLMENT_SAMPLES:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"At least {config.MIN_ENROLLMENT_SAMPLES} palm samples are required for reliable enrollment.",
                }
            ),
            400,
        )

    # Check for duplicate USN
    existing_user = db.get_user_by_usn(usn)
    if existing_user:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"A user with USN '{usn}' is already enrolled ({existing_user['name']}).",
                }
            ),
            409,
        )

    extracted_embeddings = []
    latencies = []

    for idx, s_b64 in enumerate(samples_b64):
        frame_bgr = pipeline.decode_base64_to_image(s_b64)
        if frame_bgr is None:
            return (
                jsonify({"success": False, "message": f"Sample #{idx + 1} could not be decoded."}),
                400,
            )

        res = pipeline.process_frame(frame_bgr, include_visuals=False)
        if not res["success"] or res["embedding"] is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Sample #{idx + 1} failed biometric extraction: {res['message']}. Please position palm steadily.",
                    }
                ),
                400,
            )

        extracted_embeddings.append(res["embedding"])
        latencies.append(res["latency_ms"])

    # Create user in database
    user_id = db.add_user(name, usn)
    if not user_id:
        return jsonify({"success": False, "message": "Failed to create user record in database."}), 500

    # Store templates
    db_success = db.add_templates(user_id, extracted_embeddings, model_name=config.MODEL_NAME)
    if not db_success:
        db.delete_user(user_id)
        return jsonify({"success": False, "message": "Failed to save biometric templates."}), 500

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return jsonify(
        {
            "success": True,
            "message": f"Successfully enrolled {name} ({usn}) with {len(extracted_embeddings)} palm samples.",
            "user_id": user_id,
            "name": name,
            "usn": usn,
            "sample_count": len(extracted_embeddings),
            "model_name": config.MODEL_NAME,
            "feature_dim": config.FEATURE_VECTOR_DIM,
            "avg_extraction_latency_ms": avg_latency,
        }
    )


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    Performs 1:N Biometric Authentication on query frame.
    """
    start_total = time.perf_counter()
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"success": False, "message": "No query image provided."}), 400

    frame_bgr = pipeline.decode_base64_to_image(image_b64)
    if frame_bgr is None:
        return jsonify({"success": False, "message": "Failed to decode query frame."}), 400

    # Step 1: Run Biometric Extraction Pipeline
    pipeline_res = pipeline.process_frame(frame_bgr, include_visuals=True)
    if not pipeline_res["success"] or pipeline_res["embedding"] is None:
        total_lat = (time.perf_counter() - start_total) * 1000.0
        return jsonify(
            {
                "success": False,
                "recognized": False,
                "hand_detected": pipeline_res["hand_detected"],
                "roi_extracted": pipeline_res["roi_extracted"],
                "matched_user": None,
                "similarity_score": 0.0,
                "threshold_used": matcher.threshold,
                "feature_extraction_latency_ms": pipeline_res["latency_ms"],
                "matching_latency_ms": 0.0,
                "total_latency_ms": round(total_lat, 2),
                "preview_roi_b64": pipeline_res["preview_roi_bgr_b64"],
                "preview_enhanced_b64": pipeline_res["preview_enhanced_b64"],
                "preview_annotated_b64": pipeline_res["preview_annotated_b64"],
                "message": pipeline_res["message"],
            }
        )

    # Step 2: 1:N Template Matching against Enrolled Database
    t_match_start = time.perf_counter()
    enrolled_users = db.get_all_users_with_templates()
    match_res = matcher.match_1_to_n(pipeline_res["embedding"], enrolled_users)
    matching_lat = (time.perf_counter() - t_match_start) * 1000.0
    total_lat = (time.perf_counter() - start_total) * 1000.0

    return jsonify(
        {
            "success": True,
            "recognized": match_res["recognized"],
            "hand_detected": True,
            "roi_extracted": True,
            "matched_user": match_res["matched_user"],
            "best_candidate": match_res.get("best_candidate"),
            "similarity_score": match_res["similarity_score"],
            "threshold_used": match_res["threshold_used"],
            "all_scores": match_res["all_scores"][:5],  # top 5
            "feature_extraction_latency_ms": pipeline_res["latency_ms"],
            "matching_latency_ms": round(matching_lat, 2),
            "total_latency_ms": round(total_lat, 2),
            "preview_roi_b64": pipeline_res["preview_roi_bgr_b64"],
            "preview_enhanced_b64": pipeline_res["preview_enhanced_b64"],
            "preview_annotated_b64": pipeline_res["preview_annotated_b64"],
            "message": match_res["message"],
        }
    )


@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Lists all enrolled users."""
    users = db.get_all_users_with_templates()
    safe_users = []
    for u in users:
        safe_users.append(
            {
                "id": u["id"],
                "name": u["name"],
                "usn": u["usn"],
                "sample_count": u["sample_count"],
                "created_at": u["created_at"],
            }
        )
    return jsonify({"success": True, "users": safe_users, "total": len(safe_users)})


@app.route("/api/users/<int:user_id>/delete", methods=["POST", "DELETE"])
def api_delete_user(user_id: int):
    """Deletes an enrolled user and their templates."""
    success = db.delete_user(user_id)
    if success:
        return jsonify({"success": True, "message": f"User #{user_id} successfully deleted."})
    return jsonify({"success": False, "message": "Failed to delete user."}), 400


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Gets or sets biometric configuration and threshold."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_threshold = data.get("threshold")
        if new_threshold is not None:
            try:
                t_val = float(new_threshold)
                if 0.0 <= t_val <= 1.0:
                    matcher.set_threshold(t_val)
                    return jsonify(
                        {
                            "success": True,
                            "message": f"Threshold updated to {t_val:.4f}",
                            "threshold": matcher.threshold,
                        }
                    )
                else:
                    return jsonify({"success": False, "message": "Threshold must be between 0.0 and 1.0."}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid threshold number."}), 400

    return jsonify(
        {
            "threshold": matcher.threshold,
            "model_name": config.MODEL_NAME,
            "feature_dim": config.FEATURE_VECTOR_DIM,
            "roi_size": list(config.ROI_SIZE),
            "clahe_clip_limit": config.CLAHE_CLIP_LIMIT,
            "default_threshold": config.DEFAULT_RECOGNITION_THRESHOLD,
        }
    )


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Executes empirical biometric evaluation across enrolled templates.
    Computes genuine vs impostor scores, GAR, FAR, FRR, and suggested threshold.
    """
    users_data = db.get_all_users_with_templates()
    eval_result = evaluation_engine.evaluate_enrolled_dataset(users_data)
    return jsonify(eval_result)


@app.route("/api/export_csv", methods=["GET"])
def api_export_csv():
    """Generates and downloads the evaluation CSV report."""
    users_data = db.get_all_users_with_templates()
    eval_result = evaluation_engine.evaluate_enrolled_dataset(users_data)
    csv_content = evaluation_engine.generate_csv_report(eval_result)

    filename = f"palmpay_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"},
    )


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PalmPay — Contactless Palm Biometric Authentication")
    print("Target Environment: Intel i3 / Windows CPU / RGB Camera")
    print(f"Local Server: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print("=" * 60)
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False,
    )
