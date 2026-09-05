"""
Configuration settings for PalmPay Phase-1 Prototype.
All biometric thresholds are experimental/default placeholders until calibrated on actual collected samples.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database Configuration
DATABASE_PATH = os.path.join(BASE_DIR, "database", "palmpay.db")

# Hand Detector Configuration
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1
HAND_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

# Palm ROI Extraction Configuration
ROI_WIDTH = 128
ROI_HEIGHT = 128
ROI_SIZE = (ROI_WIDTH, ROI_HEIGHT)
# Scale factor relative to palm landmark span (wrist to middle MCP / index to pinky MCP)
ROI_SCALE_FACTOR = 1.05

# Preprocessing Configuration
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# Blur DISABLED (Accuracy Sprint v2): Gaussian smoothing erased the fine
# micro-texture that distinguishes palms; impostors scored ~0.80 with it on.
# Noise is instead handled by the capture quality gate + multi-frame fusion.
GAUSSIAN_BLUR_KERNEL = (0, 0)

# =========================================================
# Feature Extraction Model Selection
# =========================================================
# To switch models, uncomment the desired block. Stale templates are automatically 
# ignored by the database when MODEL_NAME changes.

# --- Option B: HOG + MB-uLBP Fusion [ACTIVE] ---
FEATURE_VECTOR_DIM = 3652
MODEL_NAME = "HogLbpFusion-v1-fm"

# --- Option D: Deep Palm Embedder (Phase 2 Deep Model) ---
# FEATURE_VECTOR_DIM = 512
# MODEL_NAME = "deep-palm-embedder-v1"

# --- Option A: Competitive Coding (CompCode) ---
# FEATURE_VECTOR_DIM = 1536
# MODEL_NAME = "CompCode-v1-fm"

# --- Option C: Legacy MB-uLBP ---
# FEATURE_VECTOR_DIM = 1888
# MODEL_NAME = "MB-uLBP2R-CLAHE-v2-fm"

# --- Model Hyperparameters (Kept active for tests and extractors) ---
# CompCode
COMPCODE_GRID_X = 16
COMPCODE_GRID_Y = 16
COMPCODE_ORIENTATIONS = 6
# MB-uLBP
LBP_POINTS = 8
LBP_RADII = (1, 2)
LBP_GRID_X = 4
LBP_GRID_Y = 4
LBP_BINS_PER_BLOCK = 59

# ---------------------------------------------------------
# Accuracy Sprint v3 — Finger Masking
# Rows ABOVE the MCP knuckle line in the normalized ROI are replaced with a
# neutral fill before feature extraction, so finger spread / pose stops
# leaking into block histograms. Cut direction: y grows downward, wrist is at
# the bottom of the ROI, fingers toward the top.
# ---------------------------------------------------------
FINGER_MASK_ENABLED = True
FINGER_MASK_OFFSET_PX = 0     # +N keeps more upper-palm area; -N cuts higher
FINGER_MASK_FEATHER_SIGMA = 3.0  # soft edge to avoid hard artifact rows

# Biometric Matching Configuration
# Calibrated 2026-08-23 on 4-user cohort (24 genuine / 72 impostor trials):
#   texture impostor mean-template scores reach 0.988 -> T=0.95 alone is NOT
#   safe; geometry gate carries the discrimination. Joint operating point
#   (texture dual-pass AND geometry <= GEO_MAX_DIST): FAR 1.4%, GAR 67%
#   (limited by the aryan<->punith similar-shape pair; re-enroll and
#   recalibrate after enrollment-quality fixes).
DEFAULT_RECOGNITION_THRESHOLD = 0.95

# ---------------------------------------------------------
# Accuracy Sprint: Capture Quality Gate (reject garbage frames
# BEFORE feature extraction instead of scoring them)
# ---------------------------------------------------------
QUALITY_MIN_BLUR_VAR = 40.0        # Laplacian variance floor on grayscale ROI
QUALITY_MAX_GLARE_FRACTION = 0.08  # max fraction of blown-out (>245) pixels
QUALITY_MIN_MEAN_BRIGHTNESS = 40   # reject pitch-dark ROIs
QUALITY_MAX_MEAN_BRIGHTNESS = 232  # reject washed-out ROIs
QUALITY_MIN_PALM_SPAN = 60.0       # px; replaces the old 15px "too far" floor

# Multi-frame fusion: POS captures N quality-passed frames per scan; a payment
# authentication requires at least MIN_PASS_FRAMES of them above threshold.
FUSION_WINDOW = 4
FUSION_MIN_PASS_FRAMES = 3
FUSION_CAPTURE_INTERVAL_MS = 220

# Matcher hardening
MATCHER_AMBIGUITY_MARGIN = 0.02    # top1-top2 score gap below which identity is "ambiguous"

# ---------------------------------------------------------
# Accuracy Sprint v4 — Hand Geometry Fusion
# LBP texture alone cannot separate palms on contactless RGB webcam input
# (measured impostor distribution 0.88-0.99 across a 9-user cohort). Hand
# geometry (landmark-derived shape ratios + signed splay angles) is a
# complementary, illumination-immune trait. Matching requires BOTH signals
# to pass (AND-fusion) — see core/geometry.py and core/matcher.py.
# ---------------------------------------------------------
GEO_ENABLED = True
GEO_FEATURE_VERSION = "handgeo-v1"
# Max RMS normalized deviation for a candidate's geometry to qualify.
# INTERIM value calibrated on the 4-user cohort (zero impostor passes except
# the aryan<->punith similar-shape pair at 1.014). Tighten/loosen only via
# /api/evaluate geometry_eval numbers — never by intuition.
GEO_MAX_DIST = 1.5

# Evaluation policy: suggested operating point requires ZERO observed impostor
# accepts while keeping at least this genuine accept rate.
EVAL_SUGGESTION_MIN_GAR = 0.90

# Enrollment Configuration
DEFAULT_ENROLLMENT_SAMPLES = 6     # auto-captured, condition-varied (fingerprint-style)
MIN_ENROLLMENT_SAMPLES = 3
MAX_ENROLLMENT_SAMPLES = 8
ENROLL_CAPTURE_COOLDOWN_MS = 1300  # pause between auto-captured samples
# Condition script for the 6 auto-captured samples. Sample 1 is a neutral
# baseline; each subsequent sample gets one prompt.
# NOTE (Sprint v4): pitch/foreshortening prompts were REMOVED — out-of-plane
# rotation changes projected hand geometry (not removable by 2D invariance)
# and was measured to wreck geometry-template consistency. Keep poses flat.
ENROLL_POSE_PROMPTS = [
    "Tilt your palm slightly LEFT",
    "Tilt your palm slightly RIGHT",
    "Move a little CLOSER",
    "Spread your fingers the same amount",
    "Keep the palm FLAT facing the camera",
]
# Enrollment pose-consistency gate (geometry): samples whose hand-geometry
# deviates more than this RMS distance from the enrollment's median sample are
# dropped; the enrollment is rejected when fewer than MIN_ENROLLMENT_SAMPLES
# survive. Enforces a flat, repeatable pose envelope so live scans compare
# like-for-like (session drift was the dominant error source in calibration).
ENROLL_GEO_CONSISTENCY_MAX_DIST = 1.5
# Near-duplicate guard: candidate captures whose downscaled grayscale mean
# difference vs the previous sample is below this percentage are skipped.
ENROLL_DUPLICATE_DIFF_PCT = 1.5

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5001
DEBUG_MODE = True

# ---------------------------------------------------------
# PalmPay Payment System (Demo / Prototype) Configuration
# ---------------------------------------------------------

# Admin panel credentials (demo only — hardcoded per Phase-2 prototype spec)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "palmpay2025"

# Biometric payment session: recognition result is trusted for this long (server-side)
BIOMETRIC_SESSION_TTL_SECONDS = 30

# Payment amount limits (INR)
MAX_PAYMENT_AMOUNT = 5000

# Hardcoded demo merchants (POS selector). Each merchant maps to a wallet-backed
# user row via USN prefix "MERCH-" + upper-case id (e.g. "m1" -> usn "MERCH-M1").
MERCHANTS = [
    {"id": "m1", "name": "Canteen",    "category": "Food"},
    {"id": "m2", "name": "Photocopy",  "category": "Services"},
    {"id": "m3", "name": "Stationery", "category": "Retail"},
    {"id": "m4", "name": "Library",    "category": "Services"},
]

MERCHANT_USN_PREFIX = "MERCH-"

# ====================================================================
# Phase-2 Feature Flags & Configurations (Disabled by default)
# ====================================================================

# Deep Learning Palm Embedder Pipeline
DEEP_ENABLED = False
DEEP_MODEL_PATH = "models/palm_embedder.onnx"
DEEP_MODEL_NAME = "deep-palm-embedder-v1"
DEEP_FEATURE_DIM = 512
DEEP_INPUT_SIZE = 128
# Decision threshold for deep 512-D embeddings. When >0.0, PalmMatcher uses this value;
# 0.0 is an uncalibrated placeholder that falls back to DEFAULT_RECOGNITION_THRESHOLD.
DEEP_MATCH_THRESHOLD = 0.0
# Reserved scaffolding for future multi-model failover (currently DeepPalmModel
# self-contains a deterministic 512-D offline fallback to avoid vector dimension mismatches).
DEEP_FALLBACK_TO_CLASSIC = False

# Challenge-Response Liveness Detection
LIVENESS_ENABLED = False
PAYMENT_REQUIRES_LIVENESS = False
LIVENESS_CHALLENGE_POOL = ["OPEN_PALM", "FIST", "SPREAD"]
LIVENESS_CHALLENGE_LENGTH = 2
LIVENESS_WINDOW_MS = 7000
LIVENESS_MIN_FRAMES = 8
LIVENESS_MAX_FRAMES = 18
LIVENESS_TOKEN_TTL_S = 30
# Minimum articulation/transition score [0.0, 1.0] required to pass liveness.
# When >0.0, verify_sequence enforces this floor; 0.0 is placeholder allowing all valid sequences.
LIVENESS_MIN_SCORE = 0.0

# Native C++ Accelerated Core
NATIVE_CPP_ENABLED = False
NATIVE_CPP_BIN = "cpp/build/palmpay_native"

# ---------------------------------------------------------
# Phase-2 Training & Data Flywheel: Consent-Gated Dataset Capture
# ---------------------------------------------------------
# Ground Truth: Raw palm images are never stored in the database.
# When enabled AND explicit user consent is checked during enrollment,
# processed 128x128 CLAHE palm ROIs + JSON metadata are saved ephemerally
# to disk for model training. Default is strictly OFF.
DATASET_CAPTURE_ENABLED = False
DATASET_DIR = os.path.join(BASE_DIR, "data", "pilot_rois")

