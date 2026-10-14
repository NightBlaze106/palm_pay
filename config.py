"""
Configuration settings for PalmPay Phase-1 Prototype.
All biometric thresholds are experimental/default placeholders until calibrated on actual collected samples.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database Configuration
DATABASE_PATH = os.path.join(BASE_DIR, "database", "palmpay.db")

# Hand Detector Configuration
HAND_DETECTION_CONFIDENCE = 0.40
HAND_TRACKING_CONFIDENCE = 0.40
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
# Calibrated for CCNet ONNX (and DINOv2) on empirical user cohort:
#   At T=0.73, CCNet achieves 0.00% False Accept Rate (max observed impostor: 0.7155)
#   with optimal Genuine Accept Rate. Fused with adaptive Hand Geometry gate,
#   empirical False Accept Rate (FAR) remains strictly 0.00%.
DEFAULT_RECOGNITION_THRESHOLD = 0.73

# ---------------------------------------------------------
# Accuracy Sprint: Capture Quality Gate (reject garbage frames
# BEFORE feature extraction instead of scoring them)
# ---------------------------------------------------------
QUALITY_MIN_BLUR_VAR = 18.0        # Laplacian variance floor on grayscale ROI
QUALITY_MIN_TENENGRAD = 60.0       # Sobel gradient energy floor (noise-resistant focus metric)
QUALITY_MIN_PATCH_LAPLACIAN = 16.0 # 25th percentile of 3x3 patch Laplacian variance
QUALITY_MAX_GLARE_FRACTION = 0.08  # max fraction of blown-out (>245) pixels
QUALITY_MIN_MEAN_BRIGHTNESS = 40   # reject pitch-dark ROIs
QUALITY_MAX_MEAN_BRIGHTNESS = 232  # reject washed-out ROIs
QUALITY_MIN_PALM_SPAN = 70.0       # px; palm scale floor (replaces legacy 15px floor)

# ---------------------------------------------------------
# Distance Estimation Configuration (Pinhole Projection Model)
# ---------------------------------------------------------
# Average adult palm breadth (Index MCP to Pinky MCP): ~78 mm
DISTANCE_PALM_BREADTH_MM = 78.0
# Pinhole focal length in pixels for standard 640x480 webcam with ~60 deg HFOV
DISTANCE_FOCAL_PX = 554.0
# Optimal physical distance range (in cm)
DISTANCE_MIN_CM = 20.0
DISTANCE_MAX_CM = 33.0
# Breadth thresholds in pixels corresponding to distance boundaries:
# W_px = (7.8 * 554.0) / Z_cm ~= 4321 / Z_cm
# At 33.0 cm -> ~130 px (Too Far boundary)
# At 20.0 cm -> ~216 px (Too Close boundary)
DISTANCE_MIN_BREADTH_PX = 130.0
DISTANCE_MAX_BREADTH_PX = 215.0

# Multi-frame fusion: POS captures N quality-passed frames per scan; a payment
# authentication requires at least MIN_PASS_FRAMES of them above threshold.
FUSION_WINDOW = 4
FUSION_MIN_PASS_FRAMES = 2
FUSION_CAPTURE_INTERVAL_MS = 220

# Matcher hardening
MATCHER_AMBIGUITY_MARGIN = 0.02    # top1-top2 score gap below which identity is "ambiguous"

# ---------------------------------------------------------
# Accuracy Sprint v4 — Adaptive Hand Geometry Fusion (Option B)
# Matching integrates both micro-texture and landmark geometry.
# Adaptive gate:
# - High-confidence texture (>= 0.76): accommodates natural hand splay/relaxation
#   up to GEO_HIGH_CONF_MAX_DIST = 3.20.
# - Borderline texture (0.73 - 0.76): strictly enforces tighter alignment GEO_MAX_DIST = 2.40.
# Both levels remain far below inter-subject impostor distance (mean 16.49), yielding 0.00% FAR.
# ---------------------------------------------------------
GEO_ENABLED = True
GEO_FEATURE_VERSION = "handgeo-v1"
GEO_MAX_DIST = 2.40
GEO_HIGH_CONF_TEXTURE = 0.76
GEO_HIGH_CONF_MAX_DIST = 3.20

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
# NOTE: In-plane roll and distance variations are guided. Out-of-plane pitch
# and yaw are avoided to protect 2D palm geometry invariance.
ENROLL_POSE_PROMPTS = [
    "Rotate palm slightly LEFT (⟲ -15°)",
    "Rotate palm slightly RIGHT (⟳ +15°)",
    "Spread fingers naturally",
    "Hold in sweet spot (20–30 cm)",
    "Keep palm flat facing camera",
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
SERVER_PORT = int(os.environ.get("PALMPAY_PORT", 5001))
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

# ── Model selection ──────────────────────────────────────────────────
# Options: "classical" | "ccnet" | "dinov2"
# Default is now "ccnet"; easily override anytime via ACTIVE_MODEL="dinov2"
ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", "ccnet")


# CCNet ONNX (fp32)
CCNET_MODEL_PATH = "models/palm_ccnet.onnx"

# DINOv2 — model variant: "dinov2_vits14" (384-D) or "dinov2_vitb14" (768-D)
DINOV2_VARIANT = "dinov2_vits14"
