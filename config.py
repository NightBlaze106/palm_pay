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

# Feature Extraction Configuration: Multi-Block Uniform LBP
LBP_POINTS = 8
LBP_RADII = (1, 2)                 # multi-scale: R=1 fine texture + R=2 contextual texture
LBP_GRID_X = 4
LBP_GRID_Y = 4
# 58 uniform patterns + 1 non-uniform pattern = 59 bins per block per scale
LBP_BINS_PER_BLOCK = 59
# Total feature vector length = scales(2) * blocks(16) * bins(59) = 1888 dimensions
FEATURE_VECTOR_DIM = LBP_GRID_X * LBP_GRID_Y * LBP_BINS_PER_BLOCK * len(LBP_RADII)
# "-fm" suffix: finger-masked variant. Templates from older model names are
# automatically excluded by the DB loader, forcing a clean re-enrollment.
MODEL_NAME = "MB-uLBP2R-CLAHE-v2-fm"

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
# NOTE: This is an initial experimental default value for testing, NOT an academically validated threshold.
# Calibration must be conducted on empirical samples via the /test evaluation module.
DEFAULT_RECOGNITION_THRESHOLD = 0.78

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

# Evaluation policy: suggested operating point requires ZERO observed impostor
# accepts while keeping at least this genuine accept rate.
EVAL_SUGGESTION_MIN_GAR = 0.90

# Enrollment Configuration
DEFAULT_ENROLLMENT_SAMPLES = 6     # auto-captured, condition-varied (fingerprint-style)
MIN_ENROLLMENT_SAMPLES = 3
MAX_ENROLLMENT_SAMPLES = 8
ENROLL_CAPTURE_COOLDOWN_MS = 1300  # pause between auto-captured samples
# Condition script for the 6 auto-captured samples. Sample 1 is a neutral
# baseline; each subsequent sample gets one prompt. Lighting + pitch entries
# directly target the room-to-hall generalization failure.
ENROLL_POSE_PROMPTS = [
    "Tilt your palm slightly LEFT",
    "Tilt your palm slightly RIGHT",
    "Move a little CLOSER",
    "Turn your palm TOWARD the light",
    "Tip the top of your hand AWAY from the camera",
]
# Near-duplicate guard: candidate captures whose downscaled grayscale mean
# difference vs the previous sample is below this percentage are skipped.
ENROLL_DUPLICATE_DIFF_PCT = 1.5

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
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
