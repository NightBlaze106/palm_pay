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
GAUSSIAN_BLUR_KERNEL = (3, 3)

# Feature Extraction Configuration: Multi-Block Uniform LBP (MB-uLBP_{8,1}^u2)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_GRID_X = 4
LBP_GRID_Y = 4
# 58 uniform patterns + 1 non-uniform pattern = 59 bins per block
LBP_BINS_PER_BLOCK = 59
# Total feature vector length = 4 * 4 * 59 = 944 dimensions
FEATURE_VECTOR_DIM = LBP_GRID_X * LBP_GRID_Y * LBP_BINS_PER_BLOCK
MODEL_NAME = "MB-uLBP-CLAHE-v1"

# Biometric Matching Configuration
# NOTE: This is an initial experimental default value for testing, NOT an academically validated threshold.
# Calibration must be conducted on empirical samples via the /test evaluation module.
DEFAULT_RECOGNITION_THRESHOLD = 0.78

# Enrollment Configuration
DEFAULT_ENROLLMENT_SAMPLES = 4
MIN_ENROLLMENT_SAMPLES = 3
MAX_ENROLLMENT_SAMPLES = 8

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
DEBUG_MODE = True
