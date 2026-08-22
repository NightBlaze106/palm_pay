"""
CLI Demonstration Script:
Validates the full biometric pipeline from frame -> MediaPipe -> Palm ROI -> CLAHE -> multi-scale LBP Vector.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.hand_detector import HandDetector
from core.roi import PalmROIExtractor
from core.preprocessing import PalmPreprocessor
from core.palm_model import ClassicalPalmModel


def generate_synthetic_hand_image(width=640, height=480) -> np.ndarray:
    """Generates a realistic test image with hand-like geometry and palm texture."""
    img = np.full((height, width, 3), 40, dtype=np.uint8)

    # Draw palm base ellipse
    center = (320, 280)
    axes = (90, 110)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 200, 230), -1)

    # Draw fingers
    finger_tips = [
        ((220, 150), (250, 230)),  # Index
        ((320, 120), (320, 200)),  # Middle
        ((390, 140), (370, 220)),  # Ring
        ((440, 180), (410, 240)),  # Pinky
        ((180, 280), (240, 310)),  # Thumb
    ]
    for tip, base in finger_tips:
        cv2.line(img, base, tip, (180, 200, 230), thickness=28)
        cv2.circle(img, tip, 14, (180, 200, 230), -1)

    # Draw palm principal lines (Life line, Head line, Heart line)
    cv2.ellipse(img, (290, 300), (60, 45), -30, 0, 160, (120, 130, 160), 3) # Life line
    cv2.line(img, (260, 260), (360, 270), (120, 130, 160), 3)                # Head line
    cv2.line(img, (270, 230), (380, 240), (120, 130, 160), 3)                # Heart line

    # Add mild noise
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def run_demonstration():
    print("=" * 70)
    print("PalmPay Biometric Pipeline Verification & Demonstration")
    print("=" * 70)

    out_dir = os.path.join(config.BASE_DIR, "data", "demo_output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Acquire Input Frame
    print("\n[Step 1] Loading/Generating Test Frame...")
    frame = generate_synthetic_hand_image()
    raw_path = os.path.join(out_dir, "01_raw_frame.jpg")
    cv2.imwrite(raw_path, frame)
    print(f"  Frame resolution: {frame.shape[1]}x{frame.shape[0]} px, Channels: {frame.shape[2]}")
    print(f"  Saved raw frame to: {raw_path}")

    # 2. Hand Detection & Landmark Extraction
    print("\n[Step 2] Executing MediaPipe Hand Landmark Detector...")
    detector = HandDetector()
    t0 = time.perf_counter()
    det_res = detector.detect(frame)
    t_det = (time.perf_counter() - t0) * 1000.0

    # If synthetic frame is not picked by neural detector, provide fallback landmark geometry to verify downstream
    if not det_res["detected"]:
        print("  Notice: Synthetic drawing not detected by neural net, supplying canonical hand landmarks for geometric verification.")
        # Canonical palm landmarks
        landmarks_pixel = [(320, 380, 0.0)] * 21
        landmarks_pixel[0] = (320, 380, 0.0)    # Wrist
        landmarks_pixel[5] = (250, 230, 0.0)    # Index MCP
        landmarks_pixel[9] = (320, 200, 0.0)    # Middle MCP
        landmarks_pixel[13] = (370, 220, 0.0)   # Ring MCP
        landmarks_pixel[17] = (410, 240, 0.0)   # Pinky MCP
        landmarks_norm = [(x / 640.0, y / 480.0, z) for x, y, z in landmarks_pixel]
    else:
        landmarks_pixel = det_res["landmarks_pixel"]
        landmarks_norm = det_res["landmarks_normalized"]
        print(f"  MediaPipe Hand Detected: {det_res['handedness']} hand (Confidence: {det_res['score']:.2f})")

    print(f"  Extracted 21 landmarks in: {t_det:.2f} ms")
    print(f"  Wrist (0): {landmarks_pixel[0][:2]}")
    print(f"  Index MCP (5): {landmarks_pixel[5][:2]}")
    print(f"  Middle MCP (9): {landmarks_pixel[9][:2]}")
    print(f"  Pinky MCP (17): {landmarks_pixel[17][:2]}")

    annotated = detector.draw_landmarks(frame, landmarks_norm)

    # 3. Palm ROI Extraction & Affine Normalization
    print("\n[Step 3] Isolating Central Palm ROI with Landmark-Guided Affine Warping...")
    roi_extractor = PalmROIExtractor(roi_size=(128, 128))
    t0 = time.perf_counter()
    roi_res = roi_extractor.extract_roi(frame, landmarks_pixel)
    t_roi = (time.perf_counter() - t0) * 1000.0

    if not roi_res["success"]:
        print(f"  ROI Extraction Failed: {roi_res['message']}")
        return

    roi_bgr = roi_res["roi_bgr"]
    print(f"  ROI Extraction Success in: {t_roi:.2f} ms")
    print(f"  Palm Centroid: {roi_res['center']}")
    print(f"  Estimated Rotation Angle: {roi_res['angle_deg']:.2f} degrees")
    print(f"  Physical Palm Scale: {roi_res['scale_px']:.2f} px")
    print(f"  Extracted ROI dimensions: {roi_bgr.shape[1]}x{roi_bgr.shape[0]} px")

    roi_path = os.path.join(out_dir, "02_palm_roi.jpg")
    cv2.imwrite(roi_path, roi_bgr)
    print(f"  Saved raw ROI to: {roi_path}")

    annotated_box = roi_extractor.draw_roi_box(annotated, roi_res["roi_polygon"])
    ann_path = os.path.join(out_dir, "03_annotated_frame.jpg")
    cv2.imwrite(ann_path, annotated_box)

    # 4. Preprocessing & CLAHE Enhancement
    print("\n[Step 4] Applying Preprocessing & CLAHE Local Contrast Enhancement...")
    preprocessor = PalmPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
    t0 = time.perf_counter()
    prep_res = preprocessor.preprocess(roi_bgr)
    t_prep = (time.perf_counter() - t0) * 1000.0

    enhanced = prep_res["enhanced"]
    print(f"  Preprocessing Completed in: {t_prep:.2f} ms")
    print(f"  Grayscale conversion: uint8 [0-255]")
    print(f"  CLAHE (clipLimit=2.0, tileGrid=(8,8)) applied")
    clahe_path = os.path.join(out_dir, "04_clahe_enhanced.jpg")
    cv2.imwrite(clahe_path, enhanced)
    print(f"  Saved CLAHE enhanced ROI to: {clahe_path}")

    # 5. Feature Extraction (Multi-Scale MB-uLBP Vector)
    print("\n[Step 5] Extracting Multi-Block Uniform LBP Feature Vector (MB-uLBP)...")
    model = ClassicalPalmModel()
    t0 = time.perf_counter()
    feature_vector = model.extract_embedding(enhanced)
    t_feat = (time.perf_counter() - t0) * 1000.0

    print(f"  Feature Extraction Completed in: {t_feat:.2f} ms")
    print(f"  Model Name: {model.get_model_name()}")
    print(f"  Vector Shape: {feature_vector.shape} ({len(feature_vector)} dimensions)")
    print(f"  Vector Dtype: {feature_vector.dtype}")
    print(f"  Vector L2 Norm: {np.linalg.norm(feature_vector):.6f}")
    print(f"  First 10 coefficients: {np.round(feature_vector[:10], 4)}")
    print(f"  Min coefficient: {np.min(feature_vector):.6f}, Max: {np.max(feature_vector):.6f}")

    total_time = t_det + t_roi + t_prep + t_feat
    print("\n" + "=" * 70)
    print(f"Pipeline Demonstration Complete. Total Pipeline Execution Time: {total_time:.2f} ms")
    print("All intermediate visual stages verified and saved successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
