"""
Hand Detection Module using MediaPipe Hand Landmarker (Tasks API).
Extracts 21 3D hand landmarks from RGB/BGR webcam frames.
"""

import os
import urllib.request
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import config


class HandDetector:
    """
    Lightweight Hand Landmark Detector wrapping MediaPipe Tasks HandLandmarker.
    Provides pixel and normalized coordinates for 21 hand landmarks.
    """

    # Key landmark indices for palm geometry
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    PALM_BASE_LANDMARKS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    # Hand skeleton connections for visualization
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky finger
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm knuckle bridge
        (5, 9), (9, 13), (13, 17)
    ]

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(
        self,
        model_path: str = config.HAND_LANDMARKER_MODEL_PATH,
        min_detection_confidence: float = config.HAND_DETECTION_CONFIDENCE,
        max_num_hands: int = config.MAX_NUM_HANDS,
    ):
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.max_num_hands = max_num_hands
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Ensures the hand landmarker task file is present locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            print(f"Downloading MediaPipe hand landmarker model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)

    def detect(self, image_bgr: np.ndarray) -> Dict:
        """
        Detects hand landmarks in a BGR frame.

        Args:
            image_bgr: Input image in OpenCV BGR format.

        Returns:
            Dict containing:
                - detected: bool
                - landmarks_pixel: list of (x, y, z) in image pixel coordinates
                - landmarks_normalized: list of (x, y, z) in normalized [0, 1] range
                - handedness: 'Left' | 'Right' | 'Unknown'
                - score: detection confidence score
                - message: status explanation
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "detected": False,
                "landmarks_pixel": [],
                "landmarks_normalized": [],
                "handedness": "Unknown",
                "score": 0.0,
                "message": "Empty or invalid image frame.",
            }

        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        results = self.landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return {
                "detected": False,
                "landmarks_pixel": [],
                "landmarks_normalized": [],
                "handedness": "Unknown",
                "score": 0.0,
                "message": "No hand detected. Please position your palm clearly in front of the camera.",
            }

        # Select primary detected hand
        primary_landmarks = results.hand_landmarks[0]
        handedness_label = "Unknown"
        score = 0.90

        if results.handedness and len(results.handedness) > 0 and len(results.handedness[0]) > 0:
            handedness_label = results.handedness[0][0].category_name
            score = float(results.handedness[0][0].score)

        landmarks_pixel = []
        landmarks_normalized = []

        for lm in primary_landmarks:
            px_x = int(np.clip(lm.x * w, 0, w - 1))
            px_y = int(np.clip(lm.y * h, 0, h - 1))
            landmarks_pixel.append((px_x, px_y, float(lm.z)))
            landmarks_normalized.append((float(lm.x), float(lm.y), float(lm.z)))

        return {
            "detected": True,
            "landmarks_pixel": landmarks_pixel,
            "landmarks_normalized": landmarks_normalized,
            "handedness": handedness_label,
            "score": score,
            "message": "Hand landmarks successfully extracted.",
        }

    def draw_landmarks(
        self,
        image_bgr: np.ndarray,
        landmarks_normalized: Optional[List[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        Draws 21 hand landmarks and skeleton connections on a copy of the frame.
        """
        annotated = image_bgr.copy()
        if not landmarks_normalized or len(landmarks_normalized) < 21:
            return annotated

        h, w, _ = annotated.shape
        points_px = []
        for x, y, _ in landmarks_normalized:
            px = int(np.clip(x * w, 0, w - 1))
            py = int(np.clip(y * h, 0, h - 1))
            points_px.append((px, py))

        # Draw skeleton lines
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            cv2.line(
                annotated,
                points_px[start_idx],
                points_px[end_idx],
                color=(0, 215, 255),  # Amber gold
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Draw landmark joints
        for idx, (px, py) in enumerate(points_px):
            # Special color for palm landmarks
            if idx in self.PALM_BASE_LANDMARKS:
                cv2.circle(annotated, (px, py), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)  # Green for palm
                cv2.circle(annotated, (px, py), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(annotated, (px, py), 4, (0, 140, 255), -1, lineType=cv2.LINE_AA)  # Orange for fingers

        return annotated

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
