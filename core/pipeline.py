"""
Unified Biometric Pipeline.
Orchestrates:
Raw Frame -> Hand Detection -> Palm ROI Extraction -> Affine Normalization -> Preprocessing -> Feature Extraction.
"""

import base64
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

import config
from core.hand_detector import HandDetector
from core.roi import PalmROIExtractor
from core.preprocessing import PalmPreprocessor
from core.palm_model import get_palm_model


class BiometricPipeline:
    """
    Complete end-to-end contactless palm recognition pipeline.
    """

    def __init__(self):
        self.hand_detector = HandDetector()
        self.roi_extractor = PalmROIExtractor()
        self.preprocessor = PalmPreprocessor()
        self.model = get_palm_model()

    @staticmethod
    def encode_image_to_base64(image_array: np.ndarray, format_ext: str = ".jpg") -> str:
        """Converts an OpenCV image array to a base64 data URI string."""
        if image_array is None or image_array.size == 0:
            return ""
        success, buffer = cv2.imencode(format_ext, image_array)
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/jpeg" if format_ext in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime};base64,{b64_str}"

    @staticmethod
    def decode_base64_to_image(base64_data: str) -> Optional[np.ndarray]:
        """Decodes a base64 data URI string into an OpenCV BGR numpy array."""
        try:
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image_bgr
        except Exception:
            return None

    def process_frame(
        self,
        image_bgr: np.ndarray,
        include_visuals: bool = True,
    ) -> Dict:
        """
        Executes the full biometric pipeline on an input frame.

        Args:
            image_bgr: Source OpenCV BGR image.
            include_visuals: If True, returns annotated base64 preview images.

        Returns:
            Dict containing:
                - success: bool
                - hand_detected: bool
                - landmarks_detected: bool
                - landmarks_count: int
                - roi_extracted: bool
                - roi_polygon: list of (x, y) coordinates
                - embedding: np.ndarray (944,) or None
                - model_name: str
                - feature_dim: int
                - latency_ms: float
                - preview_roi_bgr_b64: base64 string or ""
                - preview_enhanced_b64: base64 string or ""
                - preview_annotated_b64: base64 string or ""
                - message: status explanation
        """
        start_time = time.perf_counter()

        if image_bgr is None or image_bgr.size == 0:
            return {
                "success": False,
                "hand_detected": False,
                "landmarks_detected": False,
                "landmarks_count": 0,
                "roi_extracted": False,
                "roi_polygon": [],
                "embedding": None,
                "model_name": self.model.get_model_name(),
                "feature_dim": self.model.get_feature_dim(),
                "latency_ms": 0.0,
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "message": "Invalid or empty image frame.",
            }

        # Stage 1: Hand Detection & 21 Landmarks
        det_result = self.hand_detector.detect(image_bgr)
        if not det_result["detected"]:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "hand_detected": False,
                "landmarks_detected": False,
                "landmarks_count": 0,
                "roi_extracted": False,
                "roi_polygon": [],
                "embedding": None,
                "model_name": self.model.get_model_name(),
                "feature_dim": self.model.get_feature_dim(),
                "latency_ms": round(elapsed_ms, 2),
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "message": det_result["message"],
            }

        # Stage 2: Palm ROI Extraction & Affine Normalization
        roi_result = self.roi_extractor.extract_roi(
            image_bgr,
            det_result["landmarks_pixel"],
        )
        if not roi_result["success"]:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "hand_detected": True,
                "landmarks_detected": True,
                "landmarks_count": len(det_result["landmarks_pixel"]),
                "roi_extracted": False,
                "roi_polygon": [],
                "embedding": None,
                "model_name": self.model.get_model_name(),
                "feature_dim": self.model.get_feature_dim(),
                "latency_ms": round(elapsed_ms, 2),
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "message": roi_result["message"],
            }

        # Stage 3: Preprocessing & CLAHE Enhancement
        prep_result = self.preprocessor.preprocess(roi_result["roi_bgr"])
        if not prep_result["success"]:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "hand_detected": True,
                "landmarks_detected": True,
                "landmarks_count": len(det_result["landmarks_pixel"]),
                "roi_extracted": True,
                "roi_polygon": roi_result["roi_polygon"],
                "embedding": None,
                "model_name": self.model.get_model_name(),
                "feature_dim": self.model.get_feature_dim(),
                "latency_ms": round(elapsed_ms, 2),
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "message": prep_result["message"],
            }

        # Stage 4: Feature Extraction (MB-uLBP 944-D Vector)
        embedding = self.model.extract_embedding(prep_result["enhanced"])

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Optional UI previews
        preview_roi_b64 = ""
        preview_enhanced_b64 = ""
        preview_annotated_b64 = ""

        if include_visuals:
            preview_roi_b64 = self.encode_image_to_base64(roi_result["roi_bgr"])
            preview_enhanced_b64 = self.encode_image_to_base64(prep_result["enhanced"])

            # Annotate full frame with landmarks and ROI polygon box
            annotated_frame = self.hand_detector.draw_landmarks(
                image_bgr,
                det_result["landmarks_normalized"],
            )
            annotated_frame = self.roi_extractor.draw_roi_box(
                annotated_frame,
                roi_result["roi_polygon"],
                color=(0, 255, 0),
                thickness=2,
            )
            preview_annotated_b64 = self.encode_image_to_base64(annotated_frame)

        return {
            "success": True,
            "hand_detected": True,
            "landmarks_detected": True,
            "landmarks_count": len(det_result["landmarks_pixel"]),
            "roi_extracted": True,
            "roi_polygon": roi_result["roi_polygon"],
            "embedding": embedding,
            "model_name": self.model.get_model_name(),
            "feature_dim": self.model.get_feature_dim(),
            "latency_ms": round(elapsed_ms, 2),
            "preview_roi_bgr_b64": preview_roi_b64,
            "preview_enhanced_b64": preview_enhanced_b64,
            "preview_annotated_b64": preview_annotated_b64,
            "message": "Palm biometric features extracted successfully.",
        }

    def close(self):
        """Clean up detector resources."""
        self.hand_detector.close()
