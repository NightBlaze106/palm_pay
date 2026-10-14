"""
Unified Biometric Pipeline.
Orchestrates:
Raw Frame -> Hand Detection -> Palm ROI Extraction -> Affine Normalization -> Preprocessing -> Feature Extraction.
"""

import base64
import os
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

import config
from core.geometry import extract_geometry_features
from core.hand_detector import HandDetector
from core.roi import PalmROIExtractor
from core.preprocessing import PalmPreprocessor
from core.palm_model import get_palm_model, build_classical_model


def apply_finger_mask(image: np.ndarray, knuckle_y) -> Tuple[np.ndarray, Dict]:
    """
    Finger Masking (Accuracy Sprint v3).

    Replaces all rows ABOVE the MCP knuckle line with a neutral fill so finger
    spread / pose variation stops leaking into block histograms. A feathered
    (Gaussian-blurred binary) mask avoids hard artifact rows at the boundary.

    Args:
        image: uint8 array — 2D grayscale (H, W) or 3D color (H, W, 3).
        knuckle_y: knuckle cut-line y in ROI coordinates (from PalmROIExtractor),
                   already offset by config.FINGER_MASK_OFFSET_PX. None disables.

    Returns:
        (masked_image_uint8, info_dict)
        info: {"applied": bool, "cut_y": int|None, "fill_value": float|None,
               "skipped": str|None}
    """
    info = {"applied": False, "cut_y": None, "fill_value": None, "skipped": None}

    if not config.FINGER_MASK_ENABLED:
        info["skipped"] = "disabled"
        return image.copy(), info
    if knuckle_y is None:
        info["skipped"] = "no_knuckle_line"
        return image.copy(), info

    h, w = image.shape[:2]
    cut_y = int(round(float(knuckle_y)))
    # Sane bounds: below this the "fingers" region is negligible; above it we
    # would be masking nearly the whole palm.
    if cut_y < 20 or cut_y > h - 4:
        info["skipped"] = f"cut_out_of_bounds({cut_y})"
        return image.copy(), info

    fill_value = float(np.mean(image[cut_y:, ...]))

    soft = np.ones((h, w), dtype=np.float32)
    soft[:cut_y, :] = 0.0
    k = int(config.FINGER_MASK_FEATHER_SIGMA * 4) | 1  # odd kernel size
    soft = cv2.GaussianBlur(soft, (k, k), sigmaX=config.FINGER_MASK_FEATHER_SIGMA)

    mask3 = soft[..., np.newaxis] if image.ndim == 3 else soft
    out = image.astype(np.float32) * mask3 + fill_value * (1.0 - mask3)
    # Round (not floor): keeps fully-masked-on regions byte-exact despite
    # float32 epsilon in the blurred mask.
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)

    info.update({"applied": True, "cut_y": cut_y, "fill_value": round(fill_value, 1)})
    return out, info


def evaluate_palm_sharpness(
    gray_roi: np.ndarray,
    min_tenengrad: float = config.QUALITY_MIN_TENENGRAD,
    min_patch_laplacian: float = config.QUALITY_MIN_PATCH_LAPLACIAN,
    fallback_blur_var: float = config.QUALITY_MIN_BLUR_VAR,
) -> Tuple[bool, Dict, str]:
    """
    Evaluates palm ROI sharpness using noise-resistant Tenengrad gradient energy
    and 3x3 patch-based Laplacian variance.

    Returns:
        (passed, metrics_dict, message)
    """
    if gray_roi is None or gray_roi.size == 0:
        return False, {"tenengrad": 0.0, "patch_laplacian_q25": 0.0, "blur_var": 0.0}, "Empty ROI"

    blur_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

    # Tenengrad focus measure (Sobel gradient magnitude energy)
    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    # 3x3 patch Laplacian distribution (detects local blur across central palm)
    h, w = gray_roi.shape[:2]
    ph, pw = max(1, h // 3), max(1, w // 3)
    patch_vars = []
    for r in range(3):
        for c in range(3):
            patch = gray_roi[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            if patch.size > 0:
                patch_vars.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))

    patch_q25 = float(np.percentile(patch_vars, 25)) if patch_vars else blur_var

    metrics = {
        "blur_var": round(blur_var, 1),
        "tenengrad": round(tenengrad, 1),
        "patch_laplacian_q25": round(patch_q25, 1),
    }

    # Pass condition: either dual-metric passes OR global blur_var meets fallback
    passed = (tenengrad >= min_tenengrad and patch_q25 >= min_patch_laplacian) or (blur_var >= fallback_blur_var)
    msg = "Sharpness OK." if passed else "Image too blurry — hold palm steady."
    return passed, metrics, msg


def estimate_palm_distance(palm_breadth_px: float) -> Tuple[float, str, str]:
    """
    Estimates physical distance from camera lens using anthropometric palm breadth.
    Formula: Z_cm = (DISTANCE_PALM_BREADTH_MM * DISTANCE_FOCAL_PX) / (palm_breadth_px * 10)
    Returns:
        (distance_cm, zone, coaching_message)
        zone: "too_far" | "optimal" | "too_close" | "unknown"
    """
    if palm_breadth_px is None or palm_breadth_px <= 5.0:
        return 999.0, "too_far", "Move palm closer to camera."

    k_constant = (config.DISTANCE_PALM_BREADTH_MM / 10.0) * config.DISTANCE_FOCAL_PX
    dist_cm = round(k_constant / palm_breadth_px, 1)

    if palm_breadth_px < config.DISTANCE_MIN_BREADTH_PX:
        return dist_cm, "too_far", "Move palm closer to camera."
    elif palm_breadth_px > config.DISTANCE_MAX_BREADTH_PX:
        return dist_cm, "too_close", "Move palm slightly back."
    else:
        return dist_cm, "optimal", "Ideal distance — hold steady."


class FrameQualityGate:
    """
    Capture Quality Gate (Accuracy Sprint).

    Rejects low-quality palm ROIs BEFORE feature extraction instead of scoring
    them. Garbage frames (blur, glare, darkness, palm-too-far) produce noisy
    embeddings that can randomly spike above threshold — blocking them here
    cuts both false accepts and false rejects.
    """

    def __init__(
        self,
        min_blur_var: float = config.QUALITY_MIN_BLUR_VAR,
        max_glare_fraction: float = config.QUALITY_MAX_GLARE_FRACTION,
        min_brightness: int = config.QUALITY_MIN_MEAN_BRIGHTNESS,
        max_brightness: int = config.QUALITY_MAX_MEAN_BRIGHTNESS,
        min_palm_span: float = config.QUALITY_MIN_PALM_SPAN,
        min_tenengrad: float = config.QUALITY_MIN_TENENGRAD,
        min_patch_laplacian: float = config.QUALITY_MIN_PATCH_LAPLACIAN,
    ):
        self.min_blur_var = float(min_blur_var)
        self.max_glare_fraction = float(max_glare_fraction)
        self.min_brightness = int(min_brightness)
        self.max_brightness = int(max_brightness)
        self.min_palm_span = float(min_palm_span)
        self.min_tenengrad = float(min_tenengrad)
        self.min_patch_laplacian = float(min_patch_laplacian)

    def check(
        self,
        raw_gray_roi: np.ndarray,
        palm_span_px: float,
        palm_breadth_px: Optional[float] = None,
        is_edge_clipped: bool = False,
    ) -> Dict:
        """
        Args:
            raw_gray_roi: un-enhanced grayscale ROI (CLAHE would mask glare).
            palm_span_px: estimated palm physical span from ROI extraction.
            palm_breadth_px: measured MCP knuckle breadth in pixels.
            is_edge_clipped: whether key palm landmarks are clipped at frame boundary.

        Returns:
            {"passed": bool, "reason": str|None, "metrics": dict, "message": str}
        """
        metrics = {
            "blur_var": None,
            "tenengrad": None,
            "patch_laplacian_q25": None,
            "glare_fraction": None,
            "mean_brightness": None,
            "palm_span": round(float(palm_span_px), 1),
            "palm_breadth": round(float(palm_breadth_px), 1) if palm_breadth_px else None,
            "distance_cm": None,
            "distance_zone": "unknown",
        }

        if raw_gray_roi is None or raw_gray_roi.size == 0:
            return {"passed": False, "reason": "empty_roi", "metrics": metrics,
                    "message": "Empty ROI frame."}

        if is_edge_clipped:
            return {"passed": False, "reason": "edge_clipped", "metrics": metrics,
                    "message": "Hand too close to edge — center palm in frame."}

        if palm_span_px < self.min_palm_span:
            return {"passed": False, "reason": "too_far",
                    "metrics": metrics,
                    "message": "Palm too far from camera — move closer."}

        # Distance estimation
        if palm_breadth_px is not None and palm_breadth_px > 0:
            dist_cm, dist_zone, _ = estimate_palm_distance(palm_breadth_px)
            metrics["distance_cm"] = dist_cm
            metrics["distance_zone"] = dist_zone
            if dist_zone == "too_close":
                return {"passed": False, "reason": "too_close", "metrics": metrics,
                        "message": "Palm too close — move palm slightly back."}
            elif dist_zone == "too_far":
                return {"passed": False, "reason": "too_far", "metrics": metrics,
                        "message": "Palm too far from camera — move closer into sweet spot (20–33 cm)."}

        gray = np.asarray(raw_gray_roi)
        mean_brightness = float(np.mean(gray))
        glare_fraction = float(np.mean(gray > 245))

        # Sharpness evaluation
        sharpness_passed, sharp_metrics, sharp_msg = evaluate_palm_sharpness(
            gray,
            min_tenengrad=self.min_tenengrad,
            min_patch_laplacian=self.min_patch_laplacian,
            fallback_blur_var=self.min_blur_var,
        )
        metrics.update({
            "blur_var": sharp_metrics["blur_var"],
            "tenengrad": sharp_metrics["tenengrad"],
            "patch_laplacian_q25": sharp_metrics["patch_laplacian_q25"],
            "glare_fraction": round(glare_fraction, 4),
            "mean_brightness": round(mean_brightness, 1),
        })

        if mean_brightness < self.min_brightness:
            return {"passed": False, "reason": "too_dark", "metrics": metrics,
                    "message": "Too dark — improve lighting on the palm."}
        if mean_brightness > self.max_brightness:
            return {"passed": False, "reason": "too_bright", "metrics": metrics,
                    "message": "Over-exposed — reduce direct light."}
        if glare_fraction > self.max_glare_fraction:
            return {"passed": False, "reason": "glare", "metrics": metrics,
                    "message": "Glare detected — tilt palm slightly away from light."}
        if not sharpness_passed:
            return {"passed": False, "reason": "blurry", "metrics": metrics,
                    "message": sharp_msg}

        return {"passed": True, "reason": None, "metrics": metrics,
                "message": "Capture quality OK."}


def build_model():
    if getattr(config, 'DEEP_ENABLED', False):
        from core.palm_model import DeepPalmModel
        m = DeepPalmModel()
        print(f"[PalmPay] Active model: {m.model_name} ({m.feature_dim}-D) [deep]")
        return m

    active = getattr(config, 'ACTIVE_MODEL', 'classical')

    if active == 'ccnet':
        path = config.CCNET_MODEL_PATH
        resolved_path = path if os.path.isabs(path) else os.path.join(config.BASE_DIR, path)
        if os.path.exists(resolved_path) or os.path.exists(path):
            actual_path = resolved_path if os.path.exists(resolved_path) else path
            from core.palm_model import CcnetOnnxModel
            m = CcnetOnnxModel(actual_path)
            print(f"[PalmPay] Active model: CCNet ONNX fp32 "
                  f"({m.feature_dim}-D) — {actual_path}")
            return m
        else:
            print(f"[PalmPay] WARNING: CCNet model not found at {path}. "
                  f"Falling back to classical.")
            active = 'classical'

    if active == 'dinov2':
        variant = getattr(config, 'DINOV2_VARIANT', 'dinov2_vits14')
        try:
            from core.palm_model import DinoV2Model
            m = DinoV2Model(variant)
            print(f"[PalmPay] Active model: DINOv2 {variant} "
                  f"({m.feature_dim}-D)")
            return m
        except Exception as e:
            print(f"[PalmPay] WARNING: DINOv2 failed ({e}). "
                  f"Falling back to classical.")
            active = 'classical'

    # classical fallback
    m = build_classical_model()   # existing call, unchanged
    print(f"[PalmPay] Active model: {m.model_name} "
          f"({m.feature_dim}-D) [classical]")
    return m


class BiometricPipeline:
    """
    Complete end-to-end contactless palm recognition pipeline.
    """

    def __init__(self):
        self.hand_detector = HandDetector()
        self.roi_extractor = PalmROIExtractor()
        self.preprocessor = PalmPreprocessor()
        self.model = build_model()
        self.quality_gate = FrameQualityGate()

    def reload_model(self):
        """Reloads the active palm biometric model according to current configuration."""
        self.model = build_model()
        return self.model

    def _sync_model(self):
        """Ensures self.model reflects current configuration state."""
        self.reload_model()

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
                - embedding: np.ndarray (FEATURE_VECTOR_DIM,) or None
                - geometry: np.ndarray (GEO_FEATURE_DIM,) or None (hand shape descriptor)
                - geometry_ok: bool
                - model_name: str
                - feature_dim: int
                - latency_ms: float
                - preview_roi_bgr_b64: base64 string or ""
                - preview_enhanced_b64: base64 string or ""
                - preview_annotated_b64: base64 string or ""
                - message: status explanation
        """
        start_time = time.perf_counter()
        self._sync_model()

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
                "distance_cm": None,
                "distance_zone": "unknown",
                "palm_center": None,
                "palm_breadth_px": 0.0,
                "in_sweet_spot": False,
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
                "distance_cm": None,
                "distance_zone": "no_hand",
                "palm_center": None,
                "palm_breadth_px": 0.0,
                "in_sweet_spot": False,
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "message": det_result["message"],
            }

        # Stage 1.5: Hand geometry features (Accuracy Sprint v4).
        # Landmark-derived shape descriptor — computed whenever a hand is
        # detected; consumed by the matcher's geometry gate and stored at
        # enrollment. Independent of ROI/quality stages.
        geo_result = extract_geometry_features(det_result["landmarks_pixel"])

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
                "distance_cm": None,
                "distance_zone": "too_far",
                "palm_center": roi_result.get("center"),
                "palm_breadth_px": roi_result.get("palm_breadth_px", 0.0),
                "in_sweet_spot": False,
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
                "distance_cm": None,
                "distance_zone": "unknown",
                "palm_center": roi_result.get("center"),
                "palm_breadth_px": roi_result.get("palm_breadth_px", 0.0),
                "in_sweet_spot": False,
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "message": prep_result["message"],
            }

        # Stage 3.5: Capture Quality Gate — reject garbage frames before scoring.
        quality = self.quality_gate.check(
            prep_result["gray"],
            palm_span_px=roi_result.get("scale_px", 0.0),
            palm_breadth_px=roi_result.get("palm_breadth_px", 0.0),
            is_edge_clipped=det_result.get("is_edge_clipped", False),
        )
        if not quality["passed"]:
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
                "distance_cm": quality["metrics"].get("distance_cm"),
                "distance_zone": quality["metrics"].get("distance_zone", "unknown"),
                "palm_center": roi_result.get("center", (0, 0)),
                "palm_breadth_px": roi_result.get("palm_breadth_px", 0.0),
                "in_sweet_spot": False,
                "preview_roi_bgr_b64": "",
                "preview_enhanced_b64": "",
                "preview_annotated_b64": "",
                "failure_stage": "quality",
                "quality_reason": quality["reason"],
                "quality_metrics": quality["metrics"],
                "message": quality["message"],
            }

        # Stage 4: Feature Extraction (multi-scale MB-uLBP vector)
        #
        # Stage 3.6 first — Finger Masking: rows above the MCP knuckle line are
        # replaced with a neutral fill (feathered edge) on BOTH the feature
        # input and the ROI preview, so the UI shows the exclusion band and
        # enroll/recognize stay consistent by construction.
        enhanced_for_features = prep_result["enhanced"]
        roi_preview_bgr = roi_result["roi_bgr"]
        finger_mask_info = {"applied": False, "cut_y": None}
        if roi_result.get("knuckle_y_roi") is not None:
            enhanced_for_features, finger_mask_info = apply_finger_mask(
                prep_result["enhanced"], roi_result["knuckle_y_roi"]
            )
            if finger_mask_info.get("applied"):
                roi_preview_bgr, _ = apply_finger_mask(
                    roi_preview_bgr, roi_result["knuckle_y_roi"]
                )

        embedding = self.model.extract_embedding(enhanced_for_features)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Optional UI previews
        preview_roi_b64 = ""
        preview_enhanced_b64 = ""
        preview_annotated_b64 = ""

        if include_visuals:
            preview_roi_b64 = self.encode_image_to_base64(roi_preview_bgr)
            preview_enhanced_b64 = self.encode_image_to_base64(enhanced_for_features)

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
            "geometry": geo_result.get("geometry"),
            "geometry_ok": bool(geo_result.get("success")),
            "model_name": self.model.get_model_name(),
            "feature_dim": self.model.get_feature_dim(),
            "latency_ms": round(elapsed_ms, 2),
            "distance_cm": quality["metrics"].get("distance_cm"),
            "distance_zone": quality["metrics"].get("distance_zone", "optimal"),
            "palm_center": roi_result.get("center", (0, 0)),
            "palm_breadth_px": roi_result.get("palm_breadth_px", 0.0),
            "in_sweet_spot": bool(quality["metrics"].get("distance_zone") == "optimal"),
            "quality_reason": None,
            "quality_metrics": quality["metrics"],
            "finger_mask_applied": bool(finger_mask_info.get("applied")),
            "finger_mask_cut_y": finger_mask_info.get("cut_y"),
            "roi_enhanced": enhanced_for_features,
            "preview_roi_bgr_b64": preview_roi_b64,
            "preview_enhanced_b64": preview_enhanced_b64,
            "preview_annotated_b64": preview_annotated_b64,
            "message": "Palm biometric features extracted successfully.",
        }

    def close(self):
        """Clean up detector resources."""
        self.hand_detector.close()
