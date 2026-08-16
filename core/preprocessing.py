"""
Palm Image Preprocessing Module.
Applies Grayscale conversion, Noise Filtering, and Local Contrast Enhancement (CLAHE)
to the isolated palm ROI. The exact same pipeline is strictly applied to both
enrollment and recognition phases.
"""

from typing import Dict, Tuple
import cv2
import numpy as np

import config


class PalmPreprocessor:
    """
    Modular preprocessor for extracted palm Region of Interest (ROI).
    Applies CLAHE local contrast enhancement to emphasize epidermal ridges and
    principal flexion lines (life line, head line, heart line) under ambient lighting.
    """

    def __init__(
        self,
        clip_limit: float = config.CLAHE_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = config.CLAHE_TILE_GRID_SIZE,
        blur_kernel: Tuple[int, int] = config.GAUSSIAN_BLUR_KERNEL,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blur_kernel = blur_kernel
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def preprocess(self, roi_bgr: np.ndarray) -> Dict:
        """
        Processes a raw BGR palm ROI image.

        Args:
            roi_bgr: Cropped BGR palm ROI image (e.g., 128x128x3).

        Returns:
            Dict containing:
                - success: bool
                - gray: Raw grayscale ROI (uint8, 0-255)
                - enhanced: CLAHE-enhanced grayscale ROI (uint8, 0-255)
                - normalized: Float representation in [0.0, 1.0]
                - message: Status description
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return {
                "success": False,
                "gray": None,
                "enhanced": None,
                "normalized": None,
                "message": "Empty ROI provided for preprocessing.",
            }

        # 1. Grayscale Conversion
        if len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_bgr.copy()

        # 2. Mild Gaussian Smoothing to suppress high-frequency webcam sensor noise
        if self.blur_kernel[0] > 0 and self.blur_kernel[1] > 0:
            smoothed = cv2.GaussianBlur(gray, self.blur_kernel, sigmaX=0.5)
        else:
            smoothed = gray

        # 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        # Emphasizes fine palm creases without over-amplifying background noise
        enhanced = self.clahe.apply(smoothed)

        # 4. Normalized float array for feature extractors requiring float input
        normalized = enhanced.astype(np.float32) / 255.0

        return {
            "success": True,
            "gray": gray,
            "enhanced": enhanced,
            "normalized": normalized,
            "message": "Preprocessing completed successfully.",
        }
