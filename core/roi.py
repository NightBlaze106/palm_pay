"""
Palm ROI Extraction and Landmark-Guided Affine Normalization Module.
Isolates the central palm region from full-hand landmarks, compensating
for hand scale, translation, and 2D rotation.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import config


class PalmROIExtractor:
    """
    Extracts a canonical, rotation-normalized square Region of Interest (ROI)
    of the central palm surface from 21 MediaPipe hand landmarks.
    """

    def __init__(
        self,
        roi_size: Tuple[int, int] = config.ROI_SIZE,
        scale_factor: float = config.ROI_SCALE_FACTOR,
    ):
        self.roi_width, self.roi_height = roi_size
        self.scale_factor = scale_factor

    def extract_roi(
        self,
        image_bgr: np.ndarray,
        landmarks_pixel: List[Tuple[int, int, float]],
    ) -> Dict:
        """
        Calculates palm geometry, computes rotation and scale, and applies
        affine transformation to isolate the normalized palm ROI.

        Args:
            image_bgr: Source camera frame in BGR format.
            landmarks_pixel: 21 landmark coordinates [(x, y, z), ...] in pixel space.

        Returns:
            Dict containing:
                - success: bool
                - roi_bgr: Cropped and normalized palm image (ROI_WIDTH x ROI_HEIGHT x 3) or None
                - center: (cx, cy) palm centroid in original frame
                - angle_deg: Rotation angle in degrees
                - scale_px: Estimated palm physical span in pixels
                - roi_polygon: 4 corner coordinates in original frame for UI overlay
                - message: Status message
        """
        if image_bgr is None or len(landmarks_pixel) < 21:
            return {
                "success": False,
                "roi_bgr": None,
                "center": (0, 0),
                "angle_deg": 0.0,
                "scale_px": 0.0,
                "palm_breadth_px": 0.0,
                "roi_polygon": [],
                "knuckle_y_roi": None,
                "message": "Insufficient hand landmarks for palm ROI extraction.",
            }

        # Extract anchor landmarks
        p_wrist = np.array(landmarks_pixel[0][:2], dtype=np.float32)       # Wrist (0)
        p_index_mcp = np.array(landmarks_pixel[5][:2], dtype=np.float32)   # Index MCP (5)
        p_middle_mcp = np.array(landmarks_pixel[9][:2], dtype=np.float32)  # Middle MCP (9)
        p_ring_mcp = np.array(landmarks_pixel[13][:2], dtype=np.float32)  # Ring MCP (13)
        p_pinky_mcp = np.array(landmarks_pixel[17][:2], dtype=np.float32) # Pinky MCP (17)

        # 1. MCP joint base midpoint
        p_mcp_mid = (p_index_mcp + p_pinky_mcp) / 2.0

        # 2. Central Palm Centroid: weighted midpoint between wrist and MCP base
        # Slightly shifted towards the center of the palm (away from finger joints and wrist line)
        p_palm_center = p_wrist * 0.45 + p_mcp_mid * 0.55
        cx, cy = float(p_palm_center[0]), float(p_palm_center[1])

        # 3. Orientation Vector: from Wrist (0) to Middle MCP (9)
        vec_v = p_middle_mcp - p_wrist
        dx, dy = float(vec_v[0]), float(vec_v[1])
        len_v = np.sqrt(dx * dx + dy * dy)

        if len_v < 15.0:  # Hand is too small or distant
            return {
                "success": False,
                "roi_bgr": None,
                "center": (int(cx), int(cy)),
                "angle_deg": 0.0,
                "scale_px": float(len_v),
                "palm_breadth_px": 0.0,
                "roi_polygon": [],
                "knuckle_y_roi": None,
                "message": "Hand is too far from camera. Please move your palm closer.",
            }

        # Calculate rotation angle in degrees so vertical axis points upright
        # np.arctan2(dy, dx) returns angle from positive X axis
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad) - 90.0  # Upright orientation has dy < 0, dx ~ 0 -> -90 - (-90) = 0

        # 4. Palm Scale Estimation:
        # Distance between Index MCP and Pinky MCP (palm breadth)
        palm_breadth = np.linalg.norm(p_index_mcp - p_pinky_mcp)
        palm_height = len_v

        # Robust reference scale: geometric mean of breadth and height
        ref_scale = max(palm_breadth, palm_height * 0.75) * self.scale_factor
        half_box = ref_scale / 2.0

        # 5. Affine Transformation Matrix:
        # Maps the rotated, scaled palm region in image_bgr to a fixed (roi_width, roi_height) canvas
        # Target center in output image is (roi_width / 2, roi_height / 2)
        scale_map = float(self.roi_width) / (2.0 * half_box)
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale_map)
        # Shift center to target ROI canvas center
        rot_mat[0, 2] += (self.roi_width / 2.0) - cx
        rot_mat[1, 2] += (self.roi_height / 2.0) - cy

        # Apply Affine Warp
        roi_bgr = cv2.warpAffine(
            image_bgr,
            rot_mat,
            (self.roi_width, self.roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 5b. Knuckle line in ROI coordinates (finger masking support):
        # project the four MCP joints through the same affine matrix. In the
        # normalized canvas the wrist sits near the bottom and knuckles above
        # the palm center, so rows with smaller y than this line are fingers.
        mcp_pts = np.array(
            [p_index_mcp, p_middle_mcp, p_ring_mcp, p_pinky_mcp],
            dtype=np.float32,
        )  # shape (4, 2)
        ones_col = np.ones((4, 1), dtype=np.float32)
        roi_mcp = (rot_mat @ np.hstack([mcp_pts, ones_col]).T).T  # (4, 2)
        knuckle_y_roi = float(np.mean(roi_mcp[:, 1])) + float(config.FINGER_MASK_OFFSET_PX)

        # 6. Compute 4 corner points of the ROI in original frame coordinates for visualization
        # Canonical corners relative to target canvas:
        canonical_corners = np.array(
            [
                [0, 0, 1],
                [self.roi_width, 0, 1],
                [self.roi_width, self.roi_height, 1],
                [0, self.roi_height, 1],
            ],
            dtype=np.float32,
        ).T

        # Invert affine transformation matrix to project back to original image
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)
        orig_corners = (inv_rot_mat @ canonical_corners).T
        polygon_points = [(int(pt[0]), int(pt[1])) for pt in orig_corners]

        return {
            "success": True,
            "roi_bgr": roi_bgr,
            "center": (int(cx), int(cy)),
            "angle_deg": float(angle_deg),
            "scale_px": float(ref_scale),
            "palm_breadth_px": float(palm_breadth),
            "roi_polygon": polygon_points,
            "knuckle_y_roi": knuckle_y_roi,
            "message": "Palm ROI successfully extracted and normalized.",
        }

    def draw_roi_box(
        self,
        image_bgr: np.ndarray,
        roi_polygon: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws the ROI bounding polygon onto a copy of the camera frame.
        """
        annotated = image_bgr.copy()
        if len(roi_polygon) == 4:
            pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        return annotated
