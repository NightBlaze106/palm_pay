"""
Hand Geometry Feature Extraction (Accuracy Sprint v4 — Geometry Fusion).

Extracts a compact, physically interpretable hand-shape descriptor from the 21
MediaPipe landmarks. Unlike the LBP texture channel (which measures skin
texture and cannot separate palms at webcam resolution), hand geometry is a
well-established biometric trait and is immune to illumination.

Invariance design:
    - All length features are DISTANCE RATIOS against a palm reference length
      (wrist -> middle MCP)  => invariant to scale and translation, and
      Euclidean distances are inherently rotation invariant.
    - Splay angles are SIGNED angles of each finger vector relative to the
      palm axis (wrist -> middle MCP) => invariant to rotation (measured
      against the palm's own axis), but they FLIP SIGN under mirroring, which
      is what allows left vs right hands of the same person to be told apart.

Feature layout (fixed order, GEO_FEATURE_DIM = 20 entries):
    [0]      palm aspect ratio      = |P5-P17| / ref
    [1:6]    finger chain lengths / ref   (thumb, index, middle, ring, pinky)
    [6:9]    knuckle widths / ref         (|P5-P9|, |P9-P13|, |P13-P17|)
    [9:15]   inter-tip distances / ref    (all 6 pairwise tip distances)
    [15:20]  signed splay angles (rad)    (thumb, index, middle, ring, pinky)
"""

import math
from typing import Dict, List, Tuple

import numpy as np

GEO_FEATURE_DIM = 20

# Landmark indices (MediaPipe Hands topology)
_WRIST = 0
_THUMB = (1, 2, 3, 4)
_INDEX = (5, 6, 7, 8)
_MIDDLE = (9, 10, 11, 12)
_RING = (13, 14, 15, 16)
_PINKY = (17, 18, 19, 20)
_FINGER_CHAINS = (_THUMB, _INDEX, _MIDDLE, _RING, _PINKY)
_KNUCKLES = (5, 9, 13, 17)
_TIPS = (8, 12, 16, 20)


def _chain_length(pts: np.ndarray, chain: Tuple[int, ...]) -> float:
    """Total polyline length through the given landmark indices."""
    total = 0.0
    for a, b in zip(chain[:-1], chain[1:]):
        total += float(np.linalg.norm(pts[a] - pts[b]))
    return total


def _signed_angle(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Signed angle (rad, [-pi, pi]) from vec_b to vec_a (counter-clockwise)."""
    cross = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
    dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
    return math.atan2(cross, dot)


def extract_geometry_features(landmarks_pixel: List[Tuple[float, float, float]]) -> Dict:
    """
    Computes the invariant hand-geometry feature vector from 21 landmarks.

    Args:
        landmarks_pixel: 21 (x, y, z) landmarks in pixel space (z ignored).

    Returns:
        {"success": bool, "geometry": np.ndarray(GEO_FEATURE_DIM,) float32 or None,
         "message": str}
    """
    if landmarks_pixel is None or len(landmarks_pixel) < 21:
        return {"success": False, "geometry": None,
                "message": "Insufficient landmarks for hand geometry."}

    pts = np.array([[lm[0], lm[1]] for lm in landmarks_pixel[:21]], dtype=np.float64)

    palm_axis = pts[_MIDDLE[0]] - pts[_WRIST]          # wrist -> middle MCP
    ref = float(np.linalg.norm(palm_axis))
    if ref < 1e-6:
        return {"success": False, "geometry": None,
                "message": "Degenerate hand geometry (zero palm axis)."}

    features: List[float] = []

    # [0] palm aspect ratio: knuckle breadth relative to palm height
    breadth = float(np.linalg.norm(pts[_KNUCKLES[0]] - pts[_KNUCKLES[3]]))
    features.append(breadth / ref)

    # [1:6] finger chain lengths (normalized)
    for chain in _FINGER_CHAINS:
        features.append(_chain_length(pts, chain) / ref)

    # [6:9] knuckle segment widths (normalized)
    for a, b in zip(_KNUCKLES[:-1], _KNUCKLES[1:]):
        features.append(float(np.linalg.norm(pts[a] - pts[b])) / ref)

    # [9:13] inter-tip distances (normalized)
    for i in range(len(_TIPS) - 1):
        for j in range(i + 1, len(_TIPS)):
            features.append(
                float(np.linalg.norm(pts[_TIPS[i]] - pts[_TIPS[j]])) / ref
            )

    # [13:18] signed splay angles: finger root direction vs palm axis.
    # Finger vector runs MCP -> TIP (root direction of the whole finger).
    for chain in _FINGER_CHAINS:
        finger_vec = pts[chain[-1]] - pts[chain[0]]
        features.append(_signed_angle(finger_vec, palm_axis))

    geometry = np.asarray(features, dtype=np.float32)
    if geometry.shape[0] != GEO_FEATURE_DIM or not np.all(np.isfinite(geometry)):
        return {"success": False, "geometry": None,
                "message": "Hand geometry computation produced invalid values."}

    return {"success": True, "geometry": geometry,
            "message": "Hand geometry features extracted."}


def geometry_distance(query: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> float:
    """
    RMS normalized deviation between a query geometry vector and a user's
    reference statistics. Lower = more similar; ~1.0 means "within one
    reference std per dimension on average".

    Args:
        query: geometry vector (GEO_FEATURE_DIM,).
        ref_mean: per-dimension mean of the user's enrollment geometries.
        ref_std: per-dimension std (floored internally to avoid divide-by-zero).

    Returns:
        float distance (>= 0).
    """
    if query is None or ref_mean is None or ref_std is None:
        return float("inf")
    query = np.asarray(query, dtype=np.float64)
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)
    if not (query.shape == ref_mean.shape == ref_std.shape) or query.size == 0:
        return float("inf")
    # Relative floor: dimensions with tiny spread do not dominate the distance.
    sigma = np.maximum(ref_std, 0.10 * np.abs(ref_mean) + 1e-3)
    return float(np.sqrt(np.mean(((query - ref_mean) / sigma) ** 2)))


# Absolute per-feature natural scale used by the enrollment consistency gate:
# length/ratio features (dims 0-14) jitter ~0.01-0.03 between same-pose frames;
# splay angles (dims 15-19, radians) jitter ~0.05-0.10. These floors are in
# PHYSICAL units, so the gate detects absolutely mis-posed samples even when
# every submitted sample is equally wild (a relative/MAD scale cannot).
_ABS_SIGMA = np.array([0.05] * 15 + [0.12] * 5, dtype=np.float64)


def enrollment_consistency_distances(geometries: List[np.ndarray]) -> List[float]:
    """
    Per-sample RMS deviation from the enrollment's per-dimension MEDIAN,
    measured in absolute feature units (see _ABS_SIGMA).

    Unlike geometry_distance (which normalizes by the cohort's own spread and
    therefore cannot detect globally-inconsistent submissions), this metric has
    a fixed physical scale: ~0.3-0.8 for a consistently-posed enrollment,
    several units for a mis-posed (bent / pitched / mid-motion) capture.

    Args:
        geometries: list of geometry vectors (GEO_FEATURE_DIM,).

    Returns:
        List of distances, one per input sample (same order).
    """
    if not geometries:
        return []
    arr = np.stack([np.asarray(g, dtype=np.float64) for g in geometries])
    median = np.median(arr, axis=0)
    return [
        float(np.sqrt(np.mean(((row - median) / _ABS_SIGMA) ** 2)))
        for row in arr
    ]
