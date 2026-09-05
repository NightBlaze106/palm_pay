"""
Unit tests for hand-geometry feature extraction (Accuracy Sprint v4).

Verifies the invariance contract that the fusion design depends on:
  - rotation / scale / translation of the hand leaves features unchanged,
  - MIRRORING keeps length ratios but flips the signed splay angles,
    i.e. left vs right hands of the same person are distinguishable,
  - distance metric behaves sanely (self ~ 0, scaled stats, inf on bad input).
"""

import unittest

import numpy as np

from core.geometry import GEO_FEATURE_DIM, extract_geometry_features, geometry_distance


def _synthetic_hand():
    """A crude but anatomically plausible right hand (palm axis = -Y)."""
    pts = []
    pts.append((200.0, 400.0))   # 0 wrist
    pts.append((120.0, 380.0))   # 1 thumb CMC
    pts.append((90.0, 350.0))    # 2 thumb MCP
    pts.append((70.0, 320.0))    # 3 thumb IP
    pts.append((55.0, 290.0))    # 4 thumb tip
    pts.append((150.0, 300.0))   # 5 index MCP
    pts.append((145.0, 250.0))   # 6 index PIP
    pts.append((142.0, 210.0))   # 7 index DIP
    pts.append((140.0, 175.0))   # 8 index tip
    pts.append((200.0, 285.0))   # 9 middle MCP
    pts.append((200.0, 225.0))   # 10 middle PIP
    pts.append((200.0, 175.0))   # 11 middle DIP
    pts.append((200.0, 135.0))   # 12 middle tip
    pts.append((250.0, 300.0))   # 13 ring MCP
    pts.append((255.0, 245.0))   # 14 ring PIP
    pts.append((258.0, 200.0))   # 15 ring DIP
    pts.append((260.0, 165.0))   # 16 ring tip
    pts.append((295.0, 320.0))   # 17 pinky MCP
    pts.append((305.0, 280.0))   # 18 pinky PIP
    pts.append((312.0, 245.0))   # 19 pinky DIP
    pts.append((318.0, 215.0))   # 20 pinky tip
    return [(x, y, 0.0) for x, y in pts]


def _transform(points, angle_rad, scale, translation):
    """Apply 2D rotation + uniform scale + translation (z passthrough)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    out = []
    for x, y, z in points:
        p = rot @ (np.array([x, y]) * scale) + np.array(translation)
        out.append((float(p[0]), float(p[1]), z))
    return out


def _mirror(points):
    """Mirror across the vertical axis (left hand <-> right hand shape)."""
    return [(-x, y, z) for x, y, z in points]


class GeometryExtractionTests(unittest.TestCase):
    def setUp(self):
        self.hand = _synthetic_hand()
        res = extract_geometry_features(self.hand)
        self.assertTrue(res["success"])
        self.base = res["geometry"]

    def test_feature_dim_and_dtype(self):
        self.assertEqual(self.base.shape, (GEO_FEATURE_DIM,))
        self.assertEqual(GEO_FEATURE_DIM, 20)
        self.assertEqual(self.base.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(self.base)))

    def test_rigid_transform_invariance(self):
        """Rotation + scale + translation must leave features unchanged."""
        for angle, scale, trans in [(0.7, 1.0, (0, 0)), (2.1, 0.35, (-500, 900)),
                                    (-1.2, 3.0, (50, -80))]:
            res = extract_geometry_features(_transform(self.hand, angle, scale, trans))
            self.assertTrue(res["success"])
            np.testing.assert_allclose(res["geometry"], self.base, atol=1e-4)

    def test_mirror_flips_splay_angles_only(self):
        """Mirroring (left<->right hand): lengths identical, splay angles negated."""
        res = extract_geometry_features(_mirror(self.hand))
        self.assertTrue(res["success"])
        mirrored = res["geometry"]
        # Length-type features (indices 0..14) unchanged
        np.testing.assert_allclose(mirrored[:15], self.base[:15], atol=1e-4)
        # Signed splay angles (15..19) flipped
        np.testing.assert_allclose(mirrored[15:], -self.base[15:], atol=1e-4)
        # And therefore the full vectors differ (left hand != right hand)
        self.assertGreater(float(np.linalg.norm(mirrored - self.base)), 0.1)

    def test_insufficient_landmarks_fails(self):
        res = extract_geometry_features(self.hand[:15])
        self.assertFalse(res["success"])
        self.assertIsNone(res["geometry"])

    def test_degenerate_hand_fails(self):
        collapsed = [(100.0, 100.0, 0.0)] * 21
        res = extract_geometry_features(collapsed)
        self.assertFalse(res["success"])


class GeometryDistanceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.samples = np.stack([
            extract_geometry_features(_transform(_synthetic_hand(), 0.1 * i, 1.0, (i, i)))["geometry"]
            for i in range(6)
        ]).astype(np.float64)
        self.mu = self.samples.mean(axis=0)
        self.sigma = self.samples.std(axis=0)

    def test_self_distance_near_zero(self):
        d = geometry_distance(self.samples[0], self.mu, self.sigma)
        self.assertLess(d, 1.5)

    def test_shifted_hand_farther_than_self(self):
        """A structurally different hand (stretched fingers) must sit farther out."""
        stretched = _transform(_synthetic_hand(), 0.0, 1.0, (0, 0))
        pts = [(x, 1.25 * (y - 400) + 400, z) for x, y, z in stretched]  # stretch fingers up
        q = extract_geometry_features(pts)["geometry"].astype(np.float64)
        self.assertGreater(geometry_distance(q, self.mu, self.sigma),
                           geometry_distance(self.samples[0], self.mu, self.sigma))

    def test_std_floor_prevents_division_explosion(self):
        tiny_sigma = np.full(GEO_FEATURE_DIM, 1e-9)
        d = geometry_distance(self.samples[0], self.mu, tiny_sigma)
        self.assertTrue(np.isfinite(d))

    def test_bad_inputs_return_inf(self):
        self.assertEqual(geometry_distance(None, self.mu, self.sigma), float("inf"))
        self.assertEqual(geometry_distance(self.samples[0], self.mu[:5], self.sigma[:5]), float("inf"))


if __name__ == "__main__":
    unittest.main()
