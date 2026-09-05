"""
Unit tests for core/cpp_bridge.py Python <-> C++ interface.
"""

import unittest
import numpy as np

import config
from core.cpp_bridge import NativeBridge, native_status, verify_liveness_native_or_fallback, cosine_similarity_native_or_fallback
from tests.test_liveness import make_synthetic_hand


class TestCppBridge(unittest.TestCase):

    def setUp(self):
        self.original_native_enabled = getattr(config, "NATIVE_CPP_ENABLED", False)

    def tearDown(self):
        config.NATIVE_CPP_ENABLED = self.original_native_enabled

    def test_bridge_disabled_reports_python_fallback(self):
        """When NATIVE_CPP_ENABLED=False, status indicates python-fallback and not available."""
        config.NATIVE_CPP_ENABLED = False
        stat = native_status()
        self.assertFalse(stat["available"])
        self.assertEqual(stat["version"], "python-fallback")

    def test_bridge_graceful_missing_binary(self):
        """When binary path points to nonexistent file, bridge is_available is False."""
        fake_bridge = NativeBridge(bin_path="nonexistent/path/palmpay_native")
        self.assertFalse(fake_bridge.is_available())
        stat = fake_bridge.get_status()
        self.assertFalse(stat["available"])

    def test_bridge_fallback_liveness(self):
        """Liveness verification works via fallback or native without exceptions."""
        frames = []
        for i in range(5):
            frames.append({"ts_ms": i * 100, "landmarks": make_synthetic_hand(False)})
        for i in range(5):
            frames.append({"ts_ms": (5 + i) * 100, "landmarks": make_synthetic_hand(True)})

        res = verify_liveness_native_or_fallback(frames, ["OPEN_PALM", "FIST"])
        self.assertTrue(res["passed"])
        self.assertIn("score", res)

    def test_bridge_cosine_similarity(self):
        """Cosine similarity helper matches NumPy behavior."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity_native_or_fallback(a, b), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity_native_or_fallback(a, c), 0.0, places=5)

    def test_bridge_with_native_binary_enabled(self):
        """If native binary is present in cpp/build, enabling the flag reports real version."""
        bridge = NativeBridge(bin_path="cpp/build/palmpay_native")
        if bridge.get_binary_path():
            config.NATIVE_CPP_ENABLED = True
            stat = bridge.get_status()
            if stat["available"]:
                self.assertEqual(stat["version"], "1.0.0-phase2")
                # Test native liveness
                frames = []
                for i in range(5):
                    frames.append({"ts_ms": i * 100, "landmarks": make_synthetic_hand(False)})
                for i in range(5):
                    frames.append({"ts_ms": (5 + i) * 100, "landmarks": make_synthetic_hand(True)})
                n_res = bridge.verify_liveness(frames, ["OPEN_PALM", "FIST"])
                self.assertTrue(n_res["passed"])
                self.assertTrue(n_res.get("native"))


if __name__ == "__main__":
    unittest.main()
