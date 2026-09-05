"""
Python <-> C++ Native Core Bridge.
Safely calls palmpay_native CLI binary with timeout, strict I/O validation,
and seamless fallback to Python when the binary is absent or disabled.
"""

import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

import config


class NativeBridge:
    """
    Subprocess-based bridge to cpp/build/palmpay_native.
    """

    def __init__(self, bin_path: Optional[str] = None):
        self.bin_path = bin_path or getattr(config, "NATIVE_CPP_BIN", "cpp/build/palmpay_native")

    def get_binary_path(self) -> str:
        path = self.bin_path
        if not os.path.isabs(path):
            path = os.path.join(config.BASE_DIR, path)
        return path

    def is_available(self) -> bool:
        """Returns True only if NATIVE_CPP_ENABLED is True and the binary exists + is executable."""
        if not getattr(config, "NATIVE_CPP_ENABLED", False):
            return False
        p = self.get_binary_path()
        return os.path.isfile(p) and os.access(p, os.X_OK)

    def get_status(self) -> Dict:
        """
        Queries native binary status subcommand.
        Returns status dictionary with 'available': bool.
        """
        if not self.is_available():
            return {
                "available": False,
                "version": "python-fallback",
                "onnx_available": False,
                "opencv_version": "python-cv2",
                "build_mode": "none",
                "binary_path": self.get_binary_path(),
            }

        try:
            res = subprocess.run(
                [self.get_binary_path(), "status"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0:
                data = json.loads(res.stdout.strip())
                data["available"] = True
                data["binary_path"] = self.get_binary_path()
                return data
        except Exception as exc:
            pass

        return {
            "available": False,
            "version": "error",
            "onnx_available": False,
            "build_mode": "unavailable",
            "binary_path": self.get_binary_path(),
        }

    def verify_liveness(
        self,
        frames: List[Dict],
        challenge_sequence: List[str],
    ) -> Dict:
        """
        Dispatches liveness sequence verification to native CLI if available,
        otherwise falls back directly to core.liveness.LandmarkLivenessDetector.
        """
        if self.is_available():
            payload = {
                "challenge": challenge_sequence,
                "frames": frames,
            }
            try:
                p = subprocess.Popen(
                    [self.get_binary_path(), "liveness"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = p.communicate(input=json.dumps(payload), timeout=3.0)
                if p.returncode == 0:
                    resp = json.loads(stdout.strip())
                    if resp.get("ok"):
                        return {
                            "passed": bool(resp.get("passed", False)),
                            "score": float(resp.get("score", 0.0)),
                            "reasons": [resp.get("reason", "ok")],
                            "native": True,
                        }
            except Exception:
                # Soft failure: drop to python fallback
                pass

        # Python fallback
        from core.liveness import LandmarkLivenessDetector
        res = LandmarkLivenessDetector.verify_sequence(frames, challenge_sequence)
        res["native"] = False
        return res

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity with native acceleration if available, else Python NumPy.
        """
        if vec_a is None or vec_b is None or len(vec_a) == 0 or len(vec_b) == 0:
            return 0.0
        if len(vec_a) != len(vec_b):
            return 0.0

        if self.is_available():
            payload = {
                "op": "cosine",
                "vec_a": [float(x) for x in vec_a],
                "vec_b": [float(x) for x in vec_b],
            }
            try:
                p = subprocess.Popen(
                    [self.get_binary_path(), "match"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, _ = p.communicate(input=json.dumps(payload), timeout=2.0)
                if p.returncode == 0:
                    data = json.loads(stdout.strip())
                    if "score" in data:
                        return float(data["score"])
            except Exception:
                pass

        # Python NumPy fallback
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0
        return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0))


# Global singleton instance
bridge = NativeBridge()


def native_status() -> Dict:
    return bridge.get_status()


def verify_liveness_native_or_fallback(frames: List[Dict], challenge_sequence: List[str]) -> Dict:
    return bridge.verify_liveness(frames, challenge_sequence)


def cosine_similarity_native_or_fallback(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return bridge.cosine_similarity(vec_a, vec_b)
