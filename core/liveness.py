"""
Challenge-Response Liveness Detection for Contactless Palm Biometrics.
Verifies server-side landmark articulation and state transitions to prevent
photo/replay and static spoof attacks.
"""

import math
import secrets
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

import config


class LivenessSessionStore:
    """
    Thread-safe in-memory store for liveness challenges and single-use verified tokens.
    """

    def __init__(self, ttl_seconds: int = config.LIVENESS_TOKEN_TTL_S):
        self.ttl_seconds = int(ttl_seconds)
        self._challenges: Dict[str, Dict] = {}
        self._tokens: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now() -> float:
        return time.time()

    def create_challenge(self, challenge_pool: Optional[List[str]] = None, length: int = 2) -> Dict:
        """Issues a fresh challenge with a random sequence of palm gestures."""
        pool = challenge_pool or config.LIVENESS_CHALLENGE_POOL
        # Pick sequence ensuring consecutive elements are not identical
        sequence = [secrets.choice(pool)]
        while len(sequence) < length:
            nxt = secrets.choice(pool)
            if nxt != sequence[-1]:
                sequence.append(nxt)

        challenge_id = secrets.token_urlsafe(24)
        expires_at = self._now() + (config.LIVENESS_WINDOW_MS / 1000.0)

        instructions = {
            "OPEN_PALM": "Show open, relaxed palm",
            "FIST": "Close your hand into a fist",
            "SPREAD": "Spread all fingers wide",
        }

        with self._lock:
            self._purge_expired_locked()
            self._challenges[challenge_id] = {
                "sequence": sequence,
                "expires_at": expires_at,
                "created_at": self._now(),
            }

        return {
            "challenge_id": challenge_id,
            "sequence": sequence,
            "expires_at": expires_at,
            "instructions": [instructions.get(c, c) for c in sequence],
            "window_ms": config.LIVENESS_WINDOW_MS,
        }

    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        """Retrieves an unexpired challenge."""
        if not challenge_id:
            return None
        with self._lock:
            ch = self._challenges.get(challenge_id)
            if ch is None:
                return None
            if self._now() > ch["expires_at"]:
                del self._challenges[challenge_id]
                return None
            return dict(ch)

    def consume_challenge(self, challenge_id: str) -> Optional[Dict]:
        """Atomically burns a challenge so it cannot be re-verified."""
        if not challenge_id:
            return None
        with self._lock:
            ch = self._challenges.pop(challenge_id, None)
            if ch is None or self._now() > ch["expires_at"]:
                return None
            return ch

    def issue_token(self, score: float) -> str:
        """Issues a single-use liveness token."""
        token = "live_" + secrets.token_urlsafe(32)
        with self._lock:
            self._purge_expired_locked()
            self._tokens[token] = {
                "score": float(score),
                "created_at": self._now(),
                "expires_at": self._now() + self.ttl_seconds,
            }
        return token

    def verify_and_consume_token(self, token: Optional[str]) -> Tuple[bool, Optional[float], str]:
        """
        Validates and burns a single-use liveness token.
        Returns: (valid, score, reason)
        """
        if not token:
            return False, None, "liveness_missing"
        with self._lock:
            entry = self._tokens.pop(token, None)
            if entry is None:
                return False, None, "liveness_consumed_or_invalid"
            if self._now() > entry["expires_at"]:
                return False, None, "liveness_expired"
            return True, entry["score"], "ok"

    def _purge_expired_locked(self):
        now = self._now()
        exp_ch = [cid for cid, ch in self._challenges.items() if now > ch["expires_at"]]
        for cid in exp_ch:
            del self._challenges[cid]
        exp_tok = [tok for tok, t in self._tokens.items() if now > t["expires_at"]]
        for tok in exp_tok:
            del self._tokens[tok]


class LandmarkLivenessDetector:
    """
    Landmark-based gesture classifier and challenge-response sequence verifier.
    Relies purely on 21 landmarks in pixel/normalized space. Never accesses raw images.
    """

    # Landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_MCPS = [2, 5, 9, 13, 17]

    @classmethod
    def classify_frame(cls, landmarks: List[Tuple[float, float, float]]) -> Dict:
        """
        Classifies hand state into: 'OPEN_PALM', 'FIST', 'SPREAD', or 'UNKNOWN'.
        Uses invariant distance ratios relative to palm height (wrist -> middle MCP).
        """
        if not landmarks or len(landmarks) < 21:
            return {"state": "UNKNOWN", "extension_ratio": 0.0, "spread_ratio": 0.0}

        pts = np.array([[lm[0], lm[1]] for lm in landmarks[:21]], dtype=np.float64)

        # Reference length: wrist (0) to middle MCP (9)
        ref_vec = pts[cls.MIDDLE_MCP] - pts[cls.WRIST]
        ref = float(np.linalg.norm(ref_vec))
        if ref < 1e-6:
            return {"state": "UNKNOWN", "extension_ratio": 0.0, "spread_ratio": 0.0}

        # 1. Tip-to-wrist distances normalized by ref
        tip_dists = [float(np.linalg.norm(pts[tip] - pts[cls.WRIST])) / ref for tip in cls.FINGER_TIPS]
        # MCP-to-wrist distances
        mcp_dists = [float(np.linalg.norm(pts[mcp] - pts[cls.WRIST])) / ref for mcp in cls.FINGER_MCPS]

        # 2. Extension ratio: are tips significantly further than MCPs?
        # In a closed fist, fingertip curls towards palm, tip dist <= MCP dist
        extended_count = sum(1 for t, m in zip(tip_dists[1:], mcp_dists[1:]) if t > (m + 0.35))
        avg_tip_dist = float(np.mean(tip_dists[1:]))

        # 3. Inter-tip spread ratio: average distance between adjacent fingertips normalized by ref
        adjacent_spreads = [
            float(np.linalg.norm(pts[cls.FINGER_TIPS[i]] - pts[cls.FINGER_TIPS[i + 1]])) / ref
            for i in range(len(cls.FINGER_TIPS) - 1)
        ]
        avg_spread = float(np.mean(adjacent_spreads))

        # Classification rules:
        # FIST: almost no extended fingers, tips retracted close to wrist/palm
        if extended_count == 0 or avg_tip_dist < 1.15:
            state = "FIST"
        elif extended_count >= 3:
            # Check for SPREAD vs OPEN_PALM
            # SPREAD has wide inter-tip distance
            if avg_spread >= 0.52:
                state = "SPREAD"
            else:
                state = "OPEN_PALM"
        else:
            state = "OPEN_PALM" if avg_tip_dist >= 1.3 else "FIST"

        return {
            "state": state,
            "extension_ratio": round(avg_tip_dist, 3),
            "spread_ratio": round(avg_spread, 3),
            "extended_count": extended_count,
        }

    @classmethod
    def verify_sequence(
        cls,
        frames: List[Dict],
        expected_sequence: List[str],
    ) -> Dict:
        """
        Verifies a temporal sequence of landmark frames against a challenge sequence.

        Checks:
          1. Minimum frame count
          2. Static image rejection (motion / velocity / variance check)
          3. Rigid motion rejection (fingers must articulate, not just translate)
          4. Correct sequential transitions matching expected_sequence
        """
        reasons = []

        if len(frames) < config.LIVENESS_MIN_FRAMES:
            return {
                "passed": False,
                "score": 0.0,
                "reasons": [f"insufficient_frames({len(frames)}<{config.LIVENESS_MIN_FRAMES})"],
                "detected_sequence": [],
            }

        if len(frames) > config.LIVENESS_MAX_FRAMES:
            frames = frames[: config.LIVENESS_MAX_FRAMES]

        # Extract landmarks per frame
        classified_states = []
        extension_series = []
        spread_series = []
        wrist_positions = []

        for f in frames:
            lms = f.get("landmarks") or []
            if len(lms) >= 21:
                wrist_positions.append([lms[0][0], lms[0][1]])
                c = cls.classify_frame(lms)
                classified_states.append(c["state"])
                extension_series.append(c["extension_ratio"])
                spread_series.append(c["spread_ratio"])

        if len(classified_states) < config.LIVENESS_MIN_FRAMES:
            return {
                "passed": False,
                "score": 0.0,
                "reasons": ["insufficient_valid_landmarks"],
                "detected_sequence": [],
            }

        # Check 1: Anti-Static photo check (is there articulation motion?)
        # Variance of extension and spread across frames must be non-trivial
        var_ext = float(np.var(extension_series))
        var_spread = float(np.var(spread_series))
        max_ext_diff = float(np.max(extension_series) - np.min(extension_series))

        if var_ext < 0.005 and max_ext_diff < 0.15:
            reasons.append("static_image_detected")

        # Check 2: Rigid-only motion check
        # If wrist moves a lot but extension variance is zero -> moving static photo
        if len(wrist_positions) > 1:
            wrist_movement = float(np.std(wrist_positions, axis=0).mean())
            if wrist_movement > 5.0 and var_ext < 0.002:
                reasons.append("rigid_motion_without_articulation")

        # Check 3: Extract temporal sequence of states (collapsing consecutive duplicates)
        collapsed_seq = []
        current_state = None
        consecutive_count = 0

        for st in classified_states:
            if st != current_state:
                if current_state is not None and consecutive_count >= 2:
                    collapsed_seq.append(current_state)
                current_state = st
                consecutive_count = 1
            else:
                consecutive_count += 1
        if current_state is not None and consecutive_count >= 2:
            collapsed_seq.append(current_state)

        # Match against expected sequence in order
        target_idx = 0
        matched_targets = []
        for st in collapsed_seq:
            if target_idx < len(expected_sequence) and st == expected_sequence[target_idx]:
                matched_targets.append(st)
                target_idx += 1

        sequence_passed = (target_idx == len(expected_sequence))
        if not sequence_passed:
            reasons.append(f"challenge_order_mismatch(expected={expected_sequence}, found={collapsed_seq})")

        # Compute liveness quality score
        score = 0.0
        if not reasons:
            # Score based on clarity of transitions and finger dynamic range
            articulation_score = min(1.0, max_ext_diff / 0.6)
            temporal_score = 1.0 if sequence_passed else 0.0
            score = round(0.4 * articulation_score + 0.6 * temporal_score, 4)
            min_score = getattr(config, "LIVENESS_MIN_SCORE", 0.0)
            if min_score > 0.0 and score < min_score:
                passed = False
                reasons.append(f"score_below_minimum({score:.4f}<{min_score:.4f})")
            else:
                passed = True
        else:
            score = round(max(0.0, 0.2 * (target_idx / max(1, len(expected_sequence)))), 4)
            passed = False

        return {
            "passed": passed,
            "score": score,
            "reasons": reasons if reasons else ["ok"],
            "detected_sequence": collapsed_seq,
        }
