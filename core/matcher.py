"""
Biometric Template Matcher and Similarity Calculator.
Performs 1:1 verification and 1:N identification using Cosine Similarity
against enrolled palm templates.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

import config


class PalmMatcher:
    """
    Computes biometric similarity scores and performs 1:N identification decisions.
    """

    def __init__(self, threshold: float = config.DEFAULT_RECOGNITION_THRESHOLD):
        self.threshold = float(threshold)

    def set_threshold(self, threshold: float):
        """Updates the decision threshold dynamically."""
        self.threshold = float(threshold)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates cosine similarity between two feature vectors.
        For unit-normalized vectors, this equals the dot product.
        """
        if vec_a is None or vec_b is None or len(vec_a) == 0 or len(vec_b) == 0:
            return 0.0
        if len(vec_a) != len(vec_b):
            # Dimension mismatch (e.g. template from an older model version).
            return 0.0

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0

        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clamp to [-1.0, 1.0] to guard against floating point imprecision
        return float(np.clip(sim, -1.0, 1.0))

    @staticmethod
    def aggregate_templates(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Combines multiple enrollment sample vectors into a unified mean template vector.
        """
        if not embeddings:
            return np.zeros(config.FEATURE_VECTOR_DIM, dtype=np.float32)

        stacked = np.stack(embeddings, axis=0)
        mean_vec = np.mean(stacked, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-7:
            mean_vec /= norm
        return mean_vec

    def frame_pass_score(self, query_embedding: np.ndarray, enrolled_users: List[Dict]) -> float:
        """
        Best cosine similarity of ONE query frame against ANY enrolled template.
        Used by multi-frame fusion to count how many frames independently clear
        the threshold.
        """
        best = -1.0
        for user in enrolled_users:
            for tmpl in user.get("templates", []):
                s = self.cosine_similarity(query_embedding, tmpl)
                if s > best:
                    best = s
        return max(best, 0.0)

    def match_1_to_n(
        self,
        query_embedding: np.ndarray,
        enrolled_users: List[Dict],
    ) -> Dict:
        """
        Matches a query palm embedding against all enrolled user templates (1:N search).

        Args:
            query_embedding: 1D normalized feature vector (config.FEATURE_VECTOR_DIM,).
            enrolled_users: List of user dicts, each containing:
                - 'id': user_id
                - 'name': user name
                - 'usn': USN / student ID
                - 'templates': list of 1D numpy arrays (individual samples)

        Returns:
            Dict containing:
                - recognized: bool (score >= threshold)
                - matched_user: Dict of recognized user or None
                - similarity_score: float (highest similarity found)
                - threshold_used: float
                - all_scores: List of (user_id, name, score)
                - message: Decision explanation
        """
        if query_embedding is None or len(query_embedding) == 0:
            return {
                "recognized": False,
                "matched_user": None,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "all_scores": [],
                "reason": "invalid_query",
                "message": "Invalid query feature vector.",
            }

        if not enrolled_users:
            return {
                "recognized": False,
                "matched_user": None,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "all_scores": [],
                "reason": "no_enrolled_users",
                "message": "No enrolled users in database. Please enroll a user first.",
            }

        # ---- Hardened scoring (Accuracy Sprint) ----
        # For each user compute TWO signals:
        #   mean_score  — cosine vs the user's normalized MEAN template (robust, primary)
        #   best_sample — max cosine over individual samples (sanity check)
        # A candidate only qualifies when BOTH clear the threshold. Ranking uses
        # the mean score. This removes the old max-of-noisy-scores inflation that
        # let unregistered palms slip through.
        candidates = []
        for user in enrolled_users:
            user_templates = user.get("templates", [])
            if not user_templates:
                continue

            mean_template = self.aggregate_templates(user_templates)
            mean_score = self.cosine_similarity(query_embedding, mean_template)
            best_sample = max(
                (self.cosine_similarity(query_embedding, t) for t in user_templates),
                default=0.0,
            )

            candidates.append(
                {
                    "user_id": user["id"],
                    "name": user["name"],
                    "usn": user.get("usn", ""),
                    "score": round(mean_score, 4),
                    "best_sample_score": round(best_sample, 4),
                    "_passes": (mean_score >= self.threshold) and (best_sample >= self.threshold),
                }
            )

        if not candidates:
            return {
                "recognized": False,
                "matched_user": None,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "all_scores": [],
                "reason": "no_valid_templates",
                "message": "No usable templates found for enrolled users.",
            }

        # Rank by mean score (tie-break on best sample), descending.
        candidates.sort(key=lambda c: (c["score"], c["best_sample_score"]), reverse=True)
        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        all_scores = [
            {"user_id": c["user_id"], "name": c["name"], "usn": c["usn"], "score": c["score"], "best_sample_score": c["best_sample_score"]}
            for c in candidates
        ]

        def _public(c):
            return {"id": c["user_id"], "name": c["name"], "usn": c["usn"]}

        if not top["_passes"]:
            return {
                "recognized": False,
                "matched_user": None,
                "best_candidate": _public(top),
                "similarity_score": round(max(top["score"], 0.0), 4),
                "threshold_used": self.threshold,
                "all_scores": all_scores,
                "reason": "below_threshold",
                "message": (
                    f"Palm Not Recognized (Best Match: {top['name']} with Score: "
                    f"{top['score']:.4f} < Threshold: {self.threshold:.2f})"
                ),
            }

        # Ambiguity guard: two qualifying identities within a tiny margin of each
        # other must NOT resolve to a payment.
        ambiguous = False
        if second is not None and second["_passes"]:
            gap = top["score"] - second["score"]
            if gap < config.MATCHER_AMBIGUITY_MARGIN:
                ambiguous = True

        if ambiguous:
            return {
                "recognized": False,
                "matched_user": None,
                "best_candidate": _public(top),
                "similarity_score": round(max(top["score"], 0.0), 4),
                "threshold_used": self.threshold,
                "all_scores": all_scores,
                "reason": "ambiguous",
                "message": (
                    f"Ambiguous palm — {top['name']} ({top['score']:.4f}) and "
                    f"{second['name']} ({second['score']:.4f}) are too close to decide."
                ),
            }

        return {
            "recognized": True,
            "matched_user": _public(top),
            "best_candidate": _public(top),
            "similarity_score": round(max(top["score"], 0.0), 4),
            "threshold_used": self.threshold,
            "all_scores": all_scores,
            "reason": "ok",
            "message": (
                f"Palm Recognized: {top['name']} (Mean-Template Score: {top['score']:.4f} "
                f">= Threshold: {self.threshold:.2f})"
            ),
        }
