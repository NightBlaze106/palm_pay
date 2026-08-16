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

    def match_1_to_n(
        self,
        query_embedding: np.ndarray,
        enrolled_users: List[Dict],
    ) -> Dict:
        """
        Matches a query palm embedding against all enrolled user templates (1:N search).

        Args:
            query_embedding: 1D normalized feature vector (944,).
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
                "message": "Invalid query feature vector.",
            }

        if not enrolled_users:
            return {
                "recognized": False,
                "matched_user": None,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "all_scores": [],
                "message": "No enrolled users in database. Please enroll a user first.",
            }

        best_user = None
        best_score = -1.0
        candidate_scores = []

        for user in enrolled_users:
            user_templates = user.get("templates", [])
            if not user_templates:
                continue

            # Compute similarity against each enrolled sample of this user
            sample_scores = [
                self.cosine_similarity(query_embedding, tmpl) for tmpl in user_templates
            ]
            # Max-score across enrollment samples for the user
            user_best_score = max(sample_scores) if sample_scores else 0.0

            candidate_scores.append(
                {
                    "user_id": user["id"],
                    "name": user["name"],
                    "usn": user.get("usn", ""),
                    "score": round(user_best_score, 4),
                }
            )

            if user_best_score > best_score:
                best_score = user_best_score
                best_user = user

        # Sort candidates by score descending
        candidate_scores.sort(key=lambda x: x["score"], reverse=True)

        is_recognized = (best_score >= self.threshold) and (best_user is not None)

        if is_recognized:
            msg = f"Palm Recognized: {best_user['name']} (Score: {best_score:.4f} >= Threshold: {self.threshold:.2f})"
        else:
            if best_user:
                msg = f"Palm Not Recognized (Best Match: {best_user['name']} with Score: {best_score:.4f} < Threshold: {self.threshold:.2f})"
            else:
                msg = "Palm Not Recognized (No matching template)."

        return {
            "recognized": is_recognized,
            "matched_user": (
                {
                    "id": best_user["id"],
                    "name": best_user["name"],
                    "usn": best_user.get("usn", ""),
                }
                if is_recognized
                else None
            ),
            "best_candidate": (
                {
                    "id": best_user["id"],
                    "name": best_user["name"],
                    "usn": best_user.get("usn", ""),
                }
                if best_user
                else None
            ),
            "similarity_score": round(max(best_score, 0.0), 4),
            "threshold_used": self.threshold,
            "all_scores": candidate_scores,
            "message": msg,
        }
