"""
Biometric Evaluation and Threshold Calibration Engine.
Performs empirical cross-matching on enrolled samples to compute:
- Genuine & Impostor Score Distributions
- Genuine Acceptance Rate (GAR / TAR)
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Equal Error Rate (EER) point
- CSV evaluation report generation

All metrics are computed strictly from empirical samples without hardcoding or fabricating results.
"""

import csv
import io
from typing import Dict, List, Optional, Tuple
import numpy as np

import config
from core.geometry import geometry_distance
from core.matcher import PalmMatcher


class EvaluationEngine:
    """
    Evaluates biometric performance across genuine and impostor comparison pairs.
    """

    def __init__(self, matcher: Optional[PalmMatcher] = None):
        self.matcher = matcher if matcher is not None else PalmMatcher()

    def evaluate_enrolled_dataset(self, users_data: List[Dict]) -> Dict:
        """
        Conducts cross-comparison across all enrolled users and their template samples.

        Args:
            users_data: List of user dicts containing 'id', 'name', 'usn', and 'templates' (list of np.ndarray).

        Returns:
            Dict containing empirical evaluation results, distributions, threshold sweep, and dataset metadata.
        """
        total_users = len(users_data)
        total_samples = sum(len(u.get("templates", [])) for u in users_data)

        if total_users == 0 or total_samples == 0:
            return {
                "status": "insufficient_data",
                "message": "No enrolled users or samples found in database. Enroll at least 2 users with multiple samples each to conduct biometric evaluation.",
                "total_users": total_users,
                "total_samples": total_samples,
                "genuine_comparisons": 0,
                "impostor_comparisons": 0,
                "genuine_stats": {},
                "impostor_stats": {},
                "threshold_sweep": [],
                "eer_estimate": None,
                "suggested_threshold": config.DEFAULT_RECOGNITION_THRESHOLD,
                "detailed_genuine": [],
                "detailed_impostor": [],
            }

        genuine_scores = []
        impostor_scores = []

        # 1. Collect Genuine Pairs (same user, distinct sample pairs)
        for user in users_data:
            templates = user.get("templates", [])
            num_t = len(templates)
            for i in range(num_t):
                for j in range(i + 1, num_t):
                    score = self.matcher.cosine_similarity(templates[i], templates[j])
                    genuine_scores.append(
                        {
                            "user_id": user["id"],
                            "user_name": user["name"],
                            "sample_i": i + 1,
                            "sample_j": j + 1,
                            "score": float(round(score, 4)),
                        }
                    )

        # 2. Collect Impostor Pairs (different users, all sample pairs)
        for i in range(total_users):
            user_a = users_data[i]
            templates_a = user_a.get("templates", [])
            for j in range(i + 1, total_users):
                user_b = users_data[j]
                templates_b = user_b.get("templates", [])
                for sa_idx, sa in enumerate(templates_a):
                    for sb_idx, sb in enumerate(templates_b):
                        score = self.matcher.cosine_similarity(sa, sb)
                        impostor_scores.append(
                            {
                                "user_a_id": user_a["id"],
                                "user_a_name": user_a["name"],
                                "user_b_id": user_b["id"],
                                "user_b_name": user_b["name"],
                                "sample_a": sa_idx + 1,
                                "sample_b": sb_idx + 1,
                                "score": float(round(score, 4)),
                            }
                        )

        num_gen = len(genuine_scores)
        num_imp = len(impostor_scores)

        gen_values = [item["score"] for item in genuine_scores]
        imp_values = [item["score"] for item in impostor_scores]

        # Calculate statistics
        genuine_stats = {
            "count": num_gen,
            "min": float(round(np.min(gen_values), 4)) if num_gen > 0 else 0.0,
            "max": float(round(np.max(gen_values), 4)) if num_gen > 0 else 0.0,
            "mean": float(round(np.mean(gen_values), 4)) if num_gen > 0 else 0.0,
            "std": float(round(np.std(gen_values), 4)) if num_gen > 0 else 0.0,
        }

        impostor_stats = {
            "count": num_imp,
            "min": float(round(np.min(imp_values), 4)) if num_imp > 0 else 0.0,
            "max": float(round(np.max(imp_values), 4)) if num_imp > 0 else 0.0,
            "mean": float(round(np.mean(imp_values), 4)) if num_imp > 0 else 0.0,
            "std": float(round(np.std(imp_values), 4)) if num_imp > 0 else 0.0,
        }

        # 3. Sweep thresholds from 0.00 to 1.00 in steps of 0.02
        threshold_sweep = []
        best_eer_diff = 999.0
        eer_point = None
        min_total_error = 999999
        suggested_t = config.DEFAULT_RECOGNITION_THRESHOLD
        suggestion_policy = "default_uncalibrated"

        for t_int in range(0, 101, 2):  # 0.00, 0.02, ..., 1.00
            t = round(t_int / 100.0, 2)

            # Genuine outcomes
            gen_accept = sum(1 for s in gen_values if s >= t) if num_gen > 0 else 0
            gen_reject = num_gen - gen_accept if num_gen > 0 else 0

            # Impostor outcomes
            imp_accept = sum(1 for s in imp_values if s >= t) if num_imp > 0 else 0
            imp_reject = num_imp - imp_accept if num_imp > 0 else 0

            gar = (gen_accept / num_gen) if num_gen > 0 else 0.0
            frr = (gen_reject / num_gen) if num_gen > 0 else 0.0
            far = (imp_accept / num_imp) if num_imp > 0 else 0.0
            trr = (imp_reject / num_imp) if num_imp > 0 else 0.0

            total_errors = gen_reject + imp_accept
            total_trials = num_gen + num_imp
            accuracy = ((gen_accept + imp_reject) / total_trials) if total_trials > 0 else 0.0

            point = {
                "threshold": t,
                "gar": round(gar, 4),
                "frr": round(frr, 4),
                "far": round(far, 4),
                "trr": round(trr, 4),
                "gen_accept_count": gen_accept,
                "gen_reject_count": gen_reject,
                "imp_accept_count": imp_accept,
                "imp_reject_count": imp_reject,
                "accuracy": round(accuracy, 4),
            }
            threshold_sweep.append(point)

            # EER approximation (|FAR - FRR| min)
            if num_gen > 0 and num_imp > 0:
                diff = abs(far - frr)
                if diff < best_eer_diff:
                    best_eer_diff = diff
                    eer_point = {
                        "threshold": t,
                        "eer_value": round((far + frr) / 2.0, 4),
                        "far": round(far, 4),
                        "frr": round(frr, 4),
                    }

                # Legacy fallback heuristic: minimum total errors with low FAR.
                if total_errors < min_total_error and far <= 0.05:
                    min_total_error = total_errors
                    suggested_t = t
                    suggestion_policy = "min_error_far_le_5pct"

        # ---- Strict operating-point policy (Accuracy Sprint) ----
        # The old rule tolerated up to 5% false accepts — unacceptable for a
        # payment demo where an unregistered palm must NEVER pass. New rule:
        # pick the HIGHEST threshold with ZERO observed impostor accepts while
        # still keeping genuine acceptance >= EVAL_SUGGESTION_MIN_GAR. This sits
        # just below the weakest genuine pair and as far as possible above the
        # entire observed impostor distribution.
        zero_far_points = [
            p for p in threshold_sweep
            if p["imp_accept_count"] == 0 and p["gar"] >= config.EVAL_SUGGESTION_MIN_GAR
        ]
        if zero_far_points:
            suggested_t = max(p["threshold"] for p in zero_far_points)
            suggestion_policy = f"zero_far_gar_ge_{int(config.EVAL_SUGGESTION_MIN_GAR * 100)}pct"

        # ---- Hand-geometry evaluation (Accuracy Sprint v4) ----
        # Distance metric: LOWER is better. Genuine = leave-one-out distance of
        # each enrollment sample against its user's remaining samples' stats;
        # impostor = distance against every OTHER user's stats. Suggested
        # GEO_MAX_DIST = the largest D with zero impostor passes while keeping
        # genuine acceptance >= EVAL_SUGGESTION_MIN_GAR.
        geometry_eval = self._evaluate_geometry(users_data)

        return {
            "status": "success",
            "message": "Evaluation completed on current empirical dataset.",
            "total_users": total_users,
            "total_samples": total_samples,
            "genuine_comparisons": num_gen,
            "impostor_comparisons": num_imp,
            "genuine_stats": genuine_stats,
            "impostor_stats": impostor_stats,
            "threshold_sweep": threshold_sweep,
            "eer_estimate": eer_point,
            "suggested_threshold": suggested_t,
            "suggestion_policy": suggestion_policy,
            "geometry_eval": geometry_eval,
            "detailed_genuine": genuine_scores,
            "detailed_impostor": impostor_scores,
        }

    @staticmethod
    def _evaluate_geometry(users_data: List[Dict]) -> Dict:
        """
        Computes genuine/impostor hand-geometry distance distributions and the
        suggested GEO_MAX_DIST under the strict zero-FAR policy. Returns an
        empty (insufficient_data) payload when geometry is unavailable.
        """
        payload = {
            "available": False,
            "genuine_stats": {},
            "impostor_stats": {},
            "suggested_geo_max_dist": config.GEO_MAX_DIST,
            "suggestion_policy": "default_uncalibrated",
            "genuine_distances": [],
            "impostor_distances": [],
        }

        users_with_geo = [u for u in users_data if len(u.get("geometry") or []) >= 2]
        if len(users_with_geo) < 1:
            return payload

        def _stats(samples):
            stacked = np.stack([np.asarray(g, dtype=np.float64) for g in samples], axis=0)
            return stacked.mean(axis=0), stacked.std(axis=0)

        genuine_d: List[float] = []
        impostor_d: List[float] = []

        for user in users_with_geo:
            geos = user["geometry"]
            # Genuine: leave-one-out so the query never contributes to its own stats.
            for i in range(len(geos)):
                rest = [g for j, g in enumerate(geos) if j != i]
                mu, sigma = _stats(rest)
                genuine_d.append(geometry_distance(np.asarray(geos[i]), mu, sigma))

        for user_a in users_with_geo:
            for user_b in users_with_geo:
                if user_a["id"] == user_b["id"]:
                    continue
                mu, sigma = _stats(user_b["geometry"])
                for g in user_a["geometry"]:
                    impostor_d.append(geometry_distance(np.asarray(g), mu, sigma))

        if not genuine_d or not impostor_d:
            return payload

        genuine_d_arr = np.array(genuine_d)
        impostor_d_arr = np.array(impostor_d)

        def _stats_payload(arr: np.ndarray) -> Dict:
            return {
                "count": int(arr.size),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
            }

        payload["available"] = True
        payload["genuine_stats"] = _stats_payload(genuine_d_arr)
        payload["impostor_stats"] = _stats_payload(impostor_d_arr)
        payload["genuine_distances"] = [round(float(d), 4) for d in genuine_d_arr]
        payload["impostor_distances"] = [round(float(d), 4) for d in impostor_d_arr]

        # Zero-FAR suggestion for a distance metric: D must sit BELOW every
        # observed impostor distance; pick the largest 0.05-step D that keeps
        # genuine acceptance >= EVAL_SUGGESTION_MIN_GAR.
        min_impostor = float(impostor_d_arr.min())
        candidate = None
        step = 0.05
        d = float(np.floor(min_impostor / step) * step)
        while d >= 0.0:
            gar = float((genuine_d_arr < d).mean())
            if gar >= config.EVAL_SUGGESTION_MIN_GAR:
                candidate = round(d, 2)
                break
            d = round(d - step, 2)

        if candidate is not None and candidate > 0:
            payload["suggested_geo_max_dist"] = candidate
            payload["suggestion_policy"] = (
                f"geo_zero_far_gar_ge_{int(config.EVAL_SUGGESTION_MIN_GAR * 100)}pct"
            )

        return payload

    @staticmethod
    def generate_csv_report(eval_result: Dict) -> str:
        """
        Generates a structured CSV report summarizing evaluation data, score sweeps, and comparison pairs.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header Metadata
        writer.writerow(["# PalmPay Biometric Evaluation Report (Phase 1)"])
        writer.writerow(["# Model Name", config.MODEL_NAME])
        writer.writerow(["# Total Enrolled Users", eval_result.get("total_users", 0)])
        writer.writerow(["# Total Biometric Samples", eval_result.get("total_samples", 0)])
        writer.writerow(["# Genuine Comparisons", eval_result.get("genuine_comparisons", 0)])
        writer.writerow(["# Impostor Comparisons", eval_result.get("impostor_comparisons", 0)])
        writer.writerow([])

        # Summary Statistics
        writer.writerow(["Category", "Count", "Min Score", "Max Score", "Mean Score", "Std Dev"])
        gs = eval_result.get("genuine_stats", {})
        writer.writerow(
            ["Genuine Attempts", gs.get("count", 0), gs.get("min", 0), gs.get("max", 0), gs.get("mean", 0), gs.get("std", 0)]
        )
        ims = eval_result.get("impostor_stats", {})
        writer.writerow(
            ["Impostor Attempts", ims.get("count", 0), ims.get("min", 0), ims.get("max", 0), ims.get("mean", 0), ims.get("std", 0)]
        )
        writer.writerow([])

        # Threshold Sweep Table
        writer.writerow(
            [
                "Threshold",
                "GAR (Genuine Accept Rate)",
                "FRR (False Reject Rate)",
                "FAR (False Accept Rate)",
                "Accuracy",
                "Genuine Accept Count",
                "False Reject Count",
                "False Accept Count",
                "True Reject Count",
            ]
        )
        for pt in eval_result.get("threshold_sweep", []):
            writer.writerow(
                [
                    pt["threshold"],
                    pt["gar"],
                    pt["frr"],
                    pt["far"],
                    pt["accuracy"],
                    pt["gen_accept_count"],
                    pt["gen_reject_count"],
                    pt["imp_accept_count"],
                    pt["imp_reject_count"],
                ]
            )

        writer.writerow([])
        # Detailed Genuine Pairs
        writer.writerow(["# Detailed Genuine Pairs"])
        writer.writerow(["User ID", "User Name", "Sample A", "Sample B", "Similarity Score"])
        for g in eval_result.get("detailed_genuine", []):
            writer.writerow([g["user_id"], g["user_name"], g["sample_i"], g["sample_j"], g["score"]])

        writer.writerow([])
        # Detailed Impostor Pairs
        writer.writerow(["# Detailed Impostor Pairs"])
        writer.writerow(["User A ID", "User A Name", "User B ID", "User B Name", "Sample A", "Sample B", "Similarity Score"])
        for imp in eval_result.get("detailed_impostor", []):
            writer.writerow([imp["user_a_id"], imp["user_a_name"], imp["user_b_id"], imp["user_b_name"], imp["sample_a"], imp["sample_b"], imp["score"]])

        return output.getvalue()
