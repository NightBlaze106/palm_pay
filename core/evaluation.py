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

                # Suggested threshold based on minimum total errors with low FAR
                if total_errors < min_total_error and far <= 0.05:
                    min_total_error = total_errors
                    suggested_t = t

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
            "detailed_genuine": genuine_scores,
            "detailed_impostor": impostor_scores,
        }

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
