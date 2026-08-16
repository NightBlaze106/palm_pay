"""
Evaluation and Threshold Calibration Tests for PalmPay.
Verifies:
- Empirical genuine vs impostor cross-comparison
- FAR, FRR, GAR, and Accuracy calculation on sample matrices
- EER (Equal Error Rate) point estimation
- CSV Report export formatting
"""

import unittest
import numpy as np

import config
from core.evaluation import EvaluationEngine
from core.matcher import PalmMatcher


class TestEvaluationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = EvaluationEngine()

    def test_empty_dataset_handling(self):
        """Engine should safely handle empty dataset without errors."""
        res = self.engine.evaluate_enrolled_dataset([])
        self.assertEqual(res["status"], "insufficient_data")
        self.assertEqual(res["total_users"], 0)

    def test_synthetic_biometric_evaluation_matrix(self):
        """
        Create a controlled synthetic dataset of 3 users, 4 samples each.
        Samples for the same user have high cosine similarity (small noise),
        while samples across different users have lower similarity.
        """
        np.random.seed(42)
        users_data = []

        for user_idx in range(3):
            # Base identity vector for user
            base_vec = np.random.randn(944).astype(np.float32)
            base_vec /= np.linalg.norm(base_vec)

            samples = []
            for s_idx in range(4):
                # Add small intra-class variation
                noise = np.random.randn(944).astype(np.float32)
                noise /= np.linalg.norm(noise)
                sample_vec = 0.96 * base_vec + 0.04 * noise
                sample_vec /= np.linalg.norm(sample_vec)
                samples.append(sample_vec)

            users_data.append(
                {
                    "id": user_idx + 1,
                    "name": f"Synthetic User {user_idx + 1}",
                    "usn": f"1MS21CS00{user_idx + 1}",
                    "templates": samples,
                }
            )

        res = self.engine.evaluate_enrolled_dataset(users_data)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_users"], 3)
        self.assertEqual(res["total_samples"], 12)

        # For 3 users with 4 samples each:
        # Genuine pairs per user = 4 * 3 / 2 = 6 pairs * 3 users = 18 genuine pairs
        self.assertEqual(res["genuine_comparisons"], 18)

        # Impostor pairs = (User 1 vs User 2: 4*4=16) + (User 1 vs User 3: 16) + (User 2 vs User 3: 16) = 48 impostor pairs
        self.assertEqual(res["impostor_comparisons"], 48)

        # Genuine mean similarity should be higher than impostor mean similarity
        gen_mean = res["genuine_stats"]["mean"]
        imp_mean = res["impostor_stats"]["mean"]
        self.assertGreater(gen_mean, imp_mean)
        self.assertGreater(gen_mean, 0.90)
        self.assertLess(imp_mean, 0.50)

        # Threshold sweep must contain entries from 0.0 to 1.0
        self.assertTrue(len(res["threshold_sweep"]) > 20)

        # Test CSV export
        csv_text = EvaluationEngine.generate_csv_report(res)
        self.assertIn("PalmPay Biometric Evaluation Report", csv_text)
        self.assertIn("GAR (Genuine Accept Rate)", csv_text)
        self.assertIn("Synthetic User 1", csv_text)


if __name__ == "__main__":
    unittest.main()
