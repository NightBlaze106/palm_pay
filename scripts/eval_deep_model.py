#!/usr/bin/env python3
"""
PalmPay Biometric Evaluation & Head-to-Head Comparison Script.
Evaluates Classical (HOG+LBP or CompCode) vs Deep Learning (MobileNetV3) models.

Computes:
- Genuine & impostor score distributions (mean, std, min, max)
- Equal Error Rate (EER) via fine-grained empirical sweep
- Zero-FAR operating threshold with GAR >= 90%
- Head-to-head comparison table
- Strict safety enforcement: NEVER auto-applies thresholds to config or DB.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.evaluation import EvaluationEngine
from core.matcher import PalmMatcher
from core.palm_model import ClassicalPalmModel, DeepPalmModel
from training.dataset import PalmDataset


def generate_synthetic_evaluation_rois(
    num_subjects: int = 6,
    samples_per_subject: int = 5,
    seed: int = 42,
) -> Dict[str, List[np.ndarray]]:
    """
    Generates deterministic synthetic palm ROI images for offline evaluation / benchmarking
    when no physical pilot images are captured yet.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for s_idx in range(1, num_subjects + 1):
        subj_id = f"synthetic_user_{s_idx:03d}"
        # Subject-specific distinctive pattern
        base_texture = rng.integers(70, 180, (128, 128), dtype=np.uint8)
        y, x = np.ogrid[:128, :128]
        # Principal lines varying per subject
        line1 = np.abs(y - (x * 0.7 + s_idx * 12)) < 3
        line2 = np.abs(y - (128 - x * 0.5 - s_idx * 8)) < 2
        base_texture[line1] = np.clip(base_texture[line1].astype(int) - 60, 0, 255).astype(np.uint8)
        base_texture[line2] = np.clip(base_texture[line2].astype(int) - 40, 0, 255).astype(np.uint8)

        samples = []
        for _ in range(samples_per_subject):
            noise = rng.normal(0, 3, (128, 128)).astype(np.int16)
            sample = np.clip(base_texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            samples.append(sample)
        data[subj_id] = samples
    return data


def load_rois_from_directory(data_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Discovers pilot ROI samples from data_dir and loads preprocessed 128x128 images.
    """
    ds = PalmDataset(data_dir=data_dir)
    subjects = ds.discover_subjects()
    data = {}
    for s_id, items in subjects.items():
        loaded_imgs = []
        for it in items:
            img = ds.load_and_preprocess(it["image_path"])
            loaded_imgs.append(img)
        if len(loaded_imgs) >= 2:
            data[s_id] = loaded_imgs
    return data


def extract_subject_templates(
    model,
    subjects_rois: Dict[str, List[np.ndarray]],
) -> List[Dict]:
    """
    Runs feature extraction across all subjects and formats for EvaluationEngine.
    """
    user_payloads = []
    for s_id, rois in subjects_rois.items():
        templates = []
        for roi in rois:
            vec = model.extract_embedding(roi)
            templates.append(vec)
        user_payloads.append({
            "id": s_id,
            "name": f"Subject_{s_id}",
            "templates": templates,
        })
    return user_payloads


def compute_fine_metrics(genuine_scores: List[float], impostor_scores: List[float]) -> Dict:
    """
    Computes fine-grained EER and Zero-FAR metrics from raw score arrays.
    """
    if not genuine_scores or not impostor_scores:
        return {"eer": None, "zero_far_threshold": None, "zero_far_gar": 0.0}

    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)

    # 1. Zero-FAR Operating Point:
    # Max impostor score is upper bound of impostors.
    max_imp = float(imp_arr.max())
    # Find genuine samples >= max_imp
    gen_above_max_imp = gen_arr[gen_arr > max_imp]
    gar_at_zero_far = len(gen_above_max_imp) / len(gen_arr)
    zero_far_t = max_imp + 1e-4

    # 2. Fine-grained EER
    all_scores = np.sort(np.concatenate([gen_arr, imp_arr]))
    # Sample thresholds across range
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)
    best_diff = 1.0
    eer_pt = {"threshold": 0.0, "eer_value": 0.0, "far": 0.0, "frr": 0.0}

    for t in thresholds:
        far = np.mean(imp_arr >= t)
        frr = np.mean(gen_arr < t)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_pt = {
                "threshold": round(float(t), 4),
                "eer_value": round(float((far + frr) / 2.0), 4),
                "far": round(float(far), 4),
                "frr": round(float(frr), 4),
            }

    return {
        "eer": eer_pt,
        "zero_far_threshold": round(zero_far_t, 4),
        "zero_far_gar": round(float(gar_at_zero_far), 4),
    }


def run_model_eval(model_name: str, model_instance, subjects_rois: Dict[str, List[np.ndarray]]) -> Dict:
    """
    Evaluates a specific model instance on the ROI dataset.
    """
    users_data = extract_subject_templates(model_instance, subjects_rois)
    engine = EvaluationEngine(matcher=PalmMatcher(threshold=config.DEFAULT_RECOGNITION_THRESHOLD))
    res = engine.evaluate_enrolled_dataset(users_data)
    res["model_name"] = model_name
    res["feature_dim"] = model_instance.get_feature_dim()

    gen_scores = [d["score"] for d in res.get("detailed_genuine", [])]
    imp_scores = [d["score"] for d in res.get("detailed_impostor", [])]
    fine = compute_fine_metrics(gen_scores, imp_scores)
    res["fine_metrics"] = fine
    return res


def print_comparison_table(classical_res: Dict, deep_res: Dict, is_synthetic: bool = False):
    """
    Prints a formatted head-to-head comparison table between Classical and Deep models.
    """
    if is_synthetic:
        print("\n" + "=" * 84)
        print("⚠️  SYNTHETIC MODE")
        print("Scores are generated deterministic test data and do NOT represent")
        print("real palm-recognition performance.")
        print("Real metrics require captured pilot palm data.")
        print("=" * 84)

    c_gen = classical_res.get("genuine_stats", {})
    c_imp = classical_res.get("impostor_stats", {})
    d_gen = deep_res.get("genuine_stats", {})
    d_imp = deep_res.get("impostor_stats", {})

    c_fine = classical_res.get("fine_metrics", {})
    d_fine = deep_res.get("fine_metrics", {})

    c_eer = c_fine.get("eer") or classical_res.get("eer_estimate") or {}
    d_eer = d_fine.get("eer") or deep_res.get("eer_estimate") or {}

    c_sep = (c_gen.get("mean", 0.0) - c_imp.get("mean", 0.0))
    d_sep = (d_gen.get("mean", 0.0) - d_imp.get("mean", 0.0))

    c_zero_t = c_fine.get("zero_far_threshold", "N/A")
    c_zero_gar = c_fine.get("zero_far_gar", 0.0)
    d_zero_t = d_fine.get("zero_far_threshold", "N/A")
    d_zero_gar = d_fine.get("zero_far_gar", 0.0)

    print("\n" + "=" * 84)
    if is_synthetic:
        print("  [SYNTHETIC TEST BENCHMARK — CI/SCAFFOLD ONLY — NOT PRODUCTION METRICS]")
    print("            PALMPAY BIOMETRIC BENCHMARK: CLASSICAL VS DEEP HEAD-TO-HEAD")
    print("=" * 84)
    hdr = f"{'Metric':<32} | {'Classical (' + str(classical_res['model_name'])[:11] + ')':<23} | {'Deep (' + str(deep_res['model_name'])[:11] + ')':<23}"
    print(hdr)
    print("-" * 84)
    print(f"{'Embedding Dimension':<32} | {str(classical_res['feature_dim']) + '-D':<23} | {str(deep_res['feature_dim']) + '-D':<23}")
    print(f"{'Genuine Comparisons':<32} | {classical_res.get('genuine_comparisons', 0):<23} | {deep_res.get('genuine_comparisons', 0):<23}")
    print(f"{'Impostor Comparisons':<32} | {classical_res.get('impostor_comparisons', 0):<23} | {deep_res.get('impostor_comparisons', 0):<23}")
    print("-" * 84)
    print(f"{'Genuine Mean ± Std':<32} | {c_gen.get('mean', 0.0):.4f} ± {c_gen.get('std', 0.0):.4f}       | {d_gen.get('mean', 0.0):.4f} ± {d_gen.get('std', 0.0):.4f}      ")
    print(f"{'Genuine [Min, Max]':<32} | [{c_gen.get('min', 0.0):.4f}, {c_gen.get('max', 0.0):.4f}]         | [{d_gen.get('min', 0.0):.4f}, {d_gen.get('max', 0.0):.4f}]        ")
    print(f"{'Impostor Mean ± Std':<32} | {c_imp.get('mean', 0.0):.4f} ± {c_imp.get('std', 0.0):.4f}       | {d_imp.get('mean', 0.0):.4f} ± {d_imp.get('std', 0.0):.4f}      ")
    print(f"{'Impostor [Min, Max]':<32} | [{c_imp.get('min', 0.0):.4f}, {c_imp.get('max', 0.0):.4f}]         | [{d_imp.get('min', 0.0):.4f}, {d_imp.get('max', 0.0):.4f}]        ")
    print(f"{'Separation (Gen μ - Imp μ)':<32} | {c_sep:+.4f}                 | {d_sep:+.4f}                ")
    print("-" * 84)
    c_eer_val = f"{c_eer.get('eer_value', 0.0) * 100:.2f}% (T={c_eer.get('threshold', 0.0):.4f})" if c_eer else "N/A"
    d_eer_val = f"{d_eer.get('eer_value', 0.0) * 100:.2f}% (T={d_eer.get('threshold', 0.0):.4f})" if d_eer else "N/A"
    print(f"{'Equal Error Rate (EER)':<32} | {c_eer_val:<23} | {d_eer_val:<23}")

    c_zero_str = f"T={c_zero_t} (GAR={c_zero_gar*100:.1f}%)" if isinstance(c_zero_t, (int, float)) else str(c_zero_t)
    d_zero_str = f"T={d_zero_t} (GAR={d_zero_gar*100:.1f}%)" if isinstance(d_zero_t, (int, float)) else str(d_zero_t)
    print(f"{'Zero-FAR Operating Point':<32} | {c_zero_str:<23} | {d_zero_str:<23}")
    print("=" * 84)
    print("SAFETY & POLICY NOTICE:")
    print("  * Evaluated on empirical ROI distributions.")
    print("  * Threshold suggestions are for analysis and documentation only.")
    print("  * Thresholds are NOT automatically applied to config.py or production DB.")
    print("  * Raw biometric images are never persisted.")
    print("=" * 84 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PalmPay Classical vs Deep Biometric Evaluation")
    parser.add_argument("--data-dir", type=str, default=config.DATASET_DIR, help="Path to pilot ROI directory")
    parser.add_argument("--synthetic", action="store_true", help="Run benchmark on synthetic dataset if no pilot ROIs exist")
    parser.add_argument("--num-synthetic-users", type=int, default=6, help="Number of synthetic users for benchmark")
    parser.add_argument("--csv", type=str, default=None, help="Optional output path for evaluation CSV report")
    args = parser.parse_args()

    # Discover dataset
    subjects_rois = {}
    is_synthetic = False
    if os.path.exists(args.data_dir):
        subjects_rois = load_rois_from_directory(args.data_dir)

    if not subjects_rois or len(subjects_rois) < 2:
        if args.synthetic or not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
            is_synthetic = True
            print(f"[Eval] No captured pilot ROIs found in {args.data_dir}.")
            print(f"[Eval] Generating {args.num_synthetic_users} synthetic evaluation subjects (5 samples each)...")
            subjects_rois = generate_synthetic_evaluation_rois(
                num_subjects=args.num_synthetic_users,
                samples_per_subject=5,
            )
        else:
            print(f"[Eval] Need at least 2 subjects with >=2 samples in {args.data_dir} (found {len(subjects_rois)}).")
            print("[Eval] To benchmark with synthetic samples, re-run with --synthetic")
            sys.exit(1)

    num_subjects = len(subjects_rois)
    total_rois = sum(len(v) for v in subjects_rois.values())
    print(f"[Eval] Evaluating {num_subjects} subjects across {total_rois} ROI samples.")

    # 1. Classical Model
    classical_model = ClassicalPalmModel()
    c_res = run_model_eval(classical_model.get_model_name(), classical_model, subjects_rois)

    # 2. Deep Model
    deep_model = DeepPalmModel()
    d_res = run_model_eval(deep_model.get_model_name(), deep_model, subjects_rois)

    # 3. Print Head-to-Head Comparison Table
    print_comparison_table(c_res, d_res, is_synthetic=is_synthetic)

    # 4. Optional CSV Report export
    if args.csv:
        c_csv = EvaluationEngine.generate_csv_report(c_res)
        with open(args.csv, "w") as f:
            f.write(c_csv)
        print(f"[Eval] CSV report written to: {args.csv}")


if __name__ == "__main__":
    main()
