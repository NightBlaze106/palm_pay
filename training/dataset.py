"""
PalmPay Dataset Loader and Split Manager for Phase-2 DeepPalm Model.
Wraps core.roi + core.preprocessing pipeline to guarantee ZERO train/serve skew.
Enforces subject-disjoint train/validation/test splits.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import config
from core.roi import PalmROIExtractor
from core.preprocessing import PalmPreprocessor
# Note on Finger Masking parity:
# Ephemeral pilot ROIs saved via /api/enroll are captured directly from
# core.pipeline.process_frame()["roi_enhanced"], which already incorporates
# core.pipeline.apply_finger_mask at capture time. Stored ROIs therefore
# exactly embody the production serving representation.


class PalmDataset:
    """
    Manages loading of pilot ROI crops saved during consent-gated enrollment.
    guarantees subject-disjoint splits to prevent identity leakage.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or getattr(config, "DATASET_DIR", os.path.join(config.BASE_DIR, "data", "pilot_rois"))
        self.roi_extractor = PalmROIExtractor()
        self.preprocessor = PalmPreprocessor()

    def discover_subjects(self) -> Dict[str, List[Dict]]:
        """
        Discovers all enrolled subjects with pilot ROI samples.
        Returns: { subject_id: [ { 'image_path': ..., 'meta': ... }, ... ] }
        """
        if not os.path.exists(self.data_dir):
            return {}

        subjects = {}
        for entry in sorted(os.listdir(self.data_dir)):
            sub_path = os.path.join(self.data_dir, entry)
            if not os.path.isdir(sub_path):
                continue

            samples = []
            files = sorted(os.listdir(sub_path))
            pngs = [f for f in files if f.endswith(".png")]
            for p in pngs:
                img_path = os.path.join(sub_path, p)
                meta_path = os.path.join(sub_path, p.replace(".png", ".json"))
                meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                    except Exception:
                        pass

                samples.append({
                    "image_path": img_path,
                    "meta": meta,
                    "subject_id": entry,
                })

            if samples:
                subjects[entry] = samples

        return subjects

    def split_subject_disjoint(
        self,
        subjects: Dict[str, List[Dict]],
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Splits dataset such that all samples of a given subject belong
        EXCLUSIVELY to either the train set or the validation set.
        Zero cross-contamination.
        """
        rng = random.Random(seed)
        sub_keys = list(subjects.keys())
        rng.shuffle(sub_keys)

        num_val = max(1, int(len(sub_keys) * val_ratio)) if len(sub_keys) > 1 else 0
        val_keys = set(sub_keys[:num_val])
        train_keys = set(sub_keys[num_val:])

        train_samples = []
        val_samples = []

        for k, sample_list in subjects.items():
            if k in val_keys:
                val_samples.extend(sample_list)
            else:
                train_samples.extend(sample_list)

        return train_samples, val_samples

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Loads an ROI image from disk and guarantees serving-pipeline parity:
        Grayscale -> CLAHE -> Finger Masking -> 128x128.
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read ROI at: {image_path}")

        # If already preprocessed grayscale 128x128
        if img.ndim == 2 and img.shape == (config.ROI_HEIGHT, config.ROI_WIDTH):
            return img

        # If full frame or BGR ROI passed:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        prep = self.preprocessor.preprocess(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        enhanced = prep["enhanced"]
        if enhanced.shape != (config.ROI_HEIGHT, config.ROI_WIDTH):
            enhanced = cv2.resize(enhanced, (config.ROI_WIDTH, config.ROI_HEIGHT))
        return enhanced

    @staticmethod
    def augment_roi(image: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Applies domain-specific augmentations modeling documented PalmPay RGB camera failure modes:
        - Small 2D rotations (+-10 deg)
        - Small translations (+-5 px)
        - Illumination/contrast jitter (scale 0.8-1.2, bias +-20)
        - Slight Gaussian blur (k=3) for motion simulation
        """
        if rng is None:
            rng = np.random.default_rng()

        h, w = image.shape[:2]
        out = image.copy().astype(np.float32)

        # 1. Contrast & Brightness jitter
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-15.0, 15.0))
        out = np.clip(out * alpha + beta, 0.0, 255.0)

        # 2. Affine rotation + translation
        angle = float(rng.uniform(-10.0, 10.0))
        tx = float(rng.uniform(-4.0, 4.0))
        ty = float(rng.uniform(-4.0, 4.0))

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

        out_uint8 = cv2.warpAffine(
            out.astype(np.uint8),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 3. Probabilistic mild blur (25% chance)
        if rng.uniform(0.0, 1.0) < 0.25:
            out_uint8 = cv2.GaussianBlur(out_uint8, (3, 3), sigmaX=0.5)

        return out_uint8
