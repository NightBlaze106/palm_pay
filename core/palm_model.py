"""
Palm Model Abstraction and Feature Extractor Interface.
Provides a modular interface for biometric embedding extraction.
"""

from abc import ABC, abstractmethod
import hashlib
import os
from typing import Dict, Optional
import cv2
import numpy as np

import config
from core.features import CompCodeFeatureExtractor, HogLbpFusionFeatureExtractor, MBuLBPFeatureExtractor

class BasePalmModel(ABC):
    """
    Abstract Base Class for Palm Biometric Feature Extractors.
    """

    @abstractmethod
    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

class ClassicalPalmModel(BasePalmModel):
    """
    Standard Phase-1 Feature Extractor:
    Selectable via config.MODEL_NAME (CompCode vs Fusion vs MB-uLBP).
    """

    def __init__(self):
        if "HogLbp" in config.MODEL_NAME:
            self.extractor = HogLbpFusionFeatureExtractor()
        elif "MB-uLBP" in config.MODEL_NAME:
            self.extractor = MBuLBPFeatureExtractor()
        else:
            self.extractor = CompCodeFeatureExtractor()
            
        self.model_name = config.MODEL_NAME
        self.feature_dim = self.extractor.expected_dim

    def extract(self, roi_gray_uint8: np.ndarray) -> np.ndarray:
        return self.extract_embedding(roi_gray_uint8)

    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        return self.extractor.extract_features(preprocessed_roi)

    def get_feature_dim(self) -> int:
        return self.extractor.expected_dim

    def get_model_name(self) -> str:
        return self.model_name

class DeepPalmModel(BasePalmModel):
    """
    Phase-2 Deep Learning Palm Biometric Model.
    Uses ONNX Runtime if available and a valid model file is present.
    Gracefully falls back to a deterministic pseudo-embedder for offline tests/development
    when ONNX Runtime or the model file is not present.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        feature_dim: Optional[int] = None,
        input_size: Optional[int] = None,
    ):
        self.model_path = model_path or getattr(config, "DEEP_MODEL_PATH", "models/palm_embedder.onnx")
        self.model_name = model_name or getattr(config, "DEEP_MODEL_NAME", "deep-palm-embedder-v1")
        self.feature_dim = feature_dim or getattr(config, "DEEP_FEATURE_DIM", 512)
        self.input_size = input_size or getattr(config, "DEEP_INPUT_SIZE", 128)

        self._session = None
        self._input_name = None
        self._output_name = None
        self.is_real_onnx = False

        self._init_session()

    def _init_session(self):
        resolved_path = self.model_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(config.BASE_DIR, resolved_path)

        if os.path.exists(resolved_path):
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(
                    resolved_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self.is_real_onnx = True
            except Exception:
                self._session = None
                self.is_real_onnx = False

    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        """
        Produces a unit-normalized float32 embedding of length self.feature_dim.
        Never persists raw biometric images.
        """
        if preprocessed_roi is None or preprocessed_roi.size == 0:
            vec = np.zeros(self.feature_dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        img = preprocessed_roi
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

        if self.is_real_onnx and self._session is not None:
            try:
                tensor = (img.astype(np.float32) / 127.5) - 1.0
                # Format to expected ONNX input shape (1-channel or 3-channel)
                inp_shape = self._session.get_inputs()[0].shape
                channels = inp_shape[1] if (len(inp_shape) > 1 and isinstance(inp_shape[1], int)) else 1
                if channels == 3:
                    tensor = np.repeat(tensor[np.newaxis, np.newaxis, :, :], 3, axis=1)
                else:
                    tensor = tensor[np.newaxis, np.newaxis, :, :]
                outputs = self._session.run([self._output_name], {self._input_name: tensor})
                emb = np.asarray(outputs[0], dtype=np.float32).flatten()
                if len(emb) != self.feature_dim:
                    if len(emb) > self.feature_dim:
                        emb = emb[: self.feature_dim]
                    else:
                        emb = np.pad(emb, (0, self.feature_dim - len(emb)))
                norm = np.linalg.norm(emb)
                if norm > 1e-7:
                    emb = emb / norm
                else:
                    emb[0] = 1.0
                return emb.astype(np.float32)
            except Exception:
                pass

        return self._deterministic_pseudo_embedding(img)

    def _deterministic_pseudo_embedding(self, gray_roi: np.ndarray) -> np.ndarray:
        """
        Fast, deterministic, continuous feature generator used for offline test suites.
        """
        h, w = gray_roi.shape[:2]
        blocks = 8
        bh, bw = max(1, h // blocks), max(1, w // blocks)
        raw_feats = []

        for r in range(blocks):
            for c in range(blocks):
                patch = gray_roi[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                m = float(np.mean(patch))
                s = float(np.std(patch))
                raw_feats.extend([m / 255.0, s / 128.0])

        raw_arr = np.array(raw_feats, dtype=np.float32)
        repeats = int(np.ceil(self.feature_dim / len(raw_arr)))
        tiled = np.tile(raw_arr, repeats)[: self.feature_dim].copy()

        h_digest = hashlib.sha256(gray_roi.tobytes()).digest()
        hash_floats = np.frombuffer(h_digest, dtype=np.uint8).astype(np.float32) / 255.0
        h_repeats = int(np.ceil(self.feature_dim / len(hash_floats)))
        h_tiled = np.tile(hash_floats, h_repeats)[: self.feature_dim]

        emb = 0.85 * tiled + 0.15 * h_tiled
        norm = np.linalg.norm(emb)
        if norm > 1e-7:
            emb = emb / norm
        else:
            emb = np.zeros(self.feature_dim, dtype=np.float32)
            emb[0] = 1.0
        return emb.astype(np.float32)

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_model_name(self) -> str:
        return self.model_name

def build_classical_model() -> ClassicalPalmModel:
    return ClassicalPalmModel()


class CcnetOnnxModel(BasePalmModel):
    model_name = "ccnet-onnx-v1"
    feature_dim = 2048

    def __init__(self, model_path: str):
        import onnxruntime as ort
        resolved_path = model_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(config.BASE_DIR, resolved_path)
        self.model_path = resolved_path
        self.session = ort.InferenceSession(
            resolved_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, roi_gray_uint8: np.ndarray) -> np.ndarray:
        # roi is 128x128 uint8 grayscale already CLAHE-processed
        # do NOT re-apply CLAHE
        if roi_gray_uint8 is None or roi_gray_uint8.size == 0:
            vec = np.zeros(self.feature_dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        img = roi_gray_uint8
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[0] != 128 or img.shape[1] != 128:
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        x = img.astype(np.float32) / 255.0
        x = x[np.newaxis, np.newaxis, :, :]        # [1,1,128,128]
        out = self.session.run(None, {self.input_name: x})[0]
        emb = out.squeeze().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        else:
            emb = np.zeros(self.feature_dim, dtype=np.float32)
            emb[0] = 1.0
        return emb

    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        return self.extract(preprocessed_roi)

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_model_name(self) -> str:
        return self.model_name


class DinoV2Model(BasePalmModel):
    model_name = "dinov2-vits14-v1"
    feature_dim = 384

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, variant: str = "dinov2_vits14"):
        import torch
        self.variant = variant
        if variant == "dinov2_vitb14":
            self.model_name = "dinov2-vitb14-v1"
            self.feature_dim = 768
        else:
            self.model_name = "dinov2-vits14-v1"
            self.feature_dim = 384

        self.torch = torch
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            variant,
            pretrained=True,
        )
        self.model.eval()
        self.model = self.model.cpu()

    def extract(self, roi_gray_uint8: np.ndarray) -> np.ndarray:
        import cv2
        if roi_gray_uint8 is None or roi_gray_uint8.size == 0:
            vec = np.zeros(self.feature_dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        img = roi_gray_uint8
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        x = img.astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        x = np.transpose(x, (2, 0, 1))
        t = self.torch.tensor(x).unsqueeze(0)
        with self.torch.no_grad():
            emb = self.model(t)
        emb = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        else:
            emb = np.zeros(self.feature_dim, dtype=np.float32)
            emb[0] = 1.0
        return emb

    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        return self.extract(preprocessed_roi)

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_model_name(self) -> str:
        return self.model_name


def get_palm_model() -> BasePalmModel:
    """
    Factory function to instantiate the configured palm biometric model.
    If config.DEEP_ENABLED is True, returns DeepPalmModel.
    Otherwise returns ClassicalPalmModel.
    """
    if getattr(config, "DEEP_ENABLED", False):
        return DeepPalmModel()
    return ClassicalPalmModel()

