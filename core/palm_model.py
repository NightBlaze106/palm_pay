"""
Palm Model Abstraction and Feature Extractor Interface.
Provides a modular interface for biometric embedding extraction, allowing
future deep learning or ONNX models to be plugged in seamlessly.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

import config
from core.features import MBuLBPFeatureExtractor


class BasePalmModel(ABC):
    """
    Abstract Base Class for Palm Biometric Feature Extractors.
    """

    @abstractmethod
    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        """
        Extracts a normalized 1D feature/embedding vector from a preprocessed palm ROI.
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Returns feature vector dimensionality.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Returns model identifier/version.
        """
        pass


class ClassicalPalmModel(BasePalmModel):
    """
    Standard Phase-1 Feature Extractor:
    Multi-Block Uniform Local Binary Patterns (MB-uLBP_{8,1}^u2) with CLAHE.
    """

    def __init__(self):
        self.extractor = MBuLBPFeatureExtractor(
            grid_x=config.LBP_GRID_X,
            grid_y=config.LBP_GRID_Y,
            bins_per_block=config.LBP_BINS_PER_BLOCK,
        )
        self.model_name = config.MODEL_NAME

    def extract_embedding(self, preprocessed_roi: np.ndarray) -> np.ndarray:
        return self.extractor.extract_features(preprocessed_roi)

    def get_feature_dim(self) -> int:
        return self.extractor.expected_dim

    def get_model_name(self) -> str:
        return self.model_name


def get_palm_model() -> BasePalmModel:
    """
    Factory function to instantiate the configured palm biometric model.
    """
    return ClassicalPalmModel()
