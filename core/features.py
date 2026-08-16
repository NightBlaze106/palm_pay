"""
Feature Extraction Module using Multi-Block Uniform Local Binary Patterns (MB-uLBP).
Computes 59-bin uniform LBP histograms across a 4x4 spatial block grid to generate
a 944-dimensional normalized palm feature vector.

Scientific References:
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation
  invariant texture classification with local binary patterns. IEEE TPAMI.
- Amrouni, M. et al. (2022). Contactless palmprint preprocessing and recognition.
"""

from typing import Tuple
import numpy as np

import config


class UniformLBPTable:
    """
    Lookup table mapping 8-bit LBP values (0-255) to Uniform LBP bins (0-58).
    Uniform patterns have at most 2 bit transitions (0->1 or 1->0).
    There are 58 uniform patterns and 1 non-uniform pattern (bin 58), totaling 59 bins.
    """

    _instance = None
    _table = None

    @classmethod
    def get_table(cls) -> np.ndarray:
        if cls._table is None:
            table = np.zeros(256, dtype=np.int32)
            uniform_bin_index = 0
            non_uniform_bin = 58  # 59th bin (index 58) for all non-uniform patterns

            for val in range(256):
                # Count bit transitions in circular 8-bit representation
                binary_str = f"{val:08b}"
                transitions = 0
                for i in range(8):
                    if binary_str[i] != binary_str[(i + 1) % 8]:
                        transitions += 1

                if transitions <= 2:
                    table[val] = uniform_bin_index
                    uniform_bin_index += 1
                else:
                    table[val] = non_uniform_bin

            cls._table = table
        return cls._table


class MBuLBPFeatureExtractor:
    """
    Multi-Block Uniform Local Binary Patterns (MB-uLBP) feature extractor.
    Generates a 944-dimensional L2-normalized feature vector from an enhanced palm ROI.
    """

    def __init__(
        self,
        grid_x: int = config.LBP_GRID_X,
        grid_y: int = config.LBP_GRID_Y,
        bins_per_block: int = config.LBP_BINS_PER_BLOCK,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.bins_per_block = bins_per_block
        self.expected_dim = grid_x * grid_y * bins_per_block
        self.lut = UniformLBPTable.get_table()

    def compute_lbp_image(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Computes 8-neighbor (P=8, R=1) LBP image using vectorized numpy slicing.
        Maps values to 59 uniform pattern indices via lookup table.

        Args:
            gray_image: 2D grayscale uint8 array (e.g. 128x128).

        Returns:
            2D int32 array containing uniform LBP bin indices in [0, 58].
        """
        h, w = gray_image.shape
        center = gray_image[1 : h - 1, 1 : w - 1]

        # 8 neighbors around center (radius = 1)
        # Top-Left, Top, Top-Right, Right, Bottom-Right, Bottom, Bottom-Left, Left
        n0 = gray_image[0 : h - 2, 0 : w - 2] >= center  # Bit 0
        n1 = gray_image[0 : h - 2, 1 : w - 1] >= center  # Bit 1
        n2 = gray_image[0 : h - 2, 2 : w] >= center      # Bit 2
        n3 = gray_image[1 : h - 1, 2 : w] >= center      # Bit 3
        n4 = gray_image[2 : h, 2 : w] >= center          # Bit 4
        n5 = gray_image[2 : h, 1 : w - 1] >= center      # Bit 5
        n6 = gray_image[2 : h, 0 : w - 2] >= center      # Bit 6
        n7 = gray_image[1 : h - 1, 0 : w - 2] >= center  # Bit 7

        # Pack bits into 8-bit integer [0, 255]
        lbp_code = (
            (n0.astype(np.uint8) << 7)
            | (n1.astype(np.uint8) << 6)
            | (n2.astype(np.uint8) << 5)
            | (n3.astype(np.uint8) << 4)
            | (n4.astype(np.uint8) << 3)
            | (n5.astype(np.uint8) << 2)
            | (n6.astype(np.uint8) << 1)
            | (n7.astype(np.uint8) << 0)
        )

        # Map to uniform bins using lookup table
        uniform_lbp = self.lut[lbp_code]
        return uniform_lbp

    def extract_features(self, enhanced_roi: np.ndarray) -> np.ndarray:
        """
        Extracts concatenated spatial block histograms from enhanced palm ROI.

        Args:
            enhanced_roi: 2D CLAHE-enhanced grayscale palm image (128x128 uint8).

        Returns:
            1D numpy array of shape (944,), float32, L2-normalized to unit length.
        """
        if enhanced_roi is None or enhanced_roi.size == 0:
            return np.zeros(self.expected_dim, dtype=np.float32)

        # Ensure single-channel 2D uint8
        if len(enhanced_roi.shape) == 3:
            enhanced_roi = enhanced_roi[:, :, 0]
        enhanced_roi = enhanced_roi.astype(np.uint8)

        # Compute Uniform LBP coded image
        lbp_image = self.compute_lbp_image(enhanced_roi)
        h, w = lbp_image.shape

        block_h = h // self.grid_y
        block_w = w // self.grid_x

        feature_blocks = []

        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                y_start = gy * block_h
                y_end = (gy + 1) * block_h if gy < self.grid_y - 1 else h
                x_start = gx * block_w
                x_end = (gx + 1) * block_w if gx < self.grid_x - 1 else w

                sub_block = lbp_image[y_start:y_end, x_start:x_end]

                # Compute 59-bin histogram
                hist, _ = np.histogram(
                    sub_block,
                    bins=self.bins_per_block,
                    range=(0, self.bins_per_block),
                )
                hist = hist.astype(np.float32)

                # Block-level L2 normalization
                norm_val = np.linalg.norm(hist)
                if norm_val > 1e-7:
                    hist /= norm_val

                feature_blocks.append(hist)

        # Concatenate all spatial block histograms
        feature_vector = np.concatenate(feature_blocks).astype(np.float32)

        # Overall global L2 normalization to unit vector
        global_norm = np.linalg.norm(feature_vector)
        if global_norm > 1e-7:
            feature_vector /= global_norm

        return feature_vector
