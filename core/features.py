"""
Feature Extraction Module using Multi-Block Multi-Scale Uniform Local Binary
Patterns (MB-uLBP, radii R=1 and R=2). Computes 59-bin uniform LBP histograms
across a 4x4 spatial block grid per scale and concatenates them into the final
L2-normalized palm feature vector.

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
    Multi-Block Multi-Scale Uniform Local Binary Patterns feature extractor.
    Concatenates uniform-LBP block histograms across multiple radii
    (config.LBP_RADII, e.g. R=1 fine texture + R=2 contextual texture) to
    generate the final L2-normalized palm feature vector.
    """

    def __init__(
        self,
        grid_x: int = config.LBP_GRID_X,
        grid_y: int = config.LBP_GRID_Y,
        bins_per_block: int = config.LBP_BINS_PER_BLOCK,
        radii=config.LBP_RADII,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.bins_per_block = bins_per_block
        self.radii = tuple(radii)
        self.expected_dim = grid_x * grid_y * bins_per_block * len(self.radii)
        self.lut = UniformLBPTable.get_table()

    def compute_lbp_image(self, gray_image: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        Computes 8-neighbor (P=8, R=radius) uniform-LBP image using vectorized
        numpy slicing. Maps values to 59 uniform pattern indices via lookup table.

        Args:
            gray_image: 2D grayscale uint8 array (e.g. 128x128).
            radius: neighbor distance in pixels.

        Returns:
            2D int32 array of uniform LBP bin indices in [0, 58].
        """
        h, w = gray_image.shape
        r = radius
        center = gray_image[r : h - r, r : w - r]

        # 8 neighbors at distance r:
        # Top-Left, Top, Top-Right, Right, Bottom-Right, Bottom, Bottom-Left, Left
        n0 = gray_image[0 : h - 2 * r, 0 : w - 2 * r] >= center      # Bit 0 (TL)
        n1 = gray_image[0 : h - 2 * r, r : w - r] >= center          # Bit 1 (T)
        n2 = gray_image[0 : h - 2 * r, 2 * r : w] >= center          # Bit 2 (TR)
        n3 = gray_image[r : h - r, 2 * r : w] >= center              # Bit 3 (R)
        n4 = gray_image[2 * r : h, 2 * r : w] >= center              # Bit 4 (BR)
        n5 = gray_image[2 * r : h, r : w - r] >= center              # Bit 5 (B)
        n6 = gray_image[2 * r : h, 0 : w - 2 * r] >= center          # Bit 6 (BL)
        n7 = gray_image[r : h - r, 0 : w - 2 * r] >= center          # Bit 7 (L)

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
        return self.lut[lbp_code]

    def extract_features(self, enhanced_roi: np.ndarray) -> np.ndarray:
        """
        Extracts concatenated multi-scale spatial block histograms.

        Layout (fixed order): for every 4x4-grid block, append one histogram
        per radius (R1 then R2). Total dim = blocks(16) x scales(2) x bins(59).

        Args:
            enhanced_roi: 2D contrast-enhanced grayscale palm image (uint8).

        Returns:
            1D float32 array, L2-normalized to unit length.
        """
        if enhanced_roi is None or enhanced_roi.size == 0:
            return np.zeros(self.expected_dim, dtype=np.float32)

        # Ensure single-channel 2D uint8
        if len(enhanced_roi.shape) == 3:
            enhanced_roi = enhanced_roi[:, :, 0]
        enhanced_roi = enhanced_roi.astype(np.uint8)

        # One coded image per radius (computed once)
        lbp_images = [self.compute_lbp_image(enhanced_roi, radius=r) for r in self.radii]
        h, w = lbp_images[0].shape

        block_h = h // self.grid_y
        block_w = w // self.grid_x

        feature_blocks = []

        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                y_start = gy * block_h
                y_end = (gy + 1) * block_h if gy < self.grid_y - 1 else h
                x_start = gx * block_w
                x_end = (gx + 1) * block_w if gx < self.grid_x - 1 else w

                for lbp_img in lbp_images:
                    sub_block = lbp_img[y_start:y_end, x_start:x_end]

                    hist, _ = np.histogram(
                        sub_block,
                        bins=self.bins_per_block,
                        range=(0, self.bins_per_block),
                    )
                    hist = hist.astype(np.float32)

                    norm_val = np.linalg.norm(hist)
                    if norm_val > 1e-7:
                        hist /= norm_val

                    feature_blocks.append(hist)

        feature_vector = np.concatenate(feature_blocks).astype(np.float32)

        global_norm = np.linalg.norm(feature_vector)
        if global_norm > 1e-7:
            feature_vector /= global_norm

        return feature_vector


import cv2

class CompCodeFeatureExtractor:
    """
    Competitive Coding (CompCode) feature extractor.
    Convolves the ROI with a bank of 6 Gabor filters (0 to 150 degrees).
    The winning orientation (0-5) at each pixel forms the competitive code map.
    To provide translation invariance and interface with the existing Cosine Similarity 
    matcher, the code map is aggregated into block histograms (e.g. 16x16 grid), 
    yielding a fixed-length embedding.
    """
    def __init__(
        self,
        grid_x: int = config.COMPCODE_GRID_X,
        grid_y: int = config.COMPCODE_GRID_Y,
        num_orientations: int = config.COMPCODE_ORIENTATIONS,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.num_orientations = num_orientations
        self.expected_dim = grid_x * grid_y * num_orientations
        
        # Create Gabor filter bank
        self.filters = []
        ksize = 17
        sigma = 3.0
        lambd = 10.0
        gamma = 0.5
        psi = 0.0
        for i in range(num_orientations):
            theta = i * (np.pi / num_orientations)
            # We want to detect dark lines on light background. 
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            # Zero mean the kernel to ignore illumination baseline
            kernel -= np.mean(kernel)
            self.filters.append(kernel)

    def extract_features(self, enhanced_roi: np.ndarray) -> np.ndarray:
        if enhanced_roi is None or enhanced_roi.size == 0:
            return np.zeros(self.expected_dim, dtype=np.float32)
        
        # Ensure single-channel 2D
        if len(enhanced_roi.shape) == 3:
            enhanced_roi = enhanced_roi[:, :, 0]
            
        img = enhanced_roi.astype(np.float32)
        
        # Convolve with all filters
        responses = []
        for kernel in self.filters:
            # We look for dark lines, so if kernel has positive center, dark lines give negative response.
            # Thus we can look for the minimum response, or we invert the image.
            # Let us invert the image so lines are bright:
            resp = cv2.filter2D(255.0 - img, cv2.CV_32F, kernel)
            responses.append(resp)
            
        responses = np.stack(responses, axis=-1) # shape: (H, W, 6)
        
        # Competitive coding: find orientation with max response
        comp_code_map = np.argmax(responses, axis=-1) # shape: (H, W)
        
        # Compute block histograms
        h, w = comp_code_map.shape
        block_h = h // self.grid_y
        block_w = w // self.grid_x
        
        feature_blocks = []
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                y_start = gy * block_h
                y_end = (gy + 1) * block_h if gy < self.grid_y - 1 else h
                x_start = gx * block_w
                x_end = (gx + 1) * block_w if gx < self.grid_x - 1 else w
                
                sub_block = comp_code_map[y_start:y_end, x_start:x_end]
                hist, _ = np.histogram(sub_block, bins=self.num_orientations, range=(0, self.num_orientations))
                hist = hist.astype(np.float32)
                
                norm_val = np.linalg.norm(hist)
                if norm_val > 1e-7:
                    hist /= norm_val
                    
                feature_blocks.append(hist)
                
        feature_vector = np.concatenate(feature_blocks).astype(np.float32)
        
        global_norm = np.linalg.norm(feature_vector)
        if global_norm > 1e-7:
            feature_vector /= global_norm
            
        return feature_vector



class HogLbpFusionFeatureExtractor:
    """
    Fuses Histogram of Oriented Gradients (HOG) with MB-uLBP.
    HOG captures macroscopic edge structures (crease lines) while LBP captures
    microscopic skin texture. Fusing them often outperforms either alone.
    """
    def __init__(self):
        import cv2
        # LBP Component
        self.lbp = MBuLBPFeatureExtractor()
        
        # HOG Component
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # Total expected dimension
        self.hog_dim = 1764
        self.expected_dim = self.lbp.expected_dim + self.hog_dim

    def extract_features(self, enhanced_roi):
        import numpy as np
        if enhanced_roi is None or enhanced_roi.size == 0:
            return np.zeros(self.expected_dim, dtype=np.float32)
            
        if len(enhanced_roi.shape) == 3:
            enhanced_roi = enhanced_roi[:, :, 0]
            
        img = enhanced_roi.astype(np.uint8)
        
        # 1. Extract LBP
        lbp_feat = self.lbp.extract_features(img)
        
        # 2. Extract HOG
        hog_feat = self.hog.compute(img).flatten()
        # L2 Normalize HOG independently before concatenation
        hog_norm = np.linalg.norm(hog_feat)
        if hog_norm > 1e-7:
            hog_feat /= hog_norm
            
        # 3. Concatenate and normalize jointly
        fused = np.concatenate([lbp_feat, hog_feat]).astype(np.float32)
        fused_norm = np.linalg.norm(fused)
        if fused_norm > 1e-7:
            fused /= fused_norm
            
        return fused

