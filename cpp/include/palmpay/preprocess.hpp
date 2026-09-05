#pragma once

#include "types.hpp"
#include <vector>
#include <cstdint>
#include <cmath>

namespace palmpay {

// Pure C++ headless preprocessing routines without strict external dependencies
struct ROIResult {
    bool success{false};
    int width{128};
    int height{128};
    std::vector<uint8_t> gray_pixels; // row-major 128x128
    std::string message{"ok"};
};

// Crops/normalizes given landmark points and dummy/test image bytes
ROIResult preprocess_palm_roi(const std::vector<uint8_t>& image_bytes,
                             int src_w,
                             int src_h,
                             const std::vector<Landmark>& landmarks,
                             int target_w = 128,
                             int target_h = 128);

// Applies CLAHE-like contrast adjustment to a grayscale buffer
void apply_simple_clahe(std::vector<uint8_t>& gray_image, int w, int h, double clip_limit = 2.0);

} // namespace palmpay
