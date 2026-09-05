#include "palmpay/preprocess.hpp"
#include <algorithm>

namespace palmpay {

ROIResult preprocess_palm_roi(const std::vector<uint8_t>& image_bytes,
                             int src_w,
                             int src_h,
                             const std::vector<Landmark>& landmarks,
                             int target_w,
                             int target_h) {
    ROIResult res;
    res.width = target_w;
    res.height = target_h;

    if (landmarks.size() < 21 || src_w <= 0 || src_h <= 0) {
        res.success = false;
        res.message = "invalid_landmarks_or_dimensions";
        return res;
    }

    res.gray_pixels.resize(target_w * target_h, 128);

    // If source buffer provided, do a center crop / sample
    if (image_bytes.size() >= static_cast<size_t>(src_w * src_h)) {
        for (int r = 0; r < target_h; ++r) {
            int sy = (r * src_h) / target_h;
            for (int c = 0; c < target_w; ++c) {
                int sx = (c * src_w) / target_w;
                res.gray_pixels[r * target_w + c] = image_bytes[sy * src_w + sx];
            }
        }
    }

    res.success = true;
    res.message = "ok";
    return res;
}

void apply_simple_clahe(std::vector<uint8_t>& gray_image, int w, int h, double clip_limit) {
    if (gray_image.empty() || w * h != static_cast<int>(gray_image.size())) {
        return;
    }

    uint8_t min_v = 255, max_v = 0;
    for (uint8_t v : gray_image) {
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    if (max_v > min_v) {
        for (size_t i = 0; i < gray_image.size(); ++i) {
            double norm = static_cast<double>(gray_image[i] - min_v) / (max_v - min_v);
            gray_image[i] = static_cast<uint8_t>(std::clamp(norm * 255.0, 0.0, 255.0));
        }
    }
}

} // namespace palmpay
