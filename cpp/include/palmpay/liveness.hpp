#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace palmpay {

// Landmark gesture classifier
std::string classify_landmark_state(const std::vector<Landmark>& landmarks,
                                   double* out_extension = nullptr,
                                   double* out_spread = nullptr);

// Sequence verifier matching the Python challenge-response logic
LivenessResult verify_liveness(const std::vector<Frame>& frames,
                               const std::vector<std::string>& challenge,
                               int min_frames = 8,
                               int max_frames = 18);

} // namespace palmpay
