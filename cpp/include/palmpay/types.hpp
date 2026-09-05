#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace palmpay {

struct Landmark {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
};

struct Frame {
    int64_t ts_ms{0};
    std::vector<Landmark> landmarks;
};

struct LivenessResult {
    bool passed{false};
    double score{0.0};
    std::string reason{"unknown"};
    std::vector<std::string> detected_sequence;
};

struct StatusResult {
    std::string version{"1.0.0-phase2"};
    bool onnx_available{false};
    std::string opencv_version{"none"};
    std::string build_mode{"Release"};
};

struct MatchResult {
    bool qualified{false};
    double best_score{0.0};
    int best_index{-1};
    std::string reason{"ok"};
    std::vector<double> scores;
};

} // namespace palmpay
