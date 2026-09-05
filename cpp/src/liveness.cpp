#include "palmpay/liveness.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace palmpay {

static double dist2d(const Landmark& a, const Landmark& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::string classify_landmark_state(const std::vector<Landmark>& landmarks,
                                   double* out_extension,
                                   double* out_spread) {
    if (landmarks.size() < 21) {
        if (out_extension) *out_extension = 0.0;
        if (out_spread) *out_spread = 0.0;
        return "UNKNOWN";
    }

    const Landmark& wrist = landmarks[0];
    const Landmark& middle_mcp = landmarks[9];
    double ref = dist2d(wrist, middle_mcp);

    if (ref < 1e-6) {
        if (out_extension) *out_extension = 0.0;
        if (out_spread) *out_spread = 0.0;
        return "UNKNOWN";
    }

    const int tips[5] = {4, 8, 12, 16, 20};
    const int mcps[5] = {2, 5, 9, 13, 17};

    double tip_dists[5];
    double mcp_dists[5];
    for (int i = 0; i < 5; ++i) {
        tip_dists[i] = dist2d(landmarks[tips[i]], wrist) / ref;
        mcp_dists[i] = dist2d(landmarks[mcps[i]], wrist) / ref;
    }

    int extended_count = 0;
    double sum_tip_dist = 0.0;
    for (int i = 1; i < 5; ++i) {
        sum_tip_dist += tip_dists[i];
        if (tip_dists[i] > (mcp_dists[i] + 0.35)) {
            extended_count++;
        }
    }
    double avg_tip_dist = sum_tip_dist / 4.0;

    double sum_spread = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum_spread += dist2d(landmarks[tips[i]], landmarks[tips[i + 1]]) / ref;
    }
    double avg_spread = sum_spread / 4.0;

    if (out_extension) *out_extension = avg_tip_dist;
    if (out_spread) *out_spread = avg_spread;

    if (extended_count == 0 || avg_tip_dist < 1.15) {
        return "FIST";
    } else if (extended_count >= 3) {
        if (avg_spread >= 0.52) {
            return "SPREAD";
        } else {
            return "OPEN_PALM";
        }
    } else {
        return (avg_tip_dist >= 1.3) ? "OPEN_PALM" : "FIST";
    }
}

LivenessResult verify_liveness(const std::vector<Frame>& frames,
                               const std::vector<std::string>& challenge,
                               int min_frames,
                               int max_frames) {
    LivenessResult res;

    if (static_cast<int>(frames.size()) < min_frames) {
        res.passed = false;
        res.score = 0.0;
        res.reason = "insufficient_frames";
        return res;
    }

    size_t limit = std::min(frames.size(), static_cast<size_t>(max_frames));
    std::vector<std::string> states;
    std::vector<double> extensions;
    std::vector<double> spreads;

    for (size_t i = 0; i < limit; ++i) {
        if (frames[i].landmarks.size() >= 21) {
            double ext = 0.0, spr = 0.0;
            std::string st = classify_landmark_state(frames[i].landmarks, &ext, &spr);
            states.push_back(st);
            extensions.push_back(ext);
            spreads.push_back(spr);
        }
    }

    if (static_cast<int>(states.size()) < min_frames) {
        res.passed = false;
        res.score = 0.0;
        res.reason = "insufficient_valid_landmarks";
        return res;
    }

    // Check motion variance (anti-static check)
    double mean_ext = std::accumulate(extensions.begin(), extensions.end(), 0.0) / extensions.size();
    double var_ext = 0.0;
    double min_ext = extensions[0], max_ext = extensions[0];

    for (double e : extensions) {
        var_ext += (e - mean_ext) * (e - mean_ext);
        if (e < min_ext) min_ext = e;
        if (e > max_ext) max_ext = e;
    }
    var_ext /= extensions.size();
    double ext_diff = max_ext - min_ext;

    if (var_ext < 0.005 && ext_diff < 0.15) {
        res.passed = false;
        res.score = 0.0;
        res.reason = "static_image_detected";
        return res;
    }

    // Collapse consecutive duplicates
    std::vector<std::string> collapsed;
    std::string curr = "";
    int run_len = 0;

    for (const auto& s : states) {
        if (s != curr) {
            if (!curr.empty() && run_len >= 2) {
                collapsed.push_back(curr);
            }
            curr = s;
            run_len = 1;
        } else {
            run_len++;
        }
    }
    if (!curr.empty() && run_len >= 2) {
        collapsed.push_back(curr);
    }

    res.detected_sequence = collapsed;

    // Match challenge in order
    size_t target_idx = 0;
    for (const auto& st : collapsed) {
        if (target_idx < challenge.size() && st == challenge[target_idx]) {
            target_idx++;
        }
    }

    if (target_idx < challenge.size()) {
        res.passed = false;
        res.score = 0.2 * (static_cast<double>(target_idx) / static_cast<double>(challenge.size()));
        res.reason = "challenge_order_mismatch";
        return res;
    }

    double articulation = std::min(1.0, ext_diff / 0.6);
    res.score = 0.4 * articulation + 0.6;
    res.passed = true;
    res.reason = "ok";
    return res;
}

} // namespace palmpay
