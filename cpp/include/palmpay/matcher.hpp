#pragma once

#include "types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace palmpay {

// Cosine similarity between two float vectors
double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

// RMS geometry distance between query and reference mean/std
double geometry_distance(const std::vector<float>& query,
                         const std::vector<float>& ref_mean,
                         const std::vector<float>& ref_std);

// 1:N Matching decision logic
MatchResult match_embeddings(const std::vector<float>& query,
                             const std::vector<std::vector<float>>& candidates,
                             double threshold,
                             double ambiguity_margin = 0.02,
                             const std::vector<double>& geo_distances = {},
                             double geo_threshold = 1.5);

} // namespace palmpay
