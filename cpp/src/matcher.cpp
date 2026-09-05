#include "palmpay/matcher.hpp"
#include <numeric>

namespace palmpay {

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }

    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a < 1e-7 || norm_b < 1e-7) {
        return 0.0;
    }

    double sim = dot / (norm_a * norm_b);
    if (sim < -1.0) sim = -1.0;
    if (sim > 1.0) sim = 1.0;
    return sim;
}

double geometry_distance(const std::vector<float>& query,
                         const std::vector<float>& ref_mean,
                         const std::vector<float>& ref_std) {
    if (query.empty() || query.size() != ref_mean.size() || query.size() != ref_std.size()) {
        return 999.0;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < query.size(); ++i) {
        double floor_val = 0.10 * std::abs(ref_mean[i]) + 1e-3;
        double sigma = std::max(static_cast<double>(ref_std[i]), floor_val);
        double diff = (static_cast<double>(query[i]) - static_cast<double>(ref_mean[i])) / sigma;
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq / static_cast<double>(query.size()));
}

MatchResult match_embeddings(const std::vector<float>& query,
                             const std::vector<std::vector<float>>& candidates,
                             double threshold,
                             double ambiguity_margin,
                             const std::vector<double>& geo_distances,
                             double geo_threshold) {
    MatchResult res;
    if (candidates.empty() || query.empty()) {
        res.qualified = false;
        res.reason = "no_candidates";
        return res;
    }

    res.scores.resize(candidates.size(), 0.0);
    int top_idx = -1;
    double top_score = -2.0;

    int second_idx = -1;
    double second_score = -2.0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        double s = cosine_similarity(query, candidates[i]);
        res.scores[i] = s;

        if (s > top_score) {
            second_score = top_score;
            second_idx = top_idx;
            top_score = s;
            top_idx = static_cast<int>(i);
        } else if (s > second_score) {
            second_score = s;
            second_idx = static_cast<int>(i);
        }
    }

    res.best_index = top_idx;
    res.best_score = top_score;

    if (top_score < threshold) {
        res.qualified = false;
        res.reason = "below_threshold";
        return res;
    }

    // Geometry check if provided
    if (!geo_distances.empty() && top_idx < static_cast<int>(geo_distances.size())) {
        if (geo_distances[top_idx] > geo_threshold) {
            res.qualified = false;
            res.reason = "geometry_mismatch";
            return res;
        }
    }

    // Ambiguity guard
    if (second_idx >= 0 && second_score >= threshold) {
        bool second_geo_pass = true;
        if (!geo_distances.empty() && second_idx < static_cast<int>(geo_distances.size())) {
            second_geo_pass = (geo_distances[second_idx] <= geo_threshold);
        }

        if (second_geo_pass && (top_score - second_score < ambiguity_margin)) {
            res.qualified = false;
            res.reason = "ambiguous";
            return res;
        }
    }

    res.qualified = true;
    res.reason = "ok";
    return res;
}

} // namespace palmpay
