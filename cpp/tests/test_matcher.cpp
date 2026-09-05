#include "palmpay/matcher.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

void test_cosine_similarity() {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f};
    assert(std::abs(palmpay::cosine_similarity(a, b) - 1.0) < 1e-5);

    std::vector<float> c = {0.0f, 1.0f, 0.0f};
    assert(std::abs(palmpay::cosine_similarity(a, c) - 0.0) < 1e-5);

    std::vector<float> empty_vec = {};
    assert(palmpay::cosine_similarity(empty_vec, a) == 0.0);
    std::cout << "[PASS] test_cosine_similarity" << std::endl;
}

void test_match_qualification_and_ambiguity() {
    std::vector<float> query = {1.0f, 0.0f, 0.0f};
    std::vector<std::vector<float>> candidates = {
        {0.99f, 0.01f, 0.0f},
        {0.985f, 0.02f, 0.0f},
        {0.10f, 0.90f, 0.0f}
    };

    // Low threshold, top two candidates are very close (< 0.02 gap) => ambiguous
    palmpay::MatchResult res = palmpay::match_embeddings(query, candidates, 0.95, 0.02);
    assert(!res.qualified);
    assert(res.reason == "ambiguous");

    // Decisive winner
    std::vector<std::vector<float>> candidates_clear = {
        {0.99f, 0.01f, 0.0f},
        {0.70f, 0.70f, 0.0f}
    };
    palmpay::MatchResult res_clear = palmpay::match_embeddings(query, candidates_clear, 0.95, 0.02);
    assert(res_clear.qualified);
    assert(res_clear.best_index == 0);
    assert(res_clear.reason == "ok");

    std::cout << "[PASS] test_match_qualification_and_ambiguity" << std::endl;
}

void test_geometry_distance_gating() {
    std::vector<float> query_geo = {1.0f, 2.0f};
    std::vector<float> ref_mean = {1.0f, 2.0f};
    std::vector<float> ref_std = {0.1f, 0.1f};

    double dist = palmpay::geometry_distance(query_geo, ref_mean, ref_std);
    assert(dist < 1e-5);

    std::vector<float> far_geo = {5.0f, 10.0f};
    double far_dist = palmpay::geometry_distance(far_geo, ref_mean, ref_std);
    assert(far_dist > 5.0);

    std::cout << "[PASS] test_geometry_distance_gating" << std::endl;
}

int main() {
    test_cosine_similarity();
    test_match_qualification_and_ambiguity();
    test_geometry_distance_gating();
    std::cout << "All C++ matcher tests passed successfully." << std::endl;
    return 0;
}
