#include "palmpay/liveness.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// Helper to construct synthetic palm landmarks
static std::vector<palmpay::Landmark> make_synthetic_hand(bool fist) {
    std::vector<palmpay::Landmark> lms(21, {0.0f, 0.0f, 0.0f});
    // Wrist
    lms[0] = {100.0f, 200.0f, 0.0f};
    // Middle MCP
    lms[9] = {100.0f, 100.0f, 0.0f}; // ref = 100.0

    // Other MCPs
    lms[2] = {70.0f, 120.0f, 0.0f};
    lms[5] = {85.0f, 105.0f, 0.0f};
    lms[13] = {115.0f, 105.0f, 0.0f};
    lms[17] = {130.0f, 115.0f, 0.0f};

    if (fist) {
        // Tips curled close to wrist / near MCPs
        lms[4] = {75.0f, 130.0f, 0.0f};
        lms[8] = {85.0f, 110.0f, 0.0f};
        lms[12] = {100.0f, 110.0f, 0.0f};
        lms[16] = {115.0f, 110.0f, 0.0f};
        lms[20] = {125.0f, 120.0f, 0.0f};
    } else {
        // Extended fingers
        lms[4] = {50.0f, 90.0f, 0.0f};
        lms[8] = {80.0f, 20.0f, 0.0f};
        lms[12] = {100.0f, 10.0f, 0.0f};
        lms[16] = {120.0f, 20.0f, 0.0f};
        lms[20] = {140.0f, 40.0f, 0.0f};
    }
    return lms;
}

void test_liveness_gesture_classification() {
    auto hand_open = make_synthetic_hand(false);
    auto hand_fist = make_synthetic_hand(true);

    double ext = 0, spr = 0;
    std::string st_open = palmpay::classify_landmark_state(hand_open, &ext, &spr);
    assert(st_open == "OPEN_PALM" || st_open == "SPREAD");

    std::string st_fist = palmpay::classify_landmark_state(hand_fist, &ext, &spr);
    assert(st_fist == "FIST");

    std::cout << "[PASS] test_liveness_gesture_classification" << std::endl;
}

void test_liveness_sequence_verification() {
    // 5 open frames then 5 fist frames
    std::vector<palmpay::Frame> valid_frames;
    for (int i = 0; i < 5; ++i) {
        valid_frames.push_back({i * 100, make_synthetic_hand(false)});
    }
    for (int i = 5; i < 10; ++i) {
        valid_frames.push_back({i * 100, make_synthetic_hand(true)});
    }

    std::vector<std::string> challenge = {"OPEN_PALM", "FIST"};
    palmpay::LivenessResult res = palmpay::verify_liveness(valid_frames, challenge, 8, 18);
    assert(res.passed);
    assert(res.score > 0.5);
    assert(res.reason == "ok");

    // Static frames rejection: all 10 frames identical
    std::vector<palmpay::Frame> static_frames(10, {0, make_synthetic_hand(false)});
    palmpay::LivenessResult static_res = palmpay::verify_liveness(static_frames, challenge, 8, 18);
    assert(!static_res.passed);
    assert(static_res.reason == "static_image_detected");

    // Wrong order rejection
    std::vector<std::string> reverse_challenge = {"FIST", "OPEN_PALM"};
    palmpay::LivenessResult rev_res = palmpay::verify_liveness(valid_frames, reverse_challenge, 8, 18);
    assert(!rev_res.passed);
    assert(rev_res.reason == "challenge_order_mismatch");

    std::cout << "[PASS] test_liveness_sequence_verification" << std::endl;
}

int main() {
    test_liveness_gesture_classification();
    test_liveness_sequence_verification();
    std::cout << "All C++ liveness tests passed successfully." << std::endl;
    return 0;
}
