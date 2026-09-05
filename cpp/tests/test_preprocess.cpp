#include "palmpay/preprocess.hpp"
#include <cassert>
#include <iostream>
#include <vector>

void test_preprocess_roi() {
    std::vector<palmpay::Landmark> lms(21, {10.0f, 20.0f, 0.0f});
    std::vector<uint8_t> dummy_bytes(640 * 480, 100);

    palmpay::ROIResult res = palmpay::preprocess_palm_roi(dummy_bytes, 640, 480, lms, 128, 128);
    assert(res.success);
    assert(res.width == 128);
    assert(res.height == 128);
    assert(res.gray_pixels.size() == 128 * 128);

    // Invalid landmarks
    std::vector<palmpay::Landmark> empty_lms;
    palmpay::ROIResult fail_res = palmpay::preprocess_palm_roi(dummy_bytes, 640, 480, empty_lms);
    assert(!fail_res.success);

    std::cout << "[PASS] test_preprocess_roi" << std::endl;
}

void test_simple_clahe() {
    std::vector<uint8_t> buf = {10, 20, 30, 40};
    palmpay::apply_simple_clahe(buf, 2, 2, 2.0);
    assert(buf[0] == 0);
    assert(buf[3] == 255);
    std::cout << "[PASS] test_simple_clahe" << std::endl;
}

int main() {
    test_preprocess_roi();
    test_simple_clahe();
    std::cout << "All C++ preprocess tests passed successfully." << std::endl;
    return 0;
}
