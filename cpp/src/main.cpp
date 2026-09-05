#include "palmpay/types.hpp"
#include "palmpay/matcher.hpp"
#include "palmpay/liveness.hpp"
#include "palmpay/preprocess.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

// Minimal robust JSON serializer/deserializer helpers for CLI I/O
// avoiding external header dependencies if nlohmann/json is not bundled.

static std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else o << c;
    }
    return o.str();
}

static void print_status() {
    std::cout << "{\n"
              << "  \"version\": \"1.0.0-phase2\",\n"
              << "  \"onnx_available\": false,\n"
              << "  \"opencv_version\": \"headless-core\",\n"
              << "  \"build_mode\": \"Release\"\n"
              << "}" << std::endl;
}

// Simple manual parser for { "challenge": [...], "frames": [ { "landmarks": [[x,y,z], ...] } ] }
static bool parse_liveness_input(const std::string& input,
                                 std::vector<std::string>& challenge,
                                 std::vector<palmpay::Frame>& frames) {
    // Look for challenge items
    size_t ch_pos = input.find("\"challenge\"");
    if (ch_pos != std::string::npos) {
        size_t b_open = input.find('[', ch_pos);
        size_t b_close = input.find(']', b_open);
        if (b_open != std::string::npos && b_close != std::string::npos) {
            std::string sub = input.substr(b_open + 1, b_close - b_open - 1);
            size_t p = 0;
            while ((p = sub.find('"', p)) != std::string::npos) {
                size_t p2 = sub.find('"', p + 1);
                if (p2 == std::string::npos) break;
                challenge.push_back(sub.substr(p + 1, p2 - p - 1));
                p = p2 + 1;
            }
        }
    }

    // Look for frames / landmarks
    size_t fr_pos = input.find("\"frames\"");
    if (fr_pos == std::string::npos) return false;

    // Scan landmark coordinate sequences [[x,y,z], ...]
    size_t cur = fr_pos;
    while ((cur = input.find("\"landmarks\"", cur)) != std::string::npos) {
        size_t start = input.find('[', cur);
        if (start == std::string::npos) break;
        cur = start + 1;

        palmpay::Frame frame;
        // Parse up to 21 [x, y, z] triples
        while (frame.landmarks.size() < 21) {
            size_t p_open = input.find('[', cur);
            if (p_open == std::string::npos) break;
            size_t p_close = input.find(']', p_open);
            if (p_close == std::string::npos) break;

            std::string coords = input.substr(p_open + 1, p_close - p_open - 1);
            std::stringstream ss(coords);
            float x = 0, y = 0, z = 0;
            char sep;
            if (ss >> x >> sep >> y) {
                if (ss >> sep >> z) {}
                frame.landmarks.push_back({x, y, z});
            }
            cur = p_close + 1;
            // Check if we hit the closing bracket of landmarks array
            size_t next_bracket = input.find_first_of("[]", cur);
            if (next_bracket != std::string::npos && input[next_bracket] == ']') {
                cur = next_bracket + 1;
                break;
            }
        }
        if (!frame.landmarks.empty()) {
            frames.push_back(frame);
        }
    }

    return !challenge.empty() && !frames.empty();
}

static void handle_liveness() {
    std::string input((std::istreambuf_iterator<char>(std::cin)),
                      std::istreambuf_iterator<char>());

    std::vector<std::string> challenge;
    std::vector<palmpay::Frame> frames;

    if (!parse_liveness_input(input, challenge, frames)) {
        std::cout << "{\"ok\": false, \"passed\": false, \"score\": 0.0, \"reason\": \"invalid_json\"}" << std::endl;
        return;
    }

    palmpay::LivenessResult res = palmpay::verify_liveness(frames, challenge);
    std::cout << "{\n"
              << "  \"ok\": true,\n"
              << "  \"passed\": " << (res.passed ? "true" : "false") << ",\n"
              << "  \"score\": " << std::fixed << std::setprecision(4) << res.score << ",\n"
              << "  \"reason\": \"" << escape_json(res.reason) << "\"\n"
              << "}" << std::endl;
}

static void handle_match() {
    std::string input((std::istreambuf_iterator<char>(std::cin)),
                      std::istreambuf_iterator<char>());

    // Check if cosine similarity op
    if (input.find("\"op\": \"cosine\"") != std::string::npos || input.find("\"op\":\"cosine\"") != std::string::npos) {
        // Extract vec_a and vec_b
        auto extract_vec = [](const std::string& str, const std::string& key) {
            std::vector<float> v;
            size_t pos = str.find(key);
            if (pos == std::string::npos) return v;
            size_t open_b = str.find('[', pos);
            size_t close_b = str.find(']', open_b);
            if (open_b == std::string::npos || close_b == std::string::npos) return v;

            std::string sub = str.substr(open_b + 1, close_b - open_b - 1);
            std::stringstream ss(sub);
            float val;
            char comma;
            while (ss >> val) {
                v.push_back(val);
                ss >> comma;
            }
            return v;
        };

        std::vector<float> a = extract_vec(input, "\"vec_a\"");
        std::vector<float> b = extract_vec(input, "\"vec_b\"");
        double score = palmpay::cosine_similarity(a, b);
        std::cout << "{\"ok\": true, \"score\": " << std::fixed << std::setprecision(6) << score << "}" << std::endl;
        return;
    }

    std::cout << "{\"ok\": false, \"reason\": \"unsupported_match_op\"}" << std::endl;
}

static void handle_embed() {
    std::cout << "{\n"
              << "  \"ok\": true,\n"
              << "  \"model_name\": \"deep-palm-embedder-v1\",\n"
              << "  \"feature_dim\": 512,\n"
              << "  \"message\": \"native_embedder_scaffold\"\n"
              << "}" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: palmpay_native <status|liveness|embed|match>" << std::endl;
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "status") {
        print_status();
    } else if (cmd == "liveness") {
        handle_liveness();
    } else if (cmd == "match") {
        handle_match();
    } else if (cmd == "embed") {
        handle_embed();
    } else {
        std::cerr << "Unknown subcommand: " << cmd << std::endl;
        return 1;
    }

    return 0;
}
