#include <cassert>
#include <iostream>
#include <cstdlib>
#include <string>
#include <array>
#include <memory>

static std::string exec_cmd(const std::string& cmd) {
    std::array<char, 256> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main(int argc, char** argv) {
    std::string bin = "./palmpay_native";
    if (argc > 1) {
        bin = argv[1];
    }

    // 1. Status command test
    std::string status_out = exec_cmd(bin + " status");
    assert(status_out.find("\"version\"") != std::string::npos);
    assert(status_out.find("\"onnx_available\"") != std::string::npos);

    // 2. Liveness invalid JSON
    std::string bad_live = exec_cmd("echo '{\"invalid\": 1}' | " + bin + " liveness");
    assert(bad_live.find("\"ok\": false") != std::string::npos || bad_live.find("\"ok\":false") != std::string::npos);

    std::cout << "All C++ CLI tests passed successfully." << std::endl;
    return 0;
}
