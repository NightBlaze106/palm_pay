# PalmPay Native C++ Core (`palmpay_native`)

Optional high-performance C++17 library and CLI for safety-critical and performance-critical operations:
- Preprocessing & headless palm ROI handling
- Cosine similarity matching and ambiguity guard
- Challenge-response liveness gesture classification & temporal sequence verification
- CLI subcommands for status, liveness, match, and embed

## Build Requirements
- C++17 compatible compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.16+
- Optional: OpenCV 4.x/5.x (`core`, `imgproc`). If OpenCV is not available, the codebase falls back automatically to its self-contained headless vision implementation without failing the build.

## Build Instructions
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

## CLI Subcommands
The compiled binary `cpp/build/palmpay_native` supports four primary subcommands:
1. `status` - Reports JSON with version, onnx availability, and build mode.
2. `liveness` - Accepts JSON via stdin with `{"challenge": [...], "frames": [...]}` and outputs verification status.
3. `match` - Accelerates vector cosine similarity operations.
4. `embed` - Scaffold for native neural network inference.

## Python Integration
The Python backend detects the compiled binary via `core/cpp_bridge.py`.
If `NATIVE_CPP_ENABLED = True` in `config.py`, Python routes liveness and matching requests to the native binary with automatic fail-soft fallback to pure Python.
