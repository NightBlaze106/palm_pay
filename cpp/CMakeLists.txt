cmake_minimum_required(VERSION 3.16)
project(palmpay_native VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

include_directories(include)

# OpenCV: Only search for specific core/imgproc components if explicitly desired,
# avoiding broken system VTK/HDF bindings in arch opencv_viz/opencv_hdf.
find_package(OpenCV QUIET COMPONENTS core imgproc)
if(OpenCV_FOUND)
    message(STATUS "OpenCV core/imgproc found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_compile_definitions(PALMPAY_HAVE_OPENCV=1)
    set(PALMPAY_EXTRA_LIBS opencv_core opencv_imgproc)
else()
    message(STATUS "OpenCV not found; compiling with built-in headless vision routines.")
    add_compile_definitions(PALMPAY_HAVE_OPENCV=0)
    set(PALMPAY_EXTRA_LIBS "")
endif()

# Core static library
add_library(palmpay_core STATIC
    src/matcher.cpp
    src/liveness.cpp
    src/preprocess.cpp
)

if(PALMPAY_EXTRA_LIBS)
    target_link_libraries(palmpay_core PUBLIC ${PALMPAY_EXTRA_LIBS})
endif()

# CLI Executable
add_executable(palmpay_native src/main.cpp)
target_link_libraries(palmpay_native PRIVATE palmpay_core)

# C++ Automated Unit Tests
enable_testing()

add_executable(test_matcher tests/test_matcher.cpp)
target_link_libraries(test_matcher PRIVATE palmpay_core)
add_test(NAME test_matcher COMMAND test_matcher)

add_executable(test_liveness tests/test_liveness.cpp)
target_link_libraries(test_liveness PRIVATE palmpay_core)
add_test(NAME test_liveness COMMAND test_liveness)

add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE palmpay_core)
add_test(NAME test_preprocess COMMAND test_preprocess)

add_executable(test_cli tests/test_cli.cpp)
target_link_libraries(test_cli PRIVATE palmpay_core)
add_test(NAME test_cli COMMAND test_cli "$<TARGET_FILE:palmpay_native>")
