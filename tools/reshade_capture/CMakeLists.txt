cmake_minimum_required(VERSION 3.24)
project(dlss5_reshade_capture LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MinHook STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/deps/minhook/src/buffer.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/deps/minhook/src/hde/hde32.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/deps/minhook/src/hde/hde64.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/deps/minhook/src/hook.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/deps/minhook/src/trampoline.c")
target_include_directories(MinHook PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/deps/minhook/include")

add_library(dlss5_reshade_capture MODULE dlss5_reshade_capture.cpp)
target_include_directories(dlss5_reshade_capture PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade/include")
target_compile_definitions(dlss5_reshade_capture PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(dlss5_reshade_capture PRIVATE MinHook)
set_target_properties(dlss5_reshade_capture PROPERTIES
    PREFIX ""
    SUFFIX ".addon64"
    OUTPUT_NAME "dlss5_reshade_capture"
    MSVC_RUNTIME_LIBRARY "MultiThreaded")
