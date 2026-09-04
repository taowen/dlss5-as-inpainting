cmake_minimum_required(VERSION 3.24)
project(stereo_harness LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# Generate from the pinned submodule; never edit upstream or depend on a temp checkout.
file(READ "${REPO}/third_party/DLSS5-Image-Converter/native/dlss5_eval/main.cpp" SOURCE)
set(ANCHOR "    void UploadDepth(const std::wstring& path);")
string(FIND "${SOURCE}" "${ANCHOR}" FOUND)
if(FOUND EQUAL -1)
  message(FATAL_ERROR "Pinned harness declaration changed")
endif()
string(REPLACE "${ANCHOR}" "${ANCHOR}\n    void UploadMotion(const std::wstring& path);" SOURCE "${SOURCE}")
string(REPLACE "void Harness::UploadDepth(" [=[void Harness::UploadMotion(const std::wstring& path) {
    const size_t pixels = static_cast<size_t>(options_.width) * options_.height;
    UploadTexture(motion_, ReadFile(path, pixels * 4));
    Execute();
}

void Harness::UploadDepth(]=] SOURCE "${SOURCE}")
string(REPLACE "        } else if (command == \"DEPTH\") {" [=[        } else if (command == "MOTION") {
            std::string rest;
            std::getline(parts, rest);
            const std::string path = Trim(rest);
            if (path.empty()) Fail("MOTION requires an RG16F file");
            harness.UploadMotion(Widen(path));
            Emit("MOTION_OK");
        } else if (command == "DEPTH") {]=] SOURCE "${SOURCE}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" "${SOURCE}")
add_executable(dlss5_stereo_eval "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
target_include_directories(dlss5_stereo_eval PRIVATE "${REPO}/third_party/NVIDIA-DLSS/include")
target_link_directories(dlss5_stereo_eval PRIVATE "${REPO}/third_party/NVIDIA-DLSS/lib/Windows_x86_64/x64")
target_link_libraries(dlss5_stereo_eval PRIVATE nvsdk_ngx_s d3d12 dxgi dxguid user32)
set_property(TARGET dlss5_stereo_eval PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
