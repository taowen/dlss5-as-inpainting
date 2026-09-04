cmake_minimum_required(VERSION 3.24)
project(dlss5_cupti_capture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUPTI_ROOT "" CACHE PATH "CUDA CUPTI package root")
set(CUDA_ROOT "" CACHE PATH "CUDA runtime package root")

if(NOT EXISTS "${CUPTI_ROOT}/include/cupti.h")
    message(FATAL_ERROR "CUPTI_ROOT must contain include/cupti.h")
endif()
if(NOT EXISTS "${CUDA_ROOT}/include/cuda.h")
    message(FATAL_ERROR "CUDA_ROOT must contain include/cuda.h")
endif()

add_library(dlss5_cupti_capture SHARED dlss5_cupti_capture.cpp)
target_include_directories(dlss5_cupti_capture PRIVATE
    "${CUPTI_ROOT}/include"
    "${CUDA_ROOT}/include"
)

# The DLL calls CUPTI through function pointers, so it deliberately has no
# import-library dependency. This keeps the probe portable across toolkit
# layouts; the runtime CUPTI DLL is selected by DLSS5_CUPTI_DLL/PATH.
set_property(TARGET dlss5_cupti_capture PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
set_target_properties(dlss5_cupti_capture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
)
