cmake_minimum_required(VERSION 3.10)
project(TVM_ONNX_CPP_Example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the path to your TVM installation
# You might need to adjust this depending on where you built/installed TVM
set(TVM_ROOT /path/to/your/tvm) # e.g., /home/user/tvm

# Add TVM include directories
include_directories(
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
)

# Find TVM runtime library
find_library(TVM_RUNTIME_LIBRARY
    NAMES tvm_runtime
    PATHS ${TVM_ROOT}/build/
          ${TVM_ROOT}/build/lib/ # Common path for built libraries
          /usr/local/lib
          /usr/lib
)

if(NOT TVM_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find TVM runtime library. Please set TVM_ROOT correctly.")
endif()

add_executable(tvm_onnx_example main.cpp)
target_link_libraries(tvm_onnx_example PUBLIC ${TVM_RUNTIME_LIBRARY})

# If using GPU/Vulkan, you might need to link against those as well
# For example, for Vulkan:
# find_package(Vulkan REQUIRED)
# target_link_libraries(tvm_onnx_example PUBLIC Vulkan::Vulkan)