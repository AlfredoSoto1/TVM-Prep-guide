cmake_minimum_required(VERSION 3.16)
project(tvm_min_runner CXX)
set(CMAKE_CXX_STANDARD 17)

# Set these or pass with -DTVM_HOME=... -DTVM_BUILD=...
set(TVM_HOME "/workspaces/TVM-Prep-guide/tvm" CACHE PATH "Path to TVM source root")
set(TVM_BUILD "/workspaces/TVM-Prep-guide/tvm/build" CACHE PATH "Path to TVM build dir")

add_executable(run_min run_resnet.cpp)

target_include_directories(run_min PRIVATE
  ${TVM_HOME}/include
  ${TVM_HOME}/3rdparty/dlpack/include
  ${TVM_HOME}/3rdparty/dmlc-core/include
)

target_link_directories(run_min PRIVATE
  ${TVM_BUILD}
)

target_link_libraries(run_min PRIVATE tvm_runtime)

# Make the runtime .so discoverable when you run from the build tree
if(UNIX AND NOT APPLE)
  set_target_properties(run_min PROPERTIES
    BUILD_RPATH "${TVM_BUILD}"
    INSTALL_RPATH "${TVM_BUILD}")
endif()
