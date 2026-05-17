cmake_minimum_required(VERSION 3.16)
project(tvm_graph_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TVM_HOME "$ENV{TVM_HOME}" CACHE PATH "Path to TVM source root")
set(TVM_BUILD "$ENV{TVM_BUILD}" CACHE PATH "Path to TVM build dir")

if(NOT TVM_HOME)
  set(TVM_HOME "/opt/tvm")
endif()
if(NOT TVM_BUILD)
  set(TVM_BUILD "${TVM_HOME}/build")
endif()

add_executable(run_tvm_graph run_tvm_graph.cpp)

target_include_directories(run_tvm_graph PRIVATE
  ${TVM_HOME}/include
  ${TVM_HOME}/3rdparty/dlpack/include
  ${TVM_HOME}/3rdparty/dmlc-core/include
  ${CMAKE_CURRENT_LIST_DIR}/../..
)

target_link_directories(run_tvm_graph PRIVATE ${TVM_BUILD})
target_link_libraries(run_tvm_graph PRIVATE tvm_runtime)

if(UNIX AND NOT APPLE)
  set_target_properties(run_tvm_graph PROPERTIES
    BUILD_RPATH "${TVM_BUILD}"
    INSTALL_RPATH "${TVM_BUILD}")
endif()
