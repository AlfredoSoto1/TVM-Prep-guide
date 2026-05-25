cmake_minimum_required(VERSION 3.16)
project(tvm_graph_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TVM_HOME "$ENV{TVM_HOME}" CACHE PATH "Optional path to TVM source root")
set(TVM_RUNTIME_TARGET "" CACHE STRING "Runtime target under examples/artifacts/runtime")
set(TVM_RUNTIME_DIR "" CACHE PATH "Directory containing packaged libtvm_runtime.so")
set(TVM_INCLUDE_DIR "" CACHE PATH "Directory containing packaged TVM C++ headers")

if(NOT TVM_RUNTIME_TARGET)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(default_runtime_target "raspi4_aarch64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7)")
    set(default_runtime_target "raspi_armv7")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(default_runtime_target "x86_64")
  endif()

  if(default_runtime_target AND
      EXISTS "${CMAKE_CURRENT_LIST_DIR}/../artifacts/runtime/${default_runtime_target}/libtvm_runtime.so")
    set(TVM_RUNTIME_TARGET "${default_runtime_target}" CACHE STRING
      "Runtime target under examples/artifacts/runtime" FORCE)
  endif()
endif()

if(NOT TVM_RUNTIME_DIR AND TVM_RUNTIME_TARGET)
  set(candidate_runtime_dir "${CMAKE_CURRENT_LIST_DIR}/../artifacts/runtime/${TVM_RUNTIME_TARGET}")
  if(EXISTS "${candidate_runtime_dir}/libtvm_runtime.so")
    set(TVM_RUNTIME_DIR "${candidate_runtime_dir}" CACHE PATH
      "Directory containing packaged libtvm_runtime.so" FORCE)
  endif()
endif()

if(NOT TVM_RUNTIME_DIR)
  file(GLOB runtime_dirs LIST_DIRECTORIES true
    "${CMAKE_CURRENT_LIST_DIR}/../artifacts/runtime/*")
  set(runtime_targets "")
  foreach(runtime_dir ${runtime_dirs})
    if(EXISTS "${runtime_dir}/libtvm_runtime.so")
      get_filename_component(runtime_target "${runtime_dir}" NAME)
      list(APPEND runtime_targets "${runtime_target}")
    endif()
  endforeach()
  list(LENGTH runtime_targets runtime_target_count)
  if(runtime_target_count EQUAL 1)
    list(GET runtime_targets 0 only_runtime_target)
    set(TVM_RUNTIME_TARGET "${only_runtime_target}" CACHE STRING
      "Runtime target under examples/artifacts/runtime" FORCE)
    set(TVM_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/../artifacts/runtime/${only_runtime_target}"
      CACHE PATH "Directory containing packaged libtvm_runtime.so" FORCE)
  endif()
endif()

if(NOT EXISTS "${TVM_RUNTIME_DIR}/libtvm_runtime.so")
  message(FATAL_ERROR
    "Cannot find libtvm_runtime.so. Copy examples/artifacts/runtime/<target>/ "
    "to this device, or pass -DTVM_RUNTIME_DIR=/path/to/runtime-package.")
endif()

if(NOT TVM_INCLUDE_DIR AND EXISTS "${TVM_RUNTIME_DIR}/include/tvm/runtime/module.h")
  set(TVM_INCLUDE_DIR "${TVM_RUNTIME_DIR}/include" CACHE PATH
    "Directory containing packaged TVM C++ headers" FORCE)
endif()

if(NOT TVM_INCLUDE_DIR)
  if(NOT TVM_HOME)
    foreach(candidate "$ENV{TVM_HOME}" "$ENV{HOME}/tvm" "/opt/tvm")
      if(NOT TVM_HOME AND EXISTS "${candidate}/include/tvm/runtime/module.h")
        set(TVM_HOME "${candidate}" CACHE PATH "Optional path to TVM source root" FORCE)
      endif()
    endforeach()
  endif()

  if(EXISTS "${TVM_HOME}/include/tvm/runtime/module.h")
    set(TVM_INCLUDE_DIR "${TVM_HOME}/include" CACHE PATH
      "Directory containing packaged TVM C++ headers" FORCE)
  endif()
endif()

if(NOT EXISTS "${TVM_INCLUDE_DIR}/tvm/runtime/module.h")
  message(FATAL_ERROR
    "Cannot find TVM C++ headers. Rebuild/copy the runtime package produced by "
    "compilation/build_runtime.sh, or pass -DTVM_INCLUDE_DIR=/path/to/include.")
endif()

set(TVM_DLPACK_INCLUDE "${TVM_HOME}/3rdparty/dlpack/include")
set(TVM_DMLC_INCLUDE "${TVM_HOME}/3rdparty/dmlc-core/include")

if(EXISTS "${TVM_INCLUDE_DIR}/dlpack/dlpack.h")
  set(DLPACK_INCLUDE "${TVM_INCLUDE_DIR}")
elseif(EXISTS "${TVM_DLPACK_INCLUDE}/dlpack/dlpack.h")
  set(DLPACK_INCLUDE "${TVM_DLPACK_INCLUDE}")
else()
  message(FATAL_ERROR
    "Cannot find dlpack/dlpack.h. Rebuild/copy the runtime package produced by "
    "compilation/build_runtime.sh, or run `git submodule update --init --recursive` inside TVM.")
endif()

if(EXISTS "${TVM_INCLUDE_DIR}/dmlc/logging.h")
  set(DMLC_INCLUDE "${TVM_INCLUDE_DIR}")
elseif(EXISTS "${TVM_DMLC_INCLUDE}/dmlc/logging.h")
  set(DMLC_INCLUDE "${TVM_DMLC_INCLUDE}")
else()
  message(FATAL_ERROR
    "Cannot find dmlc-core headers. Rebuild/copy the runtime package produced by "
    "compilation/build_runtime.sh, or run `git submodule update --init --recursive` inside TVM.")
endif()

add_executable(run_tvm_graph run_tvm_graph.cpp)

target_include_directories(run_tvm_graph PRIVATE
  ${TVM_INCLUDE_DIR}
  ${DLPACK_INCLUDE}
  ${DMLC_INCLUDE}
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_directories(run_tvm_graph PRIVATE ${TVM_RUNTIME_DIR})
target_link_libraries(run_tvm_graph PRIVATE tvm_runtime)

if(UNIX AND NOT APPLE)
  set_target_properties(run_tvm_graph PROPERTIES
    BUILD_RPATH "${TVM_RUNTIME_DIR};$ORIGIN/../../artifacts/runtime/${TVM_RUNTIME_TARGET}"
    INSTALL_RPATH "${TVM_RUNTIME_DIR};$ORIGIN/../../artifacts/runtime/${TVM_RUNTIME_TARGET}")
endif()

message(STATUS "TVM runtime package: ${TVM_RUNTIME_DIR}")
message(STATUS "TVM include dir: ${TVM_INCLUDE_DIR}")
