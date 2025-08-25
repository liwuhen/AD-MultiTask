cmake_minimum_required(VERSION 3.10.2)

option(ENABLE_CROSSCOMPILE "Enable crosscomplile for arm platform" OFF)
set(CMAKE_BUILD_TYPE
    "default"
    CACHE STRING "Cmake build type")
set(MODEL_FLAG
    "default"
    CACHE STRING "Model configuration flag")
set(HARDWARE_PLATFORM_FLAG
    "default"
    CACHE STRING "PLATFORM configuration flag")

project(${MODEL_FLAG}_project)

add_compile_options(-std=c++14)
set(CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR})

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_FLAG_LOWER)
if(${BUILD_FLAG_LOWER} STREQUAL "debug")
  add_definitions("-g")
  set(CMAKE_BUILD_TYPE Debug)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/third_library.cmake)

configure_file("${CMAKE_CURRENT_LIST_DIR}/modules/common/utils/std_cmake.h.in"
               "${CMAKE_CURRENT_LIST_DIR}/modules/common/utils/std_cmake.h")

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/modules
  ${CMAKE_CURRENT_LIST_DIR}/modules/app_multitask
  ${CMAKE_CURRENT_LIST_DIR}/modules/common
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/utils
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/module
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/logging
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_img
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_enum
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/template
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/appconfig
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_struct)

if("${BUILD_FLAG_LOWER}" STREQUAL "debug")
  set(CMAKE_CXX_FLAGS_DEBUG
      "-std=c++14 -fPIC -Wall -O0 -Wfatal-errors -pthread -g")
  set(CUDA_NVCC_FLAGS
      "-std=c++14 -O0 -Xcompiler -fPIC -g -G -w ${CUDA_GEN_CODE}")
elseif("${BUILD_FLAG_LOWER}" STREQUAL "release")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -fPIC -O3 -Wfatal-errors -pthread -w")
  set(CUDA_NVCC_FLAGS "-std=c++14 -O3 -Xcompiler -fPIC -w ${CUDA_GEN_CODE}")
endif()

add_subdirectory(modules/backend)
add_subdirectory(modules/app_multitask)
add_subdirectory(runapp/${MODEL_FLAG})
