cmake_minimum_required(VERSION 3.10.2)

option(ENABLE_CROSSCOMPILE "Enable crosscomplile for arm platform" OFF)
set(CMAKE_BUILD_TYPE "default" CACHE STRING "Cmake build type")
set(MODEL_FLAG "default" CACHE STRING "Model configuration flag")

add_compile_options(-std=c++14)
set(CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR})

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_FLAG_LOWER)
if ( ${BUILD_FLAG_LOWER} STREQUAL "debug")
  add_definitions("-g")
  set(CMAKE_BUILD_TYPE Debug)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/third_library.cmake)

if(${ENABLE_CROSSCOMPILE})
  include(${CMAKE_PATH}/platforms/linux/arm-toolchain.cmake)
endif()

configure_file("${CMAKE_CURRENT_LIST_DIR}/modules/common/utils/std_cmake.h.in"
               "${CMAKE_CURRENT_LIST_DIR}/modules/common/utils/std_cmake.h")

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/modules
  ${CMAKE_CURRENT_LIST_DIR}/modules/app_multitask
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/utils
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/logging
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_img
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_enum
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_struct)

add_subdirectory(modules/app_multitask)
add_subdirectory(runapp/${MODEL_FLAG})
