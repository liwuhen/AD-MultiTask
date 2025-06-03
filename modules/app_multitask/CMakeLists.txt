
set(LIBS "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/libs")
set(CONFIG_PATH
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/config")
set(THIRD_PARTY
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/third_party")
set(WORKSPACE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/workspace")
set(PROCESSOR_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/proprecessor")
set(POSTPRECESSOR_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/postprecessor"
)

find_package(CUDA REQUIRED)

# ##############################################################################
# Build ##
# ##############################################################################
include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/app
  ${CMAKE_CURRENT_LIST_DIR}/appinterface
  ${CMAKE_CURRENT_LIST_DIR}/architecture
  ${CMAKE_CURRENT_LIST_DIR}/architecture/trt
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda
  ${CMAKE_CURRENT_LIST_DIR}/architecture/common
  ${CMAKE_CURRENT_LIST_DIR}/architecture/decode
  ${CMAKE_CURRENT_LIST_DIR}/architecture/module
  ${CMAKE_CURRENT_LIST_DIR}/architecture/quantize
  ${CMAKE_CURRENT_LIST_DIR}/architecture/template
  ${CMAKE_CURRENT_LIST_DIR}/architecture/preprocess
  ${CMAKE_PATH}/modules/benchmark
  ${CMAKE_PATH}/modules/common/json
  ${CMAKE_PATH}/modules/common/utils
  ${CMAKE_PATH}/modules/common/logging
  ${CMAKE_PATH}/modules/common/msg_img
  ${CMAKE_PATH}/modules/common/msg_enum
  ${CMAKE_PATH}/modules/common/msg_struct)

link_libraries(${GLOG_DIR}/lib/libglog.a ${GFLAGS_DIR}/lib/libgflags.a)

if ( "${BUILD_FLAG_LOWER}" STREQUAL "debug" )
  set(CMAKE_CXX_FLAGS_DEBUG 
      "-std=c++14 -Wall -O0 -Wfatal-errors -pthread -g")
  set(CUDA_NVCC_FLAGS
      "-std=c++14 -O0 -Xcompiler -fPIC -g -G -w ${CUDA_GEN_CODE}")
elseif( "${BUILD_FLAG_LOWER}" STREQUAL "release" )
  set(CMAKE_CXX_FLAGS_RELEASE 
      "-std=c++14 -O3 -Wfatal-errors -pthread -w")
  set(CUDA_NVCC_FLAGS
      "-std=c++14 -O3 -Xcompiler -fPIC -w ${CUDA_GEN_CODE}")
endif()

file(
  GLOB_RECURSE
  CPP_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/app/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/appinterface/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/common/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/decode/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/template/*.cc
  ${CMAKE_PATH}/modules/common/logging/*.cpp
  ${CMAKE_PATH}/modules/common/json/*.cpp)

cuda_add_library(${MODEL_FLAG}_gpu_proprecessor SHARED
                 ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/warpaffine.cu)

cuda_add_library(${MODEL_FLAG}_gpu_decode SHARED
                 ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/gpu_decode.cu
                 ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/gpu_seg.cu
                 ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/gpu_nms.cu)

set_target_properties(
  ${MODEL_FLAG}_gpu_proprecessor
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROCESSOR_OUTPUT_DIR}"
             OUTPUT_NAME "${MODEL_FLAG}_gpu_proprecessor")

set_target_properties(
  ${MODEL_FLAG}_gpu_decode
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${POSTPRECESSOR_OUTPUT_DIR}"
             OUTPUT_NAME "${MODEL_FLAG}_gpu_decode")

add_library(${PROJECT_NAME} SHARED ${CPP_SRCS})

target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(${PROJECT_NAME} ${MODEL_FLAG}_gpu_proprecessor ${MODEL_FLAG}_gpu_decode)
target_link_libraries(${PROJECT_NAME} yaml-cpp ${EIGIN_DIR})
target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgcodecs
                      opencv_imgproc opencv_highgui)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBS}"
                             OUTPUT_NAME "${PROJECT_NAME}")

include(${CMAKE_SOURCE_DIR}/cmake/install-toolchain.cmake)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${LIBS})
install(TARGETS ${MODEL_FLAG}_gpu_decode
        LIBRARY DESTINATION ${POSTPRECESSOR_OUTPUT_DIR})
install(TARGETS ${MODEL_FLAG}_gpu_proprecessor
        LIBRARY DESTINATION ${PROCESSOR_OUTPUT_DIR})
