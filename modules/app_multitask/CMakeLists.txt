set(LIBS
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/libs")
set(CONFIG_PATH
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/config")
set(THIRD_PARTY
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/third_party"
)
set(WORKSPACE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/workspace"
)
set(PROCESSOR_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/preprocessor"
)
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
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda
  ${CMAKE_CURRENT_LIST_DIR}/architecture/decode
  ${CMAKE_CURRENT_LIST_DIR}/architecture/module
  ${CMAKE_CURRENT_LIST_DIR}/architecture/quantize
  ${CMAKE_CURRENT_LIST_DIR}/architecture/preprocess
  ${CMAKE_PATH}/modules/backend/tensorrt
  ${CMAKE_PATH}/modules/benchmark
  ${CMAKE_PATH}/modules/common/json
  ${CMAKE_PATH}/modules/common/utils
  ${CMAKE_PATH}/modules/common/module
  ${CMAKE_PATH}/modules/common/msg_img
  ${CMAKE_PATH}/modules/common/template
  ${CMAKE_PATH}/modules/common/msg_enum
  ${CMAKE_PATH}/modules/common/appconfig
  ${CMAKE_PATH}/modules/common/msg_struct)

file(
  GLOB_RECURSE
  CPP_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/app/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/appinterface/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/decode/*.cpp
  ${CMAKE_PATH}/modules/common/json/*.cpp
  ${CMAKE_PATH}/modules/common/logging/*.cpp
  ${CMAKE_PATH}/modules/common/template/*.cc
  ${CMAKE_PATH}/modules/common/appconfig/*.cc)

cuda_add_library(${MODEL_FLAG}_gpu_preprocessor SHARED
                 ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/warpaffine.cu)

cuda_add_library(
  ${MODEL_FLAG}_gpu_decode
  SHARED
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/gpu_decode.cu
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/gpu_seg.cu
  ${CMAKE_CURRENT_LIST_DIR}/architecture/cuda/gpu_nms.cu)

set_target_properties(
  ${MODEL_FLAG}_gpu_preprocessor
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBS}"
             OUTPUT_NAME "${MODEL_FLAG}_gpu_preprocessor")

set_target_properties(
  ${MODEL_FLAG}_gpu_decode
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBS}"
             OUTPUT_NAME "${MODEL_FLAG}_gpu_decode")

add_library(${PROJECT_NAME} SHARED ${CPP_SRCS})

target_link_libraries(${PROJECT_NAME} cudart yaml-cpp ${EIGIN_DIR})
target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgcodecs
                      opencv_imgproc opencv_highgui)
target_link_libraries(${PROJECT_NAME} backend_tensorrt
                      ${MODEL_FLAG}_gpu_preprocessor ${MODEL_FLAG}_gpu_decode)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBS}"
                             OUTPUT_NAME "${PROJECT_NAME}")

include(${CMAKE_SOURCE_DIR}/cmake/install-toolchain.cmake)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${LIBS})
