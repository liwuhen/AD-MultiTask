
set(BACKEND_OUTPUT_DIR
  "${CMAKE_CURRENT_LIST_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/backend"
)

include_directories(${CMAKE_CURRENT_LIST_DIR}/*h)
include_directories(
  ${CMAKE_PATH}/modules/app_multitask/architecture/quantize
  ${CMAKE_PATH}/modules/app_multitask/architecture/common
  ${CMAKE_PATH}/modules/app_multitask/architecture/module
  ${CMAKE_PATH}/modules/app_multitask/architecture/template
  ${CMAKE_PATH}/modules/benchmark
  ${CMAKE_PATH}/modules/common/json
  ${CMAKE_PATH}/modules/common/utils
  ${CMAKE_PATH}/modules/common/logging
  ${CMAKE_PATH}/modules/common/msg_img
  ${CMAKE_PATH}/modules/common/msg_enum
  ${CMAKE_PATH}/modules/common/msg_struct)


file(GLOB_RECURSE CPP_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/architecture/common/*.cpp
  ${CMAKE_PATH}/modules/app_multitask/architecture/template/*.cc
  ${CMAKE_PATH}/modules/common/logging/*.cpp
  ${CMAKE_PATH}/modules/common/json/*.cpp)

link_libraries(${GFLAGS_DIR}/lib/libgflags.a ${GLOG_DIR}/lib/libglog.a)

set(BACKEND backend_tensorrt)
add_library(${BACKEND} SHARED ${CPP_SRCS})

target_link_libraries(${BACKEND} cudart nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(${BACKEND} yaml-cpp ${EIGIN_DIR})

set_target_properties(
  ${BACKEND} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${BACKEND_OUTPUT_DIR}"
  OUTPUT_NAME "${BACKEND}")
install(TARGETS ${BACKEND} LIBRARY DESTINATION ${BACKEND_OUTPUT_DIR})