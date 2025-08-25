string(TOLOWER "${HARDWARE_PLATFORM_FLAG}" HARDWARE_PLATFORM)
set(APP_STR "" CACHE STRING "Application string version")
if( ${HARDWARE_PLATFORM} STREQUAL "orin" OR ${HARDWARE_PLATFORM} STREQUAL "nvidia" )
  set(APP_STR "9.x" CACHE STRING "Application string version" FORCE)
else()
  set(APP_STR "12.x" CACHE STRING "Application string version" FORCE)
endif()

if ( ${ENABLE_CROSSCOMPILE} )
  set(COMPILER_DIR_FLAG arm)
  set(COMPILER_FLAG aarch64_toolchain)
  if ( ${HARDWARE_PLATFORM} STREQUAL "orin" ) 
    set(CUDA_TOOLKIT_ROOT_DIR "/home/IM/${COMPILER_FLAG}/cuda11.4")
    set(TENSORRT_DIR "/home/IM/${COMPILER_FLAG}/tensorrt8.4")
  else()
    set(CUDA_TOOLKIT_ROOT_DIR "/home/IM/${COMPILER_FLAG}/cuda12.8")
    set(TENSORRT_DIR "/home/IM/${COMPILER_FLAG}/tensorrt10.8")
  endif()
else()
  set(COMPILER_DIR_FLAG x86)
  set(COMPILER_FLAG x86_toolchain)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4/targets/x86_64-linux")
  set(TENSORRT_DIR "/home/IM/${COMPILER_FLAG}/tensorrt8.4")
endif()

set(GLOG_DIR "/home/IM/${COMPILER_FLAG}/glog0.6.0")
set(EIGIN_DIR "/home/IM/${COMPILER_FLAG}/eigen3.4")
set(GFLAGS_DIR "/home/IM/${COMPILER_FLAG}/gflags2.2.2")
set(OPENCV_DIR "/home/IM/${COMPILER_FLAG}/opencv3.4.5")
set(YAMLCPP_DIR "/home/IM/${COMPILER_FLAG}/yaml_cpp")

include_directories(
  ${GFLAGS_DIR}/include
  ${GLOG_DIR}/include
  ${EIGIN_DIR}/include
  ${OPENCV_DIR}/include
  ${YAMLCPP_DIR}/include
  ${TENSORRT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include)

link_directories(
  ${OPENCV_DIR}/lib_${APP_STR} ${YAMLCPP_DIR}/lib
  ${GFLAGS_DIR}/lib ${GLOG_DIR}/lib ${TENSORRT_DIR}/lib/stubs ${CUDA_TOOLKIT_ROOT_DIR}/lib)
