#!/usr/bin/env bash
#
# ==================================================================
# Copyright (c) 2024, LiWuHen.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an
# BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================
#/

set -e

. ./scripts/config.sh

parse_args "$@"
CMAKE_DIR=${HOME_DIR}
BUILD_DIR=${HOME_DIR}/build_clang

if [ "${Debug_FLAG}" == "ON" ] ; then
    TBUILD_VERSION="Debug"
fi
COMM_ARGS=" -DCMAKE_BUILD_TYPE=${TBUILD_VERSION} -DMODEL_FLAG=${MODEL_FLAG}
            -DHARDWARE_PLATFORM_FLAG=${HARDWARE_PLATFORM_FLAG}"

if [ "${CROSS_COMPILE}" == "ON" ] ; then
    COMPILER_FLAG="arm"
    case ${HARDWARE_PLATFORM_FLAG} in
        ORIN)
            AARCM_FLAG=9.x
            BUILD_DIR=${HOME_DIR}/build_orin_arm
            COMM_ARGS="${COMM_ARGS} ${CMAKE_ARM_ARGS}"
            ;;
        THOR)
            AARCM_FLAG=12.x
            BUILD_DIR=${HOME_DIR}/build_thor_arm
            COMM_ARGS="${COMM_ARGS} ${CMAKE_ARM_ARGS}"
            ;;
    esac
    COMM_ARGS="${COMM_ARGS} -DENABLE_CROSSCOMPILE=ON -DAARCM_FLAG=${AARCM_FLAG}"
elif [ "${PC_X86_FLAG}" == "ON" ] ; then
    if [ "${HARDWARE_PLATFORM_FLAG}" == "NVIDIA" ] ; then
        BUILD_DIR=${HOME_DIR}/build_nv_x86
    fi
fi

echo -e "\e[1m\e[34m[Bash-Platform-${TIME}]: compiler platform: ${HARDWARE_PLATFORM_FLAG} \e[0m"

if [ "${CLEAN_FLAG}" == "ON" ] ; then
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"

echo -e "\e[1m\e[34m[Bash-Compiler-${TIME}]: cmake ${COMM_ARGS} ${CMAKE_DIR} \e[0m"
echo -e "\e[1m\e[34m[Bash-Compiler-${TIME}]: Building.... \e[0m"

LIB_PATH=""
APP_PATH=""
if [ "${MODEL_FLAG}" == "yolop"  ]; then
    LIB_PATH=app_multitask
    APP_PATH=yolop
elif [ "${MODEL_FLAG}" == "a_yolom" ]; then
    LIB_PATH=app_multitask
    APP_PATH=a_yolom
fi

# compiler
cmake ${COMM_ARGS} "${CMAKE_DIR}"
make -j8 && cd ..

if [ "${PACK_FLAG}" == "ON" ] ; then

    # install
    INSTALL_PATH=${HOME_DIR}/install_${HARDWARE_PLATFORM_FLAG,,}/${MODEL_FLAG}_bin
    if [ -d "$INSTALL_PATH" ]; then
        echo -e "\e[1m\e[34m[Bash-Pack-${TIME}]: $INSTALL_PATH  \e[0m"
    else
        rm -rf "${INSTALL_PATH}" && mkdir -p  "${INSTALL_PATH}"
    fi

    if [ "${PC_X86_FLAG}" == "ON" ]; then
        rm -rf "${INSTALL_PATH}"/x86
        mv "${HOME_DIR}"/modules/"${LIB_PATH}"/"${MODEL_FLAG}"_bin/x86 "${INSTALL_PATH}"
        mv "${HOME_DIR}"/runapp/"${APP_PATH}"/"${MODEL_FLAG}"_bin/x86/*  "${INSTALL_PATH}"/x86
        mv "${HOME_DIR}"/modules/backend/tensorrt/"${MODEL_FLAG}"_bin/x86/*  "${INSTALL_PATH}"/x86

    elif [ "${CROSS_COMPILE}" == "ON" ] ; then
        rm -rf "${INSTALL_PATH}"/arm
        mv "${HOME_DIR}"/modules/"${LIB_PATH}"/"${MODEL_FLAG}"_bin/arm "${INSTALL_PATH}"
        mv "${HOME_DIR}"/runapp/"${APP_PATH}"/"${MODEL_FLAG}"_bin/arm/*  "${INSTALL_PATH}"/arm
    fi

    rm -rf "${HOME_DIR}"/modules/"${LIB_PATH}"/"${MODEL_FLAG}"_bin
    rm -rf "${HOME_DIR}"/runapp/"${APP_PATH}"/"${MODEL_FLAG}"_bin
    rm -rf "${HOME_DIR}"/modules/backend/tensorrt/"${MODEL_FLAG}"_bin

    cp -r "${HOME_DIR}"/scripts/run.sh "${HOME_DIR}"/install_"${HARDWARE_PLATFORM_FLAG,,}"
    echo -e "\e[1m\e[34m[Bash-Pack-${TIME}]: copy ${COMPILER_FLAG} ${MODEL_FLAG} bin to install  \e[0m"

else
    echo -e "\e[1m\e[34m[Bash-Pack-${TIME}]: pack flag false  \e[0m"

fi
