set(AARCM_FLAG "default" CACHE STRING "Model configuration flag")
string(TOLOWER "${AARCM_FLAG}" AARCM_GUN_FLAG)
set(CMAKE_C_COMPILER
    /home/IM/aarch64_toolchain/aarch64_gun_${AARCM_GUN_FLAG}/bin/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER
    /home/IM/aarch64_toolchain/aarch64_gun_${AARCM_GUN_FLAG}/bin/aarch64-buildroot-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH
    /home/IM/aarch64_toolchain/aarch64_gun_${AARCM_GUN_FLAG}/aarch64-buildroot-linux-gnu/sysroot)
