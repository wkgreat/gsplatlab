cmake_minimum_required(VERSION 3.24)

# 默认按当前机器的 GPU 生成（native）。必须在 project() 之前设置：
# CMake 检测到 CUDA 编译器后会先用 nvcc 的默认架构填满该缓存变量，
# 之后再 set() 已经无法生效。
# 无 GPU / 交叉编译时用 -DCMAKE_CUDA_ARCHITECTURES=89 之类的值覆盖。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

# 若 PATH 里没有 nvcc（常见于未 source CUDA 环境的 Linux 终端），
# 回退到 CUDA 的常见安装位置，保证直接 cmake 配置就能用。
if(NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "-NOTFOUND$")
    find_program(_gsplatlab_nvcc nvcc NO_CACHE
        HINTS "/usr/local/cuda/bin" "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin")
    if(_gsplatlab_nvcc)
        set(CMAKE_CUDA_COMPILER "${_gsplatlab_nvcc}" CACHE FILEPATH "CUDA compiler")
    endif()
    unset(_gsplatlab_nvcc)
endif()

project(gsplatlab LANGUAGES CXX CUDA)

# 产物统一放到 <build>/lib 与 <build>/bin：
# pygsplat/setup.py 按这个约定查找 cugsplat 的导入库和运行时库。
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(cugsplat)
