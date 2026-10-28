add_library(cugsplat SHARED
    cuda_utils.cu
    gsplat_rasterizer.cu
    gsplat_tile.cu
)

target_include_directories(cugsplat PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# CUDA 13 把 CUB/Thrust 移到了 <toolkit>/include/cccl，需要显式加入搜索路径。
foreach(_cuda_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${_cuda_inc}/cccl/cub/cub.cuh")
        target_include_directories(cugsplat PRIVATE "${_cuda_inc}/cccl")
    endif()
endforeach()
unset(_cuda_inc)
target_compile_definitions(cugsplat PRIVATE CUGSPLAT_EXPORTS NOMINMAX)
target_link_libraries(cugsplat PRIVATE CUDA::cudart)

set_target_properties(cugsplat PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # gsplat_tile.cu 会调用 gsplat_rasterizer.cu 里定义的 compute_weight，
    # 设备端跨 TU 调用需要 relocatable device code + device link。
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA 13 的 CCCL 头需要新预处理器；中文注释需要 /utf-8。
if(MSVC)
    target_compile_options(cugsplat PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
    )
endif()
