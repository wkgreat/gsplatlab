add_library(cugsplat SHARED
    cuda_utils.cu
    gsplat_rasterizer.cu
    gsplat_tile_index.cu
)

target_include_directories(cugsplat PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(cugsplat PRIVATE CUGSPLAT_EXPORTS NOMINMAX)
target_link_libraries(cugsplat PRIVATE CUDA::cudart)

set_target_properties(cugsplat PROPERTIES
    POSITION_INDEPENDENT_CODE ON
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
