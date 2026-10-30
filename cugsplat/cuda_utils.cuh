#pragma once

#include "gsplat.hpp"
#include "matrix_gpu.hpp"
#include <cuda_runtime.h>
#include <driver_types.h>
#include <memory.h>
#include <vector_types.h>
#include <nvtx3/nvToolsExt.h>
#include <string>

namespace gsplat {
namespace cuda {

#define CUDA_CHECK_LAST_ERROR()                                                                                        \
    do {                                                                                                               \
        cudaError_t err = cudaDeviceSynchronize();                                                                     \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

// 只用于 kernel launch：launch 表达式本身是 void，不能赋给 cudaError_t，
// 必须在 launch 之后用 cudaPeekAtLastError() 取启动错误。
#define CUDATRACE(tag, ...)                                                                                            \
    do {                                                                                                               \
        nvtxRangePushA(tag);                                                                                           \
        __VA_ARGS__;                                                                                                   \
        cudaError_t err = cudaPeekAtLastError();                                                                       \
        nvtxRangePop();                                                                                                \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
        }                                                                                                              \
    } while (0)

inline unsigned int BLOCKNUM(unsigned int blockDim, unsigned int dimSize) {
    return (blockDim + dimSize - 1) / blockDim;
}

inline dim3 BLOCKNUM(dim3 blockDim, dim3 dimSize) {
    return {BLOCKNUM(blockDim.x, dimSize.x), BLOCKNUM(blockDim.y, dimSize.y), BLOCKNUM(blockDim.z, dimSize.z)};
}

inline cudaError_t cudaWarmUp() { return cudaFree(0); }

} // namespace cuda

} // namespace gsplat
