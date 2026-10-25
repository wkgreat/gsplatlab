#pragma once

#include "gsplat.hpp"
#include "matrix_gpu.hpp"
#include <cuda_runtime.h>
#include <driver_types.h>
#include <memory.h>
#include <vector_types.h>

namespace gsplat {
namespace cuda {

inline void CUDA_CHECK_LAST_ERROR(DebugPrintFn print) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        print("kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaGetLastError();
}

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

inline unsigned int BLOCKNUM(unsigned int blockDim, unsigned int dimSize) {
    return (blockDim + dimSize - 1) / blockDim;
}

inline dim3 BLOCKNUM(dim3 blockDim, dim3 dimSize) {
    return {BLOCKNUM(blockDim.x, dimSize.x), BLOCKNUM(blockDim.y, dimSize.y), BLOCKNUM(blockDim.z, dimSize.z)};
}

} // namespace cuda

} // namespace gsplat
