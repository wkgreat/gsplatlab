#pragma once

#include "gsplat.hpp"
#include <cuda_runtime.h>
#include <memory.h>

namespace gsplat {
namespace cuda {

namespace kernel {

template <typename T> __global__ void max(T *nums, const int S, const int N) {

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int id0 = id * S;
  int id1 = (id + 1) * S;

  if (id1 < N) {
    if (nums[id0] < nums[id1]) {
      nums[id0] = nums[id1];
    }
  }
}

template <typename T>
__global__ void divide(T *nums, const int d, const int N) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < N) {
    nums[id] /= d;
  }
}

} // namespace kernel

inline void CUDA_CHECK_LAST_ERROR(DebugPrintFn print) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    print("kernel launch error: %s\n", cudaGetErrorString(err));
  }
  cudaGetLastError();
}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      return;                                                                  \
    }                                                                          \
  } while (0)

} // namespace cuda
} // namespace gsplat
