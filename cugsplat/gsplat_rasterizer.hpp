#pragma once

#include "gsplat.hpp"
#include "matrix_gpu.hpp"

// cugsplat 动态库导出宏：库内部编译时定义 CUGSPLAT_EXPORTS，
// 使用方（pygsplat 的 Python 扩展）拿到的是 dllimport。
#ifdef _WIN32
#ifdef CUGSPLAT_EXPORTS
#define CUGSPLAT_API __declspec(dllexport)
#else
#define CUGSPLAT_API __declspec(dllimport)
#endif
#else
#define CUGSPLAT_API __attribute__((visibility("default")))
#endif

using namespace gsplat::matrix::gpu;

namespace gsplat {

void showCompileTile();

struct SplatScene {
    float *viewmtx;
    float *projmtx;
    float *eye;
    float *bgcolor;
    float znear;
    float zfar;
    float fovy;
    int width;
    int height;
};

struct SplatSceneGPU {
    mat4x4f viewmtx;
    mat4x4f projmtx;
    vec4f bgcolor;
    vec3f eye;
    float znear;
    float zfar;
    float fovy;
    int width;
    int height;
};

struct SplatData {
    int nsplats;
    float *pos3d;
    float *scale;
    float *rot;
    float *opacity;
    float *sh;
};

struct SplatDataGPU {
    int nsplats;
    vec4f *pos3d;
    vec4f *scale;
    vec4f *rot;
    float *opacity;
    float *sh;
};

struct SplatData2DGPU {
    int nsplats;
    vec2f *pos2d;
    float *depth;
    mat2x2f *cov2d;
    vec4f *rect;
    vec4f *color;
};

template <typename T> struct Range {
    T min, max;
};

template <typename T> struct Box2D {
    Range<T> x;
    Range<T> y;
};

// 定义在 gsplat_rasterizer.cu；跨 TU 的设备端调用依靠 cugsplat 的 RDC
// (CUDA_SEPARABLE_COMPILATION) 解析。这里不能加 __forceinline__：
// 只有声明没有函数体时会被当成 extern inline，符号不会被产出。
__host__ __device__ float compute_weight(vec2f d, mat2x2f Cov2D);

CUGSPLAT_API void gsplat_rasterize_gpu(SplatScene &scene, SplatData &splats, float **image,
                                       DebugPrintFn debug_print = nullptr);

} // namespace gsplat