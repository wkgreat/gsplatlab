#pragma once

#include "gsplat.hpp"
#include "gsplat_matrix.cuh"

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

namespace gsplat {

void showCompileTile();

struct SplatScene {
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
    vec4f *pos3d;
    vec4f *scale;
    vec4f *rot;
    float *opacity;
    float *sh;
};

struct SplatData2D {
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

struct SplatImage {
    int width;
    int height;
    vec4f *image;

    static cudaError_t createAtDevice(SplatImage *d_image, int width, int height) {
        size_t imageByteSize = width * height * sizeof(vec4f);
        d_image->width = width;
        d_image->height = height;
        cudaError_t err = cudaMalloc(&(d_image->image), imageByteSize);
        return err;
    }

    cudaError_t toHost(SplatImage *h_image) {
        cudaError_t err = cudaSuccess;
        size_t imageByteSize = width * height * sizeof(vec4f);
        h_image->width = width;
        h_image->height = height;
        h_image->image = (vec4f *)malloc(imageByteSize);
        if (h_image->image) {
            err = cudaMemcpy(h_image->image, image, imageByteSize, cudaMemcpyDeviceToHost);
        } else {
            err = cudaErrorMemoryAllocation;
        }
        return err;
    }

    cudaError_t toHost(float **h_image) {
        cudaError_t err = cudaSuccess;
        size_t imageByteSize = width * height * sizeof(vec4f);
        *h_image = (float *)malloc(imageByteSize);
        if (*h_image) {
            err = cudaMemcpy(*h_image, image, imageByteSize, cudaMemcpyDeviceToHost);
        } else {
            err = cudaErrorMemoryAllocation;
        }
        return err;
    }

    cudaError_t freeAtDevice() { return cudaFree(image); }
};

cudaError_t gsplat_render_at_device(SplatScene &scene, SplatData &d_splats, SplatImage &d_image_out);

CUGSPLAT_API cudaError_t gsplat_rasterize_gpu(SplatScene &scene, SplatData &splats, float **image,
                                       DebugPrintFn debug_print = nullptr);

} // namespace gsplat