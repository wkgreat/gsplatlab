#include "cuda_utils.cuh"
#include "gsplat_rasterizer.hpp"
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include "gsplat_tile.cuh"
#include "gsplat_math.cuh"

namespace gsplat {

void showCompileTile() {
    const char *_date = __DATE__;
    const char *_time = __TIME__;
    fprintf(stdout, "gsplatlab compile time: %s %s\n", _date, _time);
}

__global__ void image_set_color_kernel(SplatImage d_image, vec4f color) {

    int wid = blockDim.x * blockIdx.x + threadIdx.x;
    int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (wid < d_image.width && hid < d_image.height) {
        int idx = hid * d_image.width + wid;
        d_image.image[idx] = color;
    }
}

cudaError_t splat_image_set_color_at_device(SplatImage d_image, vec4f color) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((blockDim.x + d_image.width - 1) / blockDim.x, (blockDim.y + d_image.height - 1) / blockDim.y, 1);
    image_set_color_kernel<<<gridDim, blockDim, 0>>>(d_image, color);
    CUDA_CHECK_LAST_ERROR();
    return cudaSuccess;
}

__global__ void gsplat_compute_kernel(SplatScene d_scene, SplatData d_splats, SplatData2D d_splats2d) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < 0 || id >= d_splats.nsplats) {
        return;
    }

    vec4f worldpos = d_splats.pos3d[id];
    vec4f scale = d_splats.scale[id];
    vec4f rot = d_splats.rot[id];
    float opacity = d_splats.opacity[id];
    float *sh = d_splats.sh + 48 * id;

    vec4f viewpos = d_scene.viewmtx * worldpos;
    vec4f ndcpos = d_scene.projmtx * viewpos;
    ndcpos = ndcpos / ndcpos.w;
    vec2f viewport{d_scene.width * 1.0f, d_scene.height * 1.0f};
    vec2f pixpos = (ndcpos.xy() * 0.5 + 0.5) * viewport; // 这里不flipY，最终图像生成完成，再统一flip
    mat2x3f J = compute_jacobian(viewpos, viewport, d_scene.fovy);

    mat3x3f S = mat3x3f(exp(scale.x), 0, 0, 0, exp(scale.y), 0, 0, 0, exp(scale.z));

    mat3x3f R = quat_to_mat3(vec4f(rot.y, rot.z, rot.w, rot.x));
    mat3x3f Cov3D = R * S * S.T() * R.T();
    mat3x3f viewmtx3(d_scene.viewmtx.r0.xyz(), d_scene.viewmtx.r1.xyz(), d_scene.viewmtx.r2.xyz());
    Cov3D = viewmtx3 * (Cov3D * viewmtx3.T());
    mat2x2f Cov2D = J * (Cov3D * J.T());

    Cov2D[0][0] += 0.3;
    Cov2D[1][1] += 0.3;

    vec4f rect = compute_rect(pixpos, Cov2D);
    vec3f direct = normalize(worldpos.xyz() - d_scene.eye);

    vec3f color3 = compute_color(reinterpret_cast<vec3f *>(sh), direct);
    vec4f color4 = vec4f(color3, sigmoid(opacity));

    d_splats2d.pos2d[id] = pixpos;
    d_splats2d.depth[id] = -viewpos.z; // 注意: 这里的深度反号了一下
    d_splats2d.cov2d[id] = Cov2D;
    d_splats2d.rect[id] = rect;
    d_splats2d.color[id] = color4;
}

cudaError_t gsplat_compute_at_device(SplatScene d_scene, SplatData d_splats, SplatData2D d_splats2d) {
    int blockDim = 128;
    int gridDim = (blockDim + d_splats.nsplats - 1) / blockDim;
    gsplat_compute_kernel<<<gridDim, blockDim>>>(d_scene, d_splats, d_splats2d);
    CUDA_CHECK_LAST_ERROR();
    return cudaSuccess;
}

// __global__ void gsplat_pixel_splat_count_kernel(SplatScene d_scene, SplatData2D d_splats2d, int *out_pixSplatCnt) {

//     int id = blockDim.x * blockIdx.x + threadIdx.x;
//     if (id >= d_splats2d.nsplats) {
//         return;
//     }

//     vec4f rect = d_splats2d.rect[id];

//     int xmin = (int)floor(rect.x);
//     int ymin = (int)floor(rect.y);
//     int xmax = (int)ceil(rect.z);
//     int ymax = (int)ceil(rect.w);
//     if (xmin >= d_scene.width || xmax < 0 || ymin >= d_scene.height || ymax < 0) {
//         return;
//     }

//     vec2f centerpos = d_splats2d.pos2d[id];
//     mat2x2f cov2d = d_splats2d.cov2d[id];
//     vec2f pixpos;
//     vec2f d;
//     float w;

//     for (int x = xmin; x <= xmax; ++x) {
//         for (int y = ymin; y <= ymax; ++y) {
//             pixpos = vec2f(x, y);
//             d = pixpos - centerpos;
//             w = compute_weight(d, cov2d);
//             if (w >= 0.001) {
//                 atomicAdd(out_pixSplatCnt + y * d_scene.width + x, 1);
//             }
//         }
//     }
// }

// void gsplat_pixel_splat_count_gpu(SplatScene d_scene, SplatData2D d_splats2d, int **out_pixSplatCnt) {
//     int blockDim = 256;
//     int gridDim = (blockDim + d_splats2d.nsplats - 1) / blockDim;

//     cudaMalloc(out_pixSplatCnt, d_scene.width * d_scene.height * sizeof(int));
//     cudaMemset(*out_pixSplatCnt, 0, d_scene.width * d_scene.height * sizeof(int));

//     gsplat_pixel_splat_count_kernel<<<gridDim, blockDim>>>(d_scene, d_splats2d, *out_pixSplatCnt);
// }

__host__ __device__ static size_t bitCeil(size_t n) {
    if (n <= 1) {
        return 1;
    }
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__host__ __device__ void swapInt(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

__global__ void fillInt32Kernel(int *d_vals, int val, size_t n) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n)
        return;

    d_vals[i] = val;
}

__global__ void gsplat_tile_bitonic_sort(SplatData2D splat2d, int *d_entries, int n, int k, int j, bool asc) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    int a = i ^ j;

    if (i < a) {
        bool up = ((i & k) == 0) == asc; // i%k==0 equalTo i/k%2==0

        float EXTREM = asc ? FLT_MAX : FLT_MIN;
        float depth0 = d_entries[i] == -1 ? EXTREM : splat2d.depth[d_entries[i]];
        float depth1 = d_entries[a] == -1 ? EXTREM : splat2d.depth[d_entries[a]];

        if (up ? (depth0 > depth1) : (depth0 < depth1)) {
            swapInt(d_entries[i], d_entries[a]);
        }
    }
}

cudaError_t gsplat_tile_splat_sort_at_device(SplatData2D splats2d, SplatTileInfo h_tileinfo, SplatTileInfo d_tileinfo,
                                             int tileId, int *d_cache) {

    int n = h_tileinfo.index.stride[tileId];
    if (n < 2) {
        return cudaSuccess;
    }

    int m = bitCeil(n);

    int blockDim = 256;
    int blockNum = (blockDim + m - 1) / blockDim;

    int s = h_tileinfo.index.offset[tileId];

    fillInt32Kernel<<<blockNum, blockDim>>>(d_cache, -1, m);
    CUDA_CHECK(cudaMemcpy(d_cache, d_tileinfo.index.entries + s, sizeof(int) * n, cudaMemcpyDeviceToDevice));

    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            gsplat_tile_bitonic_sort<<<blockNum, blockDim>>>(splats2d, d_cache, m, k, j, true);
            CUDA_CHECK_LAST_ERROR();
        }
    }

    CUDA_CHECK(cudaMemcpy(d_tileinfo.index.entries + s, d_cache, sizeof(int) * n, cudaMemcpyDeviceToDevice));

    return cudaSuccess;
}

cudaError_t gsplat_data2d_sort_at_device(SplatData2D splats2d, SplatTileInfo d_tileinfo) {

    SplatTileInfo h_tileinfo;
    gsplat_tileinfo_to_host(d_tileinfo, h_tileinfo);

    size_t maxTileSplatCount = getTileMaxNumSplats(h_tileinfo);
    size_t m = bitCeil(maxTileSplatCount);

    int *d_cache;
    CUDA_CHECK(cudaMalloc(&d_cache, m * sizeof(int)));

    for (int tileId = 0; tileId < h_tileinfo.nums.x * h_tileinfo.nums.y; ++tileId) {
        CUDA_CHECK(gsplat_tile_splat_sort_at_device(splats2d, h_tileinfo, d_tileinfo, tileId, d_cache));
    }

    CUDA_CHECK(cudaFree(d_cache));
    free(h_tileinfo.index.offset);
    free(h_tileinfo.index.stride);
    free(h_tileinfo.index.entries);

    return cudaSuccess;
}

__global__ void gsplat_rasterize_kernel(SplatScene scene, SplatData2D splats2d, float *out_image) {

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= splats2d.nsplats) {
        return;
    }

    vec2f pixpos = splats2d.pos2d[id];
    vec4f color = splats2d.color[id];
    int x = (int)pixpos.x;
    int y = (int)pixpos.y;
    if (x >= 0 && x < scene.width && y >= 0 && y < scene.height) {
        float *pixel = out_image + (y * scene.width + x) * 4;
        pixel[0] = color[0];
        pixel[1] = color[1];
        pixel[2] = color[2];
        pixel[3] = color[3];
    }
}

__global__ void gsplat_tile_paint_kernel(SplatScene scene, SplatData2D splats2d, SplatTileInfo tileinfo, bool *imgmask,
                                         vec4f *image) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx >= tileinfo.nums.x || ty >= tileinfo.nums.y) {
        return;
    }
    Box2D<int> tilebox = gsplat_tile_box(scene, tileinfo, tx, ty);
    int tid = ty * tileinfo.nums.x + tx;
    int offset = tileinfo.index.offset[tid];
    int stride = tileinfo.index.stride[tid];
    int splatid;

    // for each splat of tile
    for (int i = 0; i < stride; ++i) {
        splatid = tileinfo.index.entries[offset + i];
        mat2x2f cov2d = splats2d.cov2d[splatid];
        vec2f pos2d = splats2d.pos2d[splatid];
        vec4f rect = splats2d.rect[splatid];
        vec4f color = splats2d.color[splatid];
        vec3f rgb(color.x, color.y, color.z);
        float opa = color.w;

        // splat rectangle
        int pxmin = (int)ceilf(rect.x - 0.5f);
        int pymin = (int)ceilf(rect.y - 0.5f);
        int pxmax = (int)floorf(rect.z - 0.5f);
        int pymax = (int)floorf(rect.w - 0.5f);

        // for each pixel
        for (int px = pxmin; px <= pxmax; px++) {
            for (int py = pymin; py <= pymax; py++) {
                if (px >= tilebox.x.min && px <= tilebox.x.max && py >= tilebox.y.min && py <= tilebox.y.max) {
                    vec2f p(px + 0.5f, py + 0.5f);
                    vec2f d = p - pos2d;

                    float weight = compute_weight(d, cov2d);
                    if (weight >= 0.001) {
                        float alpha = opa * weight;

                        vec4f rgba = image[py * scene.width + px];
                        rgba.x += rgba.w * alpha * rgb.x;
                        rgba.y += rgba.w * alpha * rgb.y;
                        rgba.z += rgba.w * alpha * rgb.z;

                        rgba.w *= (1 - alpha); // transmission

                        image[py * scene.width + px] = rgba;
                    }
                    imgmask[py * scene.width + px] = true;
                }
            }
        }
    }
}

__global__ void gsplat_image_opacity_kernel(vec4f *d_image, bool *d_imgmask, SplatScene d_scene) {

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < d_scene.width && y < d_scene.height) {
        unsigned int idx = y * d_scene.width + x;
        if (d_imgmask[idx]) {
            vec4f rgba = d_image[idx];
            rgba.x = rgba.x + rgba.w * d_scene.bgcolor.x;
            rgba.y = rgba.y + rgba.w * d_scene.bgcolor.y;
            rgba.z = rgba.z + rgba.w * d_scene.bgcolor.z;
            rgba.w = 1.0; // opacity
            d_image[idx] = rgba;
        }
    }
}

void gsplat_tile_paint(SplatScene d_scene, SplatData2D d_splats2d, SplatTileInfo d_tileinfo, vec4f *d_image) {
    dim3 blockDim(16, 16);
    dim3 gridDim((blockDim.x + d_tileinfo.nums.x - 1) / blockDim.x, (blockDim.y + d_tileinfo.nums.y - 1) / blockDim.y,
                 1);
    bool *d_imgmask;
    cudaMalloc(&d_imgmask, d_scene.width * d_scene.height * sizeof(bool));
    cudaMemset(d_imgmask, 0, d_scene.width * d_scene.height * sizeof(bool));
    gsplat_tile_paint_kernel<<<gridDim, blockDim>>>(d_scene, d_splats2d, d_tileinfo, d_imgmask, d_image);

    dim3 gridDim2((blockDim.x + d_scene.width - 1) / blockDim.x, (blockDim.y + d_scene.height - 1) / blockDim.y, 1);
    gsplat_image_opacity_kernel<<<gridDim2, blockDim>>>(d_image, d_imgmask, d_scene);

    cudaFree(d_imgmask);
}

cudaError_t gsplat_data_host_to_device(SplatData &h_splats, SplatData &d_splats) {

    int nsplats = h_splats.nsplats;
    d_splats.nsplats = nsplats;
    CUDA_CHECK(cudaMalloc(&d_splats.pos3d, nsplats * sizeof(vec4f)));
    CUDA_CHECK(cudaMalloc(&d_splats.scale, nsplats * sizeof(vec4f)));
    CUDA_CHECK(cudaMalloc(&d_splats.rot, nsplats * sizeof(vec4f)));
    CUDA_CHECK(cudaMalloc(&d_splats.opacity, nsplats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_splats.sh, 48 * nsplats * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_splats.pos3d, h_splats.pos3d, nsplats * sizeof(vec4f), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_splats.scale, h_splats.scale, nsplats * sizeof(vec4f), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_splats.rot, h_splats.rot, nsplats * sizeof(vec4f), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_splats.opacity, h_splats.opacity, nsplats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_splats.sh, h_splats.sh, 48 * nsplats * sizeof(float), cudaMemcpyHostToDevice));
    return cudaSuccess;
}

cudaError_t gsplat_data_free_at_device(SplatData &d_splats) {

    CUDA_CHECK(cudaFree(d_splats.pos3d));
    CUDA_CHECK(cudaFree(d_splats.scale));
    CUDA_CHECK(cudaFree(d_splats.rot));
    CUDA_CHECK(cudaFree(d_splats.opacity));
    CUDA_CHECK(cudaFree(d_splats.sh));
    return cudaSuccess;
}

cudaError_t gsplat_data2d_create_at_device(int nsplats, SplatData2D &d_splat2d) {
    d_splat2d.nsplats = nsplats;
    CUDA_CHECK(cudaMalloc(&d_splat2d.pos2d, nsplats * sizeof(vec2f)));
    CUDA_CHECK(cudaMalloc(&d_splat2d.depth, nsplats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_splat2d.cov2d, nsplats * sizeof(mat2x2f)));
    CUDA_CHECK(cudaMalloc(&d_splat2d.rect, nsplats * sizeof(vec4f)));
    CUDA_CHECK(cudaMalloc(&d_splat2d.color, nsplats * sizeof(vec4f)));
    return cudaSuccess;
}

cudaError_t gsplat_data2d_free_at_device(SplatData2D &d_splats) {
    CUDA_CHECK(cudaFree(d_splats.pos2d));
    CUDA_CHECK(cudaFree(d_splats.depth));
    CUDA_CHECK(cudaFree(d_splats.cov2d));
    CUDA_CHECK(cudaFree(d_splats.rect));
    CUDA_CHECK(cudaFree(d_splats.color));
    return cudaSuccess;
}

cudaError_t gsplat_render_to_image_at_device(SplatScene &scene, SplatData &d_splats, SplatImage &d_image_out) {

    SplatData2D d_splats2d;
    CUDA_CHECK(gsplat_data2d_create_at_device(d_splats.nsplats, d_splats2d));

    CUDA_CHECK(gsplat_compute_at_device(scene, d_splats, d_splats2d));

    SplatTileInfo d_tileinfo;
    CUDA_CHECK(gsplat_tile_create_at_device(scene, d_splats2d, d_tileinfo));

    CUDA_CHECK(gsplat_data2d_sort_at_device(d_splats2d, d_tileinfo));

    CUDA_CHECK(SplatImage::createAtDevice(&d_image_out, scene.width, scene.height));

    CUDA_CHECK(splat_image_set_color_at_device(d_image_out, scene.bgcolor));

    CUDA_CHECK(gsplat_image_render_at_device(scene, d_splats2d, d_tileinfo, d_image_out));

    CUDA_CHECK(gsplat_data2d_free_at_device(d_splats2d));

    CUDA_CHECK(gsplat_tileinfo_free_at_device(d_tileinfo));

    return cudaSuccess;
}

cudaError_t gsplat_rasterize_gpu(SplatScene &scene, SplatData &h_splats, float **out_image, DebugPrintFn debug_print) {

    CUDATRACE("cudaWarmUp", cuda::cudaWarmUp());

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    SplatData d_splats;
    CUDA_CHECK(gsplat_data_host_to_device(h_splats, d_splats));

    SplatImage d_image;
    CUDA_CHECK(gsplat_render_to_image_at_device(scene, d_splats, d_image));

    CUDA_CHECK(d_image.toHost(out_image));

    CUDA_CHECK(gsplat_data_free_at_device(d_splats));

    CUDA_CHECK(d_image.freeAtDevice());

    return cudaSuccess;
}

} // namespace gsplat