#include "cuda_utils.cuh"
#include "gsplat_rasterizer.hpp"
#include "matrix_gpu.hpp"
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <vector_types.h>
#include "gsplat_tile.cuh"

using namespace gsplat::matrix::gpu;

namespace gsplat {

void showCompileTile() {
    const char *_date = __DATE__;
    const char *_time = __TIME__;
    fprintf(stdout, "gsplatlab compile time: %s %s\n", _date, _time);
}

__global__ void image_set_color_kernel(vec4f *d_image, SplatSceneGPU d_scene) {

    int wid = blockDim.x * blockIdx.x + threadIdx.x;
    int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (wid < d_scene.width && hid < d_scene.height) {
        int idx = hid * d_scene.width + wid;
        d_image[idx] = d_scene.bgcolor;
    }
}

void image_set_color_gpu(vec4f *d_image, SplatSceneGPU d_scene) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((blockDim.x + d_scene.width - 1) / blockDim.x, (blockDim.y + d_scene.height - 1) / blockDim.y, 1);
    image_set_color_kernel<<<gridDim, blockDim, 0>>>(d_image, d_scene);
}

__device__ mat2x3f compute_jacobian(vec4f &viewpos, vec2f &viewport, float fovy) {
    float x = viewpos.x;
    float y = viewpos.y;
    float z = min(-0.05, viewpos.z);
    float z2 = z * z;
    float height = viewport.y;
    float focal = height / (2.0 * tan(fovy * 0.5));

    return mat2x3f(focal / z, 0.0, 0.0 - focal * x / z2, 0.0, focal / z, 0.0 - focal * y / z2);
}

__device__ mat3x3f quat_to_mat3(vec4f q) {
    vec4f nq = normalize(q);

    float x = nq.x;
    float y = nq.y;
    float z = nq.z;
    float w = nq.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    float xy = x * y;
    float xz = x * z;
    float yz = y * z;

    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return mat3x3f(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),
                   2.0 * (yz - wx), 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));
}

__device__ float clamp(float v, float a, float b) {
    v = v < a ? a : v;
    v = v > b ? b : v;
    return v;
}

__device__ vec4f compute_rect(vec2f pixpos, mat2x2f cov2d) {
    float u = pixpos.x;
    float v = pixpos.y;
    float sxx = cov2d[0][0];
    float syy = cov2d[1][1];
    float sxy = cov2d[0][1];
    float trace = sxx + syy;
    float det = max(sxx * syy - sxy * sxy, 1e-6);
    float lambda_max = 0.5 * (trace + sqrt(max(0.0, trace * trace - 4.0 * det)));
    float r = clamp(3.0 * sqrt(lambda_max), 1.0, 1024.0);
    float xmin = u - r;
    float xmax = u + r;
    float ymin = v - r;
    float ymax = v + r;
    return vec4f(xmin, ymin, xmax, ymax);
}

__device__ vec3f compute_color(vec3f *sh, vec3f d) {

    float x = d.x;
    float y = d.y;
    float z = d.z;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    const float SH_C0 = 0.28209479177387814;
    const float SH_C1 = 0.4886025119029199;
    const float SH_C2[5]{1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
                         0.5462742152960396};
    const float SH_C3[7]{-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                         -0.4570457994644658, 1.445305721320277, -0.5900435899266435};

    vec3f ret = SH_C0 * sh[0];

    // Degree 1
    ret += SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

    // Degree 2
    ret += SH_C2[0] * xy * sh[4];
    ret += SH_C2[1] * yz * sh[5];
    ret += SH_C2[2] * (2.0 * z2 - x2 - y2) * sh[6];
    ret += SH_C2[3] * xz * sh[7];
    ret += SH_C2[4] * (x2 - y2) * sh[8];

    // Degree 3
    ret += SH_C3[0] * y * (3.0 * x2 - y2) * sh[9];
    ret += SH_C3[1] * xy * z * sh[10];
    ret += SH_C3[2] * y * (4.0 * z2 - x2 - y2) * sh[11];
    ret += SH_C3[3] * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2) * sh[12];
    ret += SH_C3[4] * x * (4.0 * z2 - x2 - y2) * sh[13];
    ret += SH_C3[5] * z * (x2 - y2) * sh[14];
    ret += SH_C3[6] * x * (x2 - 3.0 * y2) * sh[15];

    return max(ret + 0.5, vec3f(0, 0, 0));
}

__device__ float sigmoid(float x) { return 1.0f / (1 + exp(-x)); }

__global__ void gsplat_compute_kernel(SplatSceneGPU d_scene, SplatDataGPU d_splats, SplatData2DGPU d_splats2d) {
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

void gsplat_compute_gpu(SplatSceneGPU d_scene, SplatDataGPU d_splats, SplatData2DGPU d_splats2d) {
    int blockDim = 128;
    int gridDim = (blockDim + d_splats.nsplats - 1) / blockDim;
    gsplat_compute_kernel<<<gridDim, blockDim>>>(d_scene, d_splats, d_splats2d);
}

__host__ __device__ float compute_weight(vec2f d, mat2x2f Cov2D) {
    float a = Cov2D[0][0];
    float b = Cov2D[1][0];
    float c = Cov2D[1][1];
    float s = max(a * c - b * b, 1e-6);
    float r2 = (c * d.x * d.x - 2.0 * b * d.x * d.y + a * d.y * d.y) / s;
    float weight = exp(-0.5 * r2);
    return weight;
}

__device__ bool point_in_box(vec2f v, vec4f box) {
    return v.x >= box.x && v.x <= box.z && v.y >= box.y && v.y <= box.w;
}

__global__ void gsplat_pixel_splat_count_kernel(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d,
                                                int *out_pixSplatCnt) {

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= d_splats2d.nsplats) {
        return;
    }

    vec4f rect = d_splats2d.rect[id];

    int xmin = (int)floor(rect.x);
    int ymin = (int)floor(rect.y);
    int xmax = (int)ceil(rect.z);
    int ymax = (int)ceil(rect.w);
    if (xmin >= d_scene.width || xmax < 0 || ymin >= d_scene.height || ymax < 0) {
        return;
    }

    vec2f centerpos = d_splats2d.pos2d[id];
    mat2x2f cov2d = d_splats2d.cov2d[id];
    vec2f pixpos;
    vec2f d;
    float w;

    for (int x = xmin; x <= xmax; ++x) {
        for (int y = ymin; y <= ymax; ++y) {
            pixpos = vec2f(x, y);
            d = pixpos - centerpos;
            w = compute_weight(d, cov2d);
            if (w >= 0.001) {
                atomicAdd(out_pixSplatCnt + y * d_scene.width + x, 1);
            }
        }
    }
}

void gsplat_pixel_splat_count_gpu(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d, int **out_pixSplatCnt) {
    int blockDim = 256;
    int gridDim = (blockDim + d_splats2d.nsplats - 1) / blockDim;

    cudaMalloc(out_pixSplatCnt, d_scene.width * d_scene.height * sizeof(int));
    cudaMemset(*out_pixSplatCnt, 0, d_scene.width * d_scene.height * sizeof(int));

    gsplat_pixel_splat_count_kernel<<<gridDim, blockDim>>>(d_scene, d_splats2d, *out_pixSplatCnt);
}

__device__ __forceinline__ int clamp(int v, int a, int b) {
    v = min(max(v, a), b);
    return v;
}

__device__ Box2D<int> gsplat_tile_box(SplatSceneGPU d_scene, SplatTileInfo d_tileinfo, int tileX, int tileY) {

    int xmin = tileX * d_tileinfo.size.x;
    int xmax = (tileX + 1) * d_tileinfo.size.x - 1;
    int ymin = tileY * d_tileinfo.size.y;
    int ymax = (tileY + 1) * d_tileinfo.size.y - 1;
    xmin = clamp(xmin, 0, d_scene.width - 1);
    xmax = clamp(xmax, 0, d_scene.width - 1);
    ymin = clamp(ymin, 0, d_scene.height - 1);
    ymax = clamp(ymax, 0, d_scene.height - 1);
    return {{xmin, xmax}, {ymin, ymax}};
}

template <typename T> __device__ __forceinline__ bool range_overlap(Range<T> r0, Range<T> r1) {
    return !(r0.max < r1.min || r0.min > r1.max);
}

template <typename T> __device__ __forceinline__ bool box_overlap(Box2D<T> a, Box2D<T> b) {
    return range_overlap(a.x, b.x) && range_overlap(a.y, b.y);
}

__global__ void gsplat_tile_splat_count_kernel(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d,
                                               SplatTileInfo d_tileinfo) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= d_splats2d.nsplats) {
        return;
    }

    vec4f rect = d_splats2d.rect[id];
    Box2D<int> splatBox{(int)floor(rect.x), (int)ceil(rect.z), (int)floor(rect.y), (int)ceil(rect.w)};
    Box2D<int> tileBox;
    for (int tileX = 0; tileX < d_tileinfo.nums.x; ++tileX) {
        for (int tileY = 0; tileY < d_tileinfo.nums.y; ++tileY) {
            tileBox = gsplat_tile_box(d_scene, d_tileinfo, tileX, tileY);
            if (box_overlap(splatBox, tileBox)) {
                atomicAdd(d_tileinfo.index.stride + tileY * d_tileinfo.nums.x + tileX, 1);
            }
        }
    }
}

__global__ void gsplat_tile_splat_index_kernel(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d,
                                               SplatTileInfo d_tileinfo, int *tileK) {

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= d_splats2d.nsplats) {
        return;
    }

    vec4f rect = d_splats2d.rect[id];
    Box2D<int> splatBox{(int)floor(rect.x), (int)ceil(rect.z), (int)floor(rect.y), (int)ceil(rect.w)};
    Box2D<int> tileBox;
    for (int tileX = 0; tileX < d_tileinfo.nums.x; ++tileX) {
        for (int tileY = 0; tileY < d_tileinfo.nums.y; ++tileY) {
            int tileId = tileY * d_tileinfo.nums.x + tileX;
            tileBox = gsplat_tile_box(d_scene, d_tileinfo, tileX, tileY);
            if (box_overlap(splatBox, tileBox)) {
                int k = atomicAdd(&tileK[tileId], 1);
                if (d_tileinfo.index.offset[tileId] + k < d_tileinfo.index.nEntries) {
                    d_tileinfo.index.entries[d_tileinfo.index.offset[tileId] + k] = id;
                }
            }
        }
    }
}

SplatTileInfo gsplat_tile_info_create(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d) {

    SplatTileInfo tileinfo;
    tileinfo.size.x = 64;
    tileinfo.size.y = 64;
    tileinfo.nums.x = (tileinfo.size.x + d_scene.width - 1) / tileinfo.size.x;
    tileinfo.nums.y = (tileinfo.size.y + d_scene.height - 1) / tileinfo.size.y;
    int totTileCnt = tileinfo.nums.x * tileinfo.nums.y;

    int splatBlockDim = 256;
    int splatGridDim = (splatBlockDim + d_splats2d.nsplats - 1) / splatBlockDim;

    cudaMalloc(&tileinfo.index.stride, totTileCnt * sizeof(int));
    cudaMemset(tileinfo.index.stride, 0, totTileCnt * sizeof(int));

    gsplat_tile_splat_count_kernel<<<splatGridDim, splatBlockDim>>>(d_scene, d_splats2d, tileinfo);

    // TODO compute nEntries and offset in gpu
    int *idxcnt_cpu = (int *)std::malloc(totTileCnt * sizeof(int));
    int *idxoff_cpu = (int *)std::malloc(totTileCnt * sizeof(int));
    memset(idxoff_cpu, 0, totTileCnt * sizeof(int));
    cudaMemcpy(idxcnt_cpu, tileinfo.index.stride, totTileCnt * sizeof(int), cudaMemcpyDeviceToHost);

    int nEntries = 0;
    // 计算tile-splat索引里面，每个tile的指针偏移
    for (int x = 0; x < tileinfo.nums.x; ++x) {
        for (int y = 0; y < tileinfo.nums.y; ++y) {
            int k = y * tileinfo.nums.x + x;
            nEntries += idxcnt_cpu[k];
        }
    }
    int cumsum = 0;
    for (int k = 0; k < totTileCnt; ++k) {
        idxoff_cpu[k] = cumsum;
        cumsum += idxcnt_cpu[k];
    }
    tileinfo.index.nEntries = nEntries;

    cudaMalloc(&tileinfo.index.offset, totTileCnt * sizeof(int));
    cudaMemcpy(tileinfo.index.offset, idxoff_cpu, totTileCnt * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&tileinfo.index.entries, nEntries * sizeof(int));

    int *tileK;
    cudaMalloc(&tileK, totTileCnt * sizeof(int));
    cudaMemset(tileK, 0, totTileCnt * sizeof(int));

    gsplat_tile_splat_index_kernel<<<splatGridDim, splatBlockDim>>>(d_scene, d_splats2d, tileinfo, tileK);

    free(idxcnt_cpu);
    free(idxoff_cpu);
    cudaFree(tileK);

    return tileinfo;
}

void gsplat_tile_info_free(SplatTileInfo d_tileinfo) {
    cudaFree(d_tileinfo.index.offset);
    cudaFree(d_tileinfo.index.stride);
    cudaFree(d_tileinfo.index.entries);
}

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

__global__ void gsplat_tile_bitonic_sort(SplatData2DGPU splat2d, int *d_entries, int n, int k, int j, bool asc) {

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

void gsplat_tile_splat_sort(SplatData2DGPU splats2d, SplatTileInfo h_tileinfo, SplatTileInfo d_tileinfo, int tileId,
                            int *d_cache) {

    int n = h_tileinfo.index.stride[tileId];
    if (n < 2)
        return;

    int m = bitCeil(n);

    int blockDim = 256;
    int blockNum = (blockDim + m - 1) / blockDim;

    int s = h_tileinfo.index.offset[tileId];

    fillInt32Kernel<<<blockNum, blockDim>>>(d_cache, -1, m);
    cudaMemcpy(d_cache, d_tileinfo.index.entries + s, sizeof(int) * n, cudaMemcpyDeviceToDevice);

    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            gsplat_tile_bitonic_sort<<<blockNum, blockDim>>>(splats2d, d_cache, m, k, j, true);
        }
    }

    cudaMemcpy(d_tileinfo.index.entries + s, d_cache, sizeof(int) * n, cudaMemcpyDeviceToDevice);
}

void gsplat_splat_sort(SplatData2DGPU splats2d, SplatTileInfo d_tileinfo) {

    SplatTileInfo h_tileinfo = splat_tileinfo_to_host(d_tileinfo);

    size_t maxTileSplatCount = getTileMaxNumSplats(h_tileinfo);
    size_t m = bitCeil(maxTileSplatCount);

    int *d_cache;
    cudaMalloc(&d_cache, m * sizeof(int));

    for (int tileId = 0; tileId < h_tileinfo.nums.x * h_tileinfo.nums.y; ++tileId) {
        gsplat_tile_splat_sort(splats2d, h_tileinfo, d_tileinfo, tileId, d_cache);
    }

    cudaFree(d_cache);
    free(h_tileinfo.index.offset);
    free(h_tileinfo.index.stride);
    free(h_tileinfo.index.entries);
}

__global__ void gsplat_rasterize_kernel(SplatSceneGPU scene, SplatData2DGPU splats2d, float *out_image) {

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

__global__ void gsplat_tile_paint_kernel(SplatSceneGPU scene, SplatData2DGPU splats2d, SplatTileInfo tileinfo,
                                         bool *imgmask, vec4f *image) {
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

__global__ void gsplat_image_opacity_kernel(vec4f *d_image, bool *d_imgmask, SplatSceneGPU d_scene) {

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

void gsplat_tile_paint(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d, SplatTileInfo d_tileinfo, vec4f *d_image) {
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

SplatDataGPU gsplat_data_gpu_create(SplatData &h_splats) {

    SplatDataGPU splats_gpu;
    int nsplats = h_splats.nsplats;
    splats_gpu.nsplats = nsplats;
    cudaMalloc(&splats_gpu.pos3d, nsplats * sizeof(vec4f));
    cudaMalloc(&splats_gpu.scale, nsplats * sizeof(vec4f));
    cudaMalloc(&splats_gpu.rot, nsplats * sizeof(vec4f));
    cudaMalloc(&splats_gpu.opacity, nsplats * sizeof(float));
    cudaMalloc(&splats_gpu.sh, 48 * nsplats * sizeof(float));

    cudaMemcpy(splats_gpu.pos3d, h_splats.pos3d, nsplats * sizeof(vec4f), cudaMemcpyHostToDevice);
    cudaMemcpy(splats_gpu.scale, h_splats.scale, nsplats * sizeof(vec4f), cudaMemcpyHostToDevice);
    cudaMemcpy(splats_gpu.rot, h_splats.rot, nsplats * sizeof(vec4f), cudaMemcpyHostToDevice);
    cudaMemcpy(splats_gpu.opacity, h_splats.opacity, nsplats * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(splats_gpu.sh, h_splats.sh, 48 * nsplats * sizeof(float), cudaMemcpyHostToDevice);
    return splats_gpu;
}

void gsplat_data_gpu_free(SplatDataGPU &d_splats) {

    cudaFree(d_splats.pos3d);
    cudaFree(d_splats.scale);
    cudaFree(d_splats.rot);
    cudaFree(d_splats.opacity);
    cudaFree(d_splats.sh);
}

SplatData2DGPU gsplat_data_2d_gpu_create(int nsplats) {
    SplatData2DGPU splats2d;
    splats2d.nsplats = nsplats;
    cudaMalloc(&splats2d.pos2d, nsplats * sizeof(vec2f));
    cudaMalloc(&splats2d.depth, nsplats * sizeof(float));
    cudaMalloc(&splats2d.cov2d, nsplats * sizeof(mat2x2f));
    cudaMalloc(&splats2d.rect, nsplats * sizeof(vec4f));
    cudaMalloc(&splats2d.color, nsplats * sizeof(vec4f));
    return splats2d;
}

void gsplat_data_2d_gpu_free(SplatData2DGPU &d_splats) {
    cudaFree(d_splats.pos2d);
    cudaFree(d_splats.depth);
    cudaFree(d_splats.cov2d);
    cudaFree(d_splats.rect);
    cudaFree(d_splats.color);
}

void gsplat_rasterize_gpu(SplatScene &scene, SplatData &splats, float **out_image, DebugPrintFn debug_print) {

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    SplatSceneGPU d_scene{mat4x4f(scene.viewmtx),
                          mat4x4f(scene.projmtx),
                          vec4f(scene.bgcolor),
                          vec3f(scene.eye),
                          scene.znear,
                          scene.zfar,
                          scene.fovy,
                          scene.width,
                          scene.height};

    SplatDataGPU d_splats = gsplat_data_gpu_create(splats);

    int nsplats = d_splats.nsplats;

    SplatData2DGPU d_splats2d = gsplat_data_2d_gpu_create(nsplats);

    gsplat_compute_gpu(d_scene, d_splats, d_splats2d);
    CUDA_CHECK_LAST_ERROR();

    SplatTileInfo d_tileinfo = gsplat_tile_info_create(d_scene, d_splats2d);

    // sort for each tile
    gsplat_splat_sort(d_splats2d, d_tileinfo);

    // image
    size_t imageByteSize = scene.width * scene.height * sizeof(vec4f);
    vec4f *d_image;
    cudaMalloc(&d_image, imageByteSize);

    debug_print("scene bgcolor=(%f,%f,%f,%f)\n", d_scene.bgcolor.x, d_scene.bgcolor.y, d_scene.bgcolor.z,
                d_scene.bgcolor.w);

    image_set_color_gpu(d_image, d_scene);
    CUDA_CHECK_LAST_ERROR();

    gsplat_tile_paint2(d_scene, d_splats2d, d_tileinfo, d_image);
    // gsplat_tile_paint(d_scene, d_splats2d, d_tileinfo, d_image);
    CUDA_CHECK_LAST_ERROR();

    // image output
    float *h_image = (float *)malloc(imageByteSize);
    if (h_image) {
        cudaMemcpy(h_image, d_image, imageByteSize, cudaMemcpyDeviceToHost);
    }

    gsplat_data_gpu_free(d_splats);
    gsplat_data_2d_gpu_free(d_splats2d);
    gsplat_tile_info_free(d_tileinfo);
    cudaFree(d_image);

    *out_image = h_image;
}

} // namespace gsplat