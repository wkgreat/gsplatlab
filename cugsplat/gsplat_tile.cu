#include "gsplat_rasterizer.hpp"
#include <cub/cub.cuh>
#include <driver_types.h>
#include "gsplat_rasterizer.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <cub/device/device_segmented_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include "gsplat_tile.cuh"

namespace gsplat {

size_t getTileMaxNumSplats(SplatTileInfo h_tileInfo) {

    int maxStride = 0;

    int nTiles = h_tileInfo.nums.x * h_tileInfo.nums.y;

    for (int i = 0; i < nTiles; ++i) {
        if (h_tileInfo.index.stride[i] > maxStride) {
            maxStride = h_tileInfo.index.stride[i];
        }
    }

    return maxStride;
}

SplatTileInfo splat_tileinfo_to_host(SplatTileInfo d_tileinfo) {

    SplatTileInfo h_tileinfo;

    h_tileinfo.size = d_tileinfo.size;
    h_tileinfo.nums = d_tileinfo.nums;
    size_t byteSize = d_tileinfo.nums.x * d_tileinfo.nums.y * sizeof(int);
    h_tileinfo.index.offset = (int *)malloc(byteSize);
    h_tileinfo.index.stride = (int *)malloc(byteSize);
    cudaMemcpy(h_tileinfo.index.offset, d_tileinfo.index.offset, byteSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tileinfo.index.stride, d_tileinfo.index.stride, byteSize, cudaMemcpyDeviceToHost);
    h_tileinfo.index.nEntries = d_tileinfo.index.nEntries;
    size_t entryByteSize = h_tileinfo.index.nEntries * sizeof(int);
    h_tileinfo.index.entries = (int *)malloc(entryByteSize);
    cudaMemcpy(h_tileinfo.index.entries, d_tileinfo.index.entries, entryByteSize, cudaMemcpyDeviceToHost);

    return h_tileinfo;
}

__device__ inline vec2i gsplatTileLocalPixIdxToGlobalPixIdx(SplatTileInfo d_tileinfo, vec2i tileIdx, vec2i pixIdx) {
    int gx = tileIdx.x * d_tileinfo.size.x + pixIdx.x;
    int gy = tileIdx.y * d_tileinfo.size.y + pixIdx.y;
    return vec2i(gx, gy);
}

__device__ inline vec2f gsplatTileLocalPixelToScreenCoords(SplatTileInfo d_tileinfo, vec2i tileIdx, vec2i pixIdx) {
    vec2i gidx = gsplatTileLocalPixIdxToGlobalPixIdx(d_tileinfo, tileIdx, pixIdx);
    return vec2f(gidx.x + 0.5, gidx.y + 0.5);
}

__global__ void gsplatTileAlphaCacheKernel(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d, SplatTileInfo d_tileinfo,
                                           vec2i tileIdx, int stride, int tmns, vec4f *d_cacheColor,
                                           float *d_cacheTrans) {

    int pidx = blockDim.x * blockIdx.x + threadIdx.x;    // x of pixel on tile
    int pidy = blockDim.y * blockIdx.y + threadIdx.y;    // y of pixel on tile
    int enid = blockDim.z * blockIdx.z + threadIdx.z;    // index entry id on tile (enid-th splat on tile)
    int tid = tileIdx.y * d_tileinfo.nums.x + tileIdx.x; // tile 1d idx
    int cacheIdx = (pidy * d_tileinfo.size.x + pidx) * tmns + enid;

    if (pidx < d_tileinfo.size.x && pidy < d_tileinfo.size.y && enid < stride) {
        int toff = d_tileinfo.index.offset[tid];
        int tstr = d_tileinfo.index.stride[tid];
        vec2f p = gsplatTileLocalPixelToScreenCoords(d_tileinfo, tileIdx, vec2i(pidx, pidy));
        vec4f c;
        if (enid < tstr) {
            int spid = d_tileinfo.index.entries[toff + enid]; // splat id in d_splats2d
            vec2f mu = d_splats2d.pos2d[spid];
            mat2x2f cov2d = d_splats2d.cov2d[spid];
            vec4f rect = d_splats2d.rect[spid];
            vec2f dist = p - mu;
            bool inRect = p.x >= rect.x && p.x <= rect.z && p.y >= rect.y && p.y <= rect.w;
            float weight = inRect ? compute_weight(dist, cov2d) : 0.0f;
            c = d_splats2d.color[spid];
            if (weight >= 0.001) {
                c.w *= weight;
            } else {
                c.w = 0.0f;
            }
        } else {
            c = vec4f(0.0, 0.0, 0.0, 0.0);
        }
        d_cacheColor[cacheIdx] = c;
        d_cacheTrans[cacheIdx] = 1.0 - c.w;
    }
}

__global__ void fillTileSegmentInfoKernel(unsigned int *d_begin, unsigned int *d_end, unsigned int nsegs,
                                          unsigned int tmns, unsigned int stride) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < nsegs) {
        unsigned int begin = x * tmns;
        d_begin[x] = begin;
        d_end[x] = begin + stride;
    }
}

struct TileSegmentInfo {
    std::int64_t nsegs = 0;
    unsigned int *d_begin;
    unsigned int *d_end;
};

struct TileSegmentInfoCache {

    std::map<unsigned int, TileSegmentInfo> cache;

    TileSegmentInfo getTileSegment(SplatTileInfo &tileinfo, unsigned int tid, unsigned int tmns, unsigned int stride) {
        if (this->cache.find(tid) == this->cache.end()) {
            createTileSegment(tileinfo, tid, tmns, stride);
        }
        return this->cache[tid];
    }

    void createTileSegment(SplatTileInfo &tileinfo, unsigned int tid, unsigned int tmns, unsigned int stride) {
        cache[tid] = TileSegmentInfo();
        std::int64_t nsegs = tileinfo.size.x * tileinfo.size.y;
        cache[tid].nsegs = nsegs;
        cudaMalloc(&(cache[tid].d_begin), sizeof(size_t) * nsegs);
        cudaMalloc(&(cache[tid].d_end), sizeof(size_t) * nsegs);
        unsigned int blockDim = 512;
        unsigned int blockNum = (nsegs + blockDim - 1) / blockDim;
        fillTileSegmentInfoKernel<<<blockNum, blockDim>>>(cache[tid].d_begin, cache[tid].d_end, nsegs, tmns, stride);
    }

    void destroy() {
        for (auto &entry : this->cache) {
            cudaFree(entry.second.d_begin);
            cudaFree(entry.second.d_end);
        }
    }
};

struct MultiplyOp {
    __device__ float operator()(float a, float b) const { return a * b; }
};

cudaError_t gsplatTileTransCumProd(SplatTileInfo d_tileinfo, unsigned int tid, int stride, int tmns,
                                   float *d_cacheTrans, TileSegmentInfoCache &tileSegCache) {

    MultiplyOp mulop;

    auto seginfo = tileSegCache.getTileSegment(d_tileinfo, tid, tmns, stride);

    cudaError_t err;

    err = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(d_cacheTrans, d_cacheTrans, seginfo.d_begin, seginfo.d_end,
                                                           seginfo.nsegs, mulop, 1.0f);

    return err;
}

__global__ void gsplatTilePixelSplatColorKernel(SplatTileInfo d_tileinfo, vec2i tileIdx, vec4f *d_cacheColor,
                                                float *d_cacheTrans, int stride, int tmns) {
    int pidx = blockDim.x * blockIdx.x + threadIdx.x; // x of pixel on tile
    int pidy = blockDim.y * blockIdx.y + threadIdx.y; // y of pixel on tile
    int enid = blockDim.z * blockIdx.z + threadIdx.z; // index entry id on tile (enid-th splat on tile)
    int cacheIdx = (pidy * d_tileinfo.size.x + pidx) * tmns + enid;
    if (pidx < d_tileinfo.size.x && pidy < d_tileinfo.size.y && enid < stride) {
        vec4f color = d_cacheColor[cacheIdx];
        float alpha = color.w;
        float trans = d_cacheTrans[cacheIdx];
        color.x = color.x * alpha * trans;
        color.y = color.y * alpha * trans;
        color.z = color.z * alpha * trans;
        color.w = trans * alpha; // attention here
        d_cacheColor[cacheIdx] = color;
    }
}

struct ColorSumOp {
    __device__ __forceinline__ vec4f operator()(const vec4f &a, const vec4f &b) const { return a + b; }
};

cudaError_t gsplatTilePixelFinalColor(SplatTileInfo d_tileinfo, unsigned int tid, int stride, int tmns,
                                      vec4f *d_cacheColor, vec4f *d_tilecolor, TileSegmentInfoCache &tileSegCache) {

    ColorSumOp sumop;

    auto seginfo = tileSegCache.getTileSegment(d_tileinfo, tid, tmns, stride);

    vec4f initColor{0.0, 0.0, 0.0, 0.0};
    cudaError_t err = cub::DeviceSegmentedReduce::Reduce(d_cacheColor, d_tilecolor, seginfo.nsegs, seginfo.d_begin,
                                                         seginfo.d_end, sumop, initColor);

    return err;
}

__global__ void gsplatTileColor2ImageKernel(SplatSceneGPU d_scene, SplatTileInfo d_tileinfo, vec2i tileIdx,
                                            vec4f *d_tilecolor, vec4f *d_image) {

    unsigned int px = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px < d_tileinfo.size.x && py < d_tileinfo.size.y) {
        vec2i pixIdx(px, py);
        vec2i imgIdx = gsplatTileLocalPixIdxToGlobalPixIdx(d_tileinfo, tileIdx, pixIdx);
        if (imgIdx.x < d_scene.width && imgIdx.y < d_scene.height) {
            vec4f color = d_tilecolor[py * d_tileinfo.size.x + px];
            float trans = 1.0f - color.w;
            color.x += trans * d_scene.bgcolor.x;
            color.y += trans * d_scene.bgcolor.y;
            color.z += trans * d_scene.bgcolor.z;
            color.w = 1.0f;
            d_image[imgIdx.y * d_scene.width + imgIdx.x] = color;
        }
    }
}

__global__ void gsplatTileSingleColor2ImageKernel(SplatSceneGPU d_scene, SplatTileInfo d_tileinfo, vec2i tileIdx,
                                                  vec4f color, vec4f *d_image) {
    unsigned int px = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px < d_tileinfo.size.x && py < d_tileinfo.size.y) {
        vec2i pixIdx(px, py);
        vec2i imgIdx = gsplatTileLocalPixIdxToGlobalPixIdx(d_tileinfo, tileIdx, pixIdx);
        if (imgIdx.x < d_scene.width && imgIdx.y < d_scene.height) {
            d_image[imgIdx.y * d_scene.width + imgIdx.x] = color;
        }
    }
}

void gsplat_tile_paint2(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d, SplatTileInfo d_tileinfo, vec4f *d_image) {
    SplatTileInfo h_tileInfo = splat_tileinfo_to_host(d_tileinfo);
    int tmns = (int)getTileMaxNumSplats(h_tileInfo);
    size_t tnpx = (size_t)h_tileInfo.size.x * h_tileInfo.size.y;
    vec4f *d_cacheColor = nullptr; //(rgb,alpha)
    float *d_cacheTrans = nullptr; // transmission
    vec4f *d_tilecolor = nullptr;

    cudaError_t err = cudaMalloc(&d_cacheColor, sizeof(vec4f) * tnpx * tmns);
    if (err == cudaSuccess) {
        err = cudaMalloc(&d_cacheTrans, sizeof(float) * tnpx * tmns);
    }
    if (err == cudaSuccess) {
        err = cudaMalloc(&d_tilecolor, sizeof(vec4f) * tnpx);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc failed: %s, cache needs %.1f MiB\n", __FUNCTION__, cudaGetErrorString(err),
                (sizeof(vec4f) + sizeof(float)) * (double)tnpx * tmns / 1048576.0);
        cudaFree(d_cacheColor);
        cudaFree(d_cacheTrans);
        cudaFree(d_tilecolor);
        free(h_tileInfo.index.offset);
        free(h_tileInfo.index.stride);
        free(h_tileInfo.index.entries);
        return;
    }

    fprintf(stdout, "%s tmns=%d tnpx=%zu\n", __FUNCTION__, tmns, tnpx);

    dim3 tileDimSize(d_tileinfo.size.x, d_tileinfo.size.y, 1);
    dim3 tileBlockDim(16, 16, 1);
    dim3 tileBlockNum = cuda::BLOCKNUM(tileBlockDim, tileDimSize);

    TileSegmentInfoCache tileSegCache;

    for (int tileX = 0; tileX < d_tileinfo.nums.x; ++tileX) {
        for (int tileY = 0; tileY < d_tileinfo.nums.y; ++tileY) {
            vec2i tileIdx(tileX, tileY);
            int tid = tileY * d_tileinfo.nums.x + tileX;
            int stride = h_tileInfo.index.stride[tid];
            dim3 dimSize(d_tileinfo.size.x, d_tileinfo.size.y, stride);
            dim3 blockDim(8, 8, 8);
            dim3 blockNum = cuda::BLOCKNUM(blockDim, dimSize);

            if (stride > 0) {
                CUDATRACE("gsplatTileAlphaCacheKernel",
                          gsplatTileAlphaCacheKernel<<<blockNum, blockDim>>>(d_scene, d_splats2d, d_tileinfo, tileIdx,
                                                                             stride, tmns, d_cacheColor, d_cacheTrans));
                CUDATRACE("gsplatTileTransCumProd",
                          gsplatTileTransCumProd(d_tileinfo, tid, stride, tmns, d_cacheTrans, tileSegCache));
                CUDATRACE("gsplatTilePixelSplatColorKernel",
                          gsplatTilePixelSplatColorKernel<<<blockNum, blockDim>>>(d_tileinfo, tileIdx, d_cacheColor,
                                                                                  d_cacheTrans, stride, tmns));
                CUDATRACE(
                    "gsplatTilePixelFinalColor",
                    gsplatTilePixelFinalColor(d_tileinfo, tid, stride, tmns, d_cacheColor, d_tilecolor, tileSegCache));
                CUDATRACE("gsplatTileColor2ImageKernel", gsplatTileColor2ImageKernel<<<tileBlockNum, tileBlockDim>>>(
                                                             d_scene, d_tileinfo, tileIdx, d_tilecolor, d_image));
            } else {
                gsplatTileSingleColor2ImageKernel<<<tileBlockNum, tileBlockDim>>>(d_scene, d_tileinfo, tileIdx,
                                                                                  d_scene.bgcolor, d_image);
            }
        }
    }

    tileSegCache.destroy();
    cudaFree(d_tilecolor);
    cudaFree(d_cacheColor);
    cudaFree(d_cacheTrans);
    free(h_tileInfo.index.offset);
    free(h_tileInfo.index.stride);
    free(h_tileInfo.index.entries);
}

} // namespace gsplat