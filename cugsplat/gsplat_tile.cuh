#pragma once

#include "gsplat_rasterizer.hpp"
#include "cuda_utils.cuh"
#include "gsplat_math.cuh"

namespace gsplat {

struct SplatTileInfo {
    vec2i size;
    vec2i nums;
    struct {
        int nEntries;
        int *offset;
        int *stride;
        int *entries;
    } index;
};

__device__ __forceinline__ Box2D<int> gsplat_tile_box(SplatScene d_scene, SplatTileInfo d_tileinfo, int tileX,
                                                      int tileY) {

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

cudaError_t gsplat_tile_create_at_device(SplatScene d_scene, SplatData2D d_splats2d, SplatTileInfo &d_tileinfo);

cudaError_t gsplat_tileinfo_free_at_device(SplatTileInfo d_tileinfo);

cudaError_t gsplat_tileinfo_to_host(SplatTileInfo d_tileinfo, SplatTileInfo &h_tileinfo);

size_t getTileMaxNumSplats(SplatTileInfo h_tileInfo);

cudaError_t gsplat_image_render_at_device(SplatScene d_scene, SplatData2D d_splats2d, SplatTileInfo d_tileinfo,
                                          SplatImage d_image);

} // namespace gsplat