#pragma once

#include "gsplat_rasterizer.hpp"

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

SplatTileInfo splat_tileinfo_to_host(SplatTileInfo d_tileinfo);

size_t getTileMaxNumSplats(SplatTileInfo h_tileInfo);

cudaError_t gsplat_image_render_at_device(SplatScene d_scene, SplatData2D d_splats2d, SplatTileInfo d_tileinfo,
                                          SplatImage d_image);

} // namespace gsplat