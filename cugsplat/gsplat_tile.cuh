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

void gsplat_tile_paint2(SplatSceneGPU d_scene, SplatData2DGPU d_splats2d, SplatTileInfo d_tileinfo, vec4f *d_image);

} // namespace gsplat