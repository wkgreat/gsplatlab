#include "cub/device/device_reduce.cuh"
#include "gsplat_rasterizer.hpp"
#include <cub/cub.cuh>
#include <driver_types.h>

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

} // namespace gsplat