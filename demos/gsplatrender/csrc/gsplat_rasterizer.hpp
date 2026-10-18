#pragma once

#include "gsplat.hpp"
#include "matrix_gpu.hpp"

using namespace gsplat::matrix::gpu;

namespace gsplat {

struct SplatScene {
  float *viewmtx;
  float *projmtx;
  float *eye;
  float *bgcolor;
  float near;
  float far;
  float fovy;
  int width;
  int height;
};

struct SplatSceneGPU {
  mat4x4f viewmtx;
  mat4x4f projmtx;
  vec4f bgcolor;
  vec3f eye;
  float near;
  float far;
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

template <typename T> struct Range {
  T min, max;
};

template <typename T> struct Box2D {
  Range<T> x;
  Range<T> y;
};

void gsplat_rasterize_gpu(SplatScene &scene, SplatData &splats, float **image,
                          DebugPrintFn debug_print = nullptr);

} // namespace gsplat