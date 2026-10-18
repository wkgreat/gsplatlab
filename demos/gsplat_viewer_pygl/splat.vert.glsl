#version 430 core

struct Viewport {
    vec2 size;
};

uniform Viewport viewport;

struct Splat {
    vec4 ndcpos;
    vec4 color;
    vec4 rect; // xmin, ymin, xmax, ymax，单位为像素
    mat2 cov2d;
};

layout(std430, binding = 0) readonly buffer SplatBuffer {
    Splat splats[];
};

// 两个三角形组成一个包围高斯椭圆的 quad。
const vec2 QUAD[6] = vec2[6](
        vec2(-1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0)
    );

noperspective out vec2 v_distance;
flat out mat2 v_cov2d;
flat out vec4 v_color;

void main()
{
    Splat splat = splats[gl_InstanceID];
    vec2 q = QUAD[gl_VertexID];

    vec2 rect_min = splat.rect.xy;
    vec2 rect_max = splat.rect.zw;
    vec2 center_px = 0.5 * (rect_min + rect_max);
    vec2 half_extent_px = 0.5 * (rect_max - rect_min);
    vec2 pixel_pos = center_px + q * half_extent_px;

    // compute shader 的 rect 是像素坐标，这里转换回 OpenGL NDC。
    vec2 vertex_ndc = pixel_pos / viewport.size * 2.0 - 1.0;
    gl_Position = vec4(vertex_ndc, splat.ndcpos.z, 1.0);

    v_distance = pixel_pos - center_px;
    v_cov2d = splat.cov2d;
    v_color = splat.color;
}
