#version 430 core

layout(local_size_x = 256) in;

struct Splat {
    vec4 ndcpos;
    vec4 color;
    vec4 rect;
    mat2 cov2d;
};

layout(std430, binding = 0) buffer SplatBuffer {
    Splat splats[];
};

uniform uint sort_count;
uniform uint k;
uniform uint j;

float splatDepth(uint index)
{
    return splats[index].ndcpos.z;
}

void swapSplats(uint a, uint b)
{
    Splat tmp = splats[a];
    splats[a] = splats[b];
    splats[b] = tmp;
}

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= sort_count) {
        return;
    }

    uint partner = i ^ j;
    if (partner <= i || partner >= sort_count) {
        return;
    }

    bool descending = (i & k) == 0u;
    float depth_i = splatDepth(i);
    float depth_partner = splatDepth(partner);

    bool should_swap = descending
        ? depth_i < depth_partner : depth_i > depth_partner;

    if (should_swap) {
        swapSplats(i, partner);
    }
}
