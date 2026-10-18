#version 430 core

layout(local_size_x = 256) in;

struct SplatInput {
    vec4 posopa; // vec3 pos and float opacity
    vec4 scale;
    vec4 rot;
    vec4 shcolor[16];
};

struct SplatOutput {
    vec4 ndcpos;
    vec4 color;
    vec4 rect;
    mat2 cov2d;
};

struct Camera {
    vec4 eye;
    mat4 viewmtx;
};

struct Projection {
    float fovy;
    mat4 projmtx;
};

struct Viewport {
    vec2 size;
};

layout(std430, binding = 0) readonly buffer SplatInputBuffer {
    SplatInput in_splats[];
};

layout(std430, binding = 1) writeonly buffer SplatOutputBuffer {
    SplatOutput out_splats[];
};

uniform Camera camera;
uniform Projection projection;
uniform Viewport viewport;
uniform uint nsplats;

mat3 quatToMat3(vec4 q)
{
    q = normalize(q);

    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    float xy = x * y;
    float xz = x * z;
    float yz = y * z;

    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return mat3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy),
        2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),
        2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)
    );
}

mat3x2 computeJacobian(vec3 viewpos) {
    float x = viewpos.x;
    float y = viewpos.y;
    float z = min(-0.05, viewpos.z);
    float z2 = z * z;
    float height = viewport.size[1];
    float fovy = projection.fovy;
    float focal = height / (2.0 * tan(fovy * 0.5));

    return mat3x2(
        vec2(focal / z, 0.0),
        vec2(0.0, focal / z),
        vec2(-focal * x / z2, -focal * y / z2)
    );
}

vec4 computeRect(vec2 pixelpos, mat2 cov2d) {
    float u = pixelpos.x;
    float v = pixelpos.y;
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
    return vec4(xmin, ymin, xmax, ymax);
}

vec3 sh2rgb(vec4 sh[16], vec3 direct) {
    float x = direct.x;
    float y = direct.y;
    float z = direct.z;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    // SH 常数定义
    const float SH_C0 = 0.28209479177387814;
    const float SH_C1 = 0.4886025119029199;
    const float[5] SH_C2 = float[5](
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        );
    const float[7] SH_C3 = float[7](
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        );

    // Degree 0 (基础色)
    vec3 result = SH_C0 * sh[0].rgb;

    // Degree 1
    result += SH_C1 * (-y * sh[1].rgb + z * sh[2].rgb - x * sh[3].rgb);

    // Degree 2
    result += SH_C2[0] * xy * sh[4].rgb;
    result += SH_C2[1] * yz * sh[5].rgb;
    result += SH_C2[2] * (2.0 * z2 - x2 - y2) * sh[6].rgb;
    result += SH_C2[3] * xz * sh[7].rgb;
    result += SH_C2[4] * (x2 - y2) * sh[8].rgb;

    // Degree 3
    result += SH_C3[0] * y * (3.0 * x2 - y2) * sh[9].rgb;
    result += SH_C3[1] * xy * z * sh[10].rgb;
    result += SH_C3[2] * y * (4.0 * z2 - x2 - y2) * sh[11].rgb;
    result += SH_C3[3] * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2) * sh[12].rgb;
    result += SH_C3[4] * x * (4.0 * z2 - x2 - y2) * sh[13].rgb;
    result += SH_C3[5] * z * (x2 - y2) * sh[14].rgb;
    result += SH_C3[6] * x * (x2 - 3.0 * y2) * sh[15].rgb;

    return max(result + 0.5, vec3(0.0));
}

float sigmoid(float x) {
    return 1.0f / (1 + exp(-x));
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= nsplats) {
        return;
    }
    SplatInput splat = in_splats[id];

    vec4 worldpos = vec4(splat.posopa.xyz, 1.0);
    vec4 viewpos = camera.viewmtx * worldpos;
    vec4 ndcpos = projection.projmtx * viewpos;
    ndcpos = ndcpos / ndcpos.w;

    vec2 pixelpos = (ndcpos.xy * 0.5 + 0.5)
            * viewport.size;

    mat3x2 J = computeJacobian(viewpos.xyz);
    mat3x3 S = mat3x3(
            exp(splat.scale.x), 0.0, 0.0,
            0.0, exp(splat.scale.y), 0.0,
            0.0, 0.0, exp(splat.scale.z)
        );

    mat3x3 R = quatToMat3(vec4(splat.rot.yzw, splat.rot.x));
    mat3x3 Cov3D = R * S * transpose(S) * transpose(R);
    mat3x3 viewmtx3 = mat3x3(
            camera.viewmtx[0].xyz,
            camera.viewmtx[1].xyz,
            camera.viewmtx[2].xyz
        );
    Cov3D = viewmtx3 * (Cov3D * transpose(viewmtx3));
    mat2x2 Cov2D = J * (Cov3D * transpose(J));

    // 与参考 3DGS rasterizer 一致的屏幕空间低通项：防止协方差
    // 退化成极细的针状椭圆，并保证后续求逆数值稳定。
    Cov2D[0][0] += 0.3;
    Cov2D[1][1] += 0.3;

    vec4 rect = computeRect(pixelpos, Cov2D);

    float opacity = sigmoid(splat.posopa.a);
    // SH 的观察方向定义为从相机指向 Gaussian。
    vec3 direct = normalize(worldpos.xyz - camera.eye.xyz);
    vec4 color = vec4(sh2rgb(splat.shcolor, direct), opacity);

    out_splats[id].ndcpos = ndcpos;
    out_splats[id].color = color;
    out_splats[id].rect = rect;
    out_splats[id].cov2d = Cov2D;
}
