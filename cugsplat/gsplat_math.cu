// #include "gsplat_math.cuh"
// #include <cuda_device_runtime_api.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <driver_types.h>

// namespace gsplat {

// __host__ __device__ float clamp(float v, float a, float b) {
//     v = v < a ? a : v;
//     v = v > b ? b : v;
//     return v;
// }

// __host__ __device__ float sigmoid(float x) { return 1.0f / (1 + exp(-x)); }

// __host__ __device__ float compute_weight(vec2f d, mat2x2f Cov2D) {
//     float a = Cov2D[0][0];
//     float b = Cov2D[1][0];
//     float c = Cov2D[1][1];
//     float s = fmaxf(a * c - b * b, 1e-6f);
//     float r2 = (c * d.x * d.x - 2.0 * b * d.x * d.y + a * d.y * d.y) / s;
//     float weight = exp(-0.5 * r2);
//     return weight;
// }

// __host__ __device__ mat2x3f compute_jacobian(vec4f &viewpos, vec2f &viewport, float fovy) {
//     float x = viewpos.x;
//     float y = viewpos.y;
//     float z = min(-0.05, viewpos.z);
//     float z2 = z * z;
//     float height = viewport.y;
//     float focal = height / (2.0 * tan(fovy * 0.5));

//     return mat2x3f(focal / z, 0.0, 0.0 - focal * x / z2, 0.0, focal / z, 0.0 - focal * y / z2);
// }

// __host__ __device__ mat3x3f quat_to_mat3(vec4f q) {
//     vec4f nq = normalize(q);

//     float x = nq.x;
//     float y = nq.y;
//     float z = nq.z;
//     float w = nq.w;

//     float xx = x * x;
//     float yy = y * y;
//     float zz = z * z;

//     float xy = x * y;
//     float xz = x * z;
//     float yz = y * z;

//     float wx = w * x;
//     float wy = w * y;
//     float wz = w * z;

//     return mat3x3f(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),
//                    2.0 * (yz - wx), 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));
// }

// __host__ __device__ vec4f compute_rect(vec2f pixpos, mat2x2f cov2d) {
//     float u = pixpos.x;
//     float v = pixpos.y;
//     float sxx = cov2d[0][0];
//     float syy = cov2d[1][1];
//     float sxy = cov2d[0][1];
//     float trace = sxx + syy;
//     float det = fmaxf(sxx * syy - sxy * sxy, 1e-6f);
//     float lambda_max = 0.5f * (trace + sqrtf(fmaxf(0.0f, trace * trace - 4.0f * det)));
//     float r = clamp(3.0 * sqrt(lambda_max), 1.0, 1024.0);
//     float xmin = u - r;
//     float xmax = u + r;
//     float ymin = v - r;
//     float ymax = v + r;
//     return vec4f(xmin, ymin, xmax, ymax);
// }

// __host__ __device__ vec3f compute_color(vec3f *sh, vec3f d) {

//     float x = d.x;
//     float y = d.y;
//     float z = d.z;

//     float x2 = x * x;
//     float y2 = y * y;
//     float z2 = z * z;
//     float xy = x * y;
//     float yz = y * z;
//     float xz = x * z;

//     const float SH_C0 = 0.28209479177387814;
//     const float SH_C1 = 0.4886025119029199;
//     const float SH_C2[5]{1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
//                          0.5462742152960396};
//     const float SH_C3[7]{-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
//                          -0.4570457994644658, 1.445305721320277, -0.5900435899266435};

//     vec3f ret = SH_C0 * sh[0];

//     // Degree 1
//     ret += SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

//     // Degree 2
//     ret += SH_C2[0] * xy * sh[4];
//     ret += SH_C2[1] * yz * sh[5];
//     ret += SH_C2[2] * (2.0 * z2 - x2 - y2) * sh[6];
//     ret += SH_C2[3] * xz * sh[7];
//     ret += SH_C2[4] * (x2 - y2) * sh[8];

//     // Degree 3
//     ret += SH_C3[0] * y * (3.0 * x2 - y2) * sh[9];
//     ret += SH_C3[1] * xy * z * sh[10];
//     ret += SH_C3[2] * y * (4.0 * z2 - x2 - y2) * sh[11];
//     ret += SH_C3[3] * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2) * sh[12];
//     ret += SH_C3[4] * x * (4.0 * z2 - x2 - y2) * sh[13];
//     ret += SH_C3[5] * z * (x2 - y2) * sh[14];
//     ret += SH_C3[6] * x * (x2 - 3.0 * y2) * sh[15];

//     return vec3f_max(ret + 0.5, vec3f(0, 0, 0));
// }

// } // namespace gsplat