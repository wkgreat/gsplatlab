#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

namespace gsplat {
namespace matrix {
namespace gpu {

struct alignas(8) vec2i {
    int x = 0, y = 0;
    __host__ __device__ vec2i() {};
    __host__ __device__ vec2i(int x, int y) : x(x), y(y) {};
    __host__ __device__ vec2i(const int *data) {
        x = data[0];
        y = data[1];
    }
    __host__ __device__ int &operator[](int i) { return (&x)[i]; }
    __host__ __device__ int operator[](int i) const { return (&x)[i]; }
};

struct alignas(8) vec2f {
    float x = 0.0, y = 0.0;
    __host__ __device__ vec2f() {};
    __host__ __device__ vec2f(float x, float y) : x(x), y(y) {};
    __host__ __device__ vec2f(vec2i v) : x((float)v.x), y((float)v.y) {};
    __host__ __device__ vec2f(const float *data) {
        x = data[0];
        y = data[1];
    }
    __host__ __device__ float &operator[](int i) { return (&x)[i]; }
    __host__ __device__ float operator[](int i) const { return (&x)[i]; }
};

struct vec3f {
    float x = 0.0, y = 0.0, z = 0.0;
    __host__ __device__ vec3f() {};
    __host__ __device__ vec3f(float x, float y, float z) : x(x), y(y), z(z) {};
    __host__ __device__ vec3f(const float *data) {
        x = data[0];
        y = data[1];
        z = data[2];
    }
    __host__ __device__ float &operator[](int i) { return (&x)[i]; }
    __host__ __device__ float operator[](int i) const { return (&x)[i]; }
    __host__ __device__ vec2f xy() const { return {x, y}; }
};

struct alignas(16) vec4f {
    float x, y, z, w;
    __host__ __device__ vec4f() {};
    __host__ __device__ vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};
    __host__ __device__ vec4f(const float *data) {
        x = data[0];
        y = data[1];
        z = data[2];
        w = data[3];
    }
    __host__ __device__ vec4f(const vec3f v, float w) {
        x = v.x;
        y = v.y;
        z = v.z;
        this->w = w;
    }
    __host__ __device__ float &operator[](int i) { return (&x)[i]; }
    __host__ __device__ float operator[](int i) const { return (&x)[i]; }
    __host__ __device__ vec4f operator+(const vec4f &a) const { return vec4f(a.x + x, a.y + y, a.z + z, a.w + w); }
    __host__ __device__ vec2f xy() { return {x, y}; }
    __host__ __device__ vec3f xyz() { return {x, y, z}; }
};

struct mat2x2f {
    vec2f r0;
    vec2f r1;
    __host__ __device__ mat2x2f() {};
    __host__ __device__ mat2x2f(float m00, float m01, float m10, float m11) {
        r0[0] = m00;
        r0[1] = m01;
        r1[0] = m10;
        r1[1] = m11;
    };
    __host__ __device__ mat2x2f(const float *data) {
        r0[0] = data[0];
        r0[1] = data[1];
        r1[0] = data[2];
        r1[1] = data[3];
    }
    __host__ __device__ vec2f &operator[](int i) { return (&r0)[i]; }
    __host__ __device__ vec2f operator[](int i) const { return (&r0)[i]; }
};

struct mat3x2f;

struct mat2x3f {
    vec3f r0;
    vec3f r1;
    __host__ __device__ mat2x3f() {};
    __host__ __device__ mat2x3f(const float *data) {
        r0[0] = data[0];
        r0[1] = data[1];
        r0[2] = data[2];
        r1[0] = data[3];
        r1[1] = data[4];
        r1[2] = data[5];
    }
    __host__ __device__ vec3f &operator[](int i) { return (&r0)[i]; }
    __host__ __device__ vec3f operator[](int i) const { return (&r0)[i]; }
    __host__ __device__ mat2x3f(float m00, float m01, float m02, float m10, float m11, float m12) {
        r0[0] = m00;
        r0[1] = m01;
        r0[2] = m02;
        r1[0] = m10;
        r1[1] = m11;
        r1[2] = m12;
    };
    __host__ __device__ mat3x2f T();
};

struct mat3x2f {
    vec2f r0;
    vec2f r1;
    vec2f r2;
    __host__ __device__ mat3x2f() {};
    __host__ __device__ mat3x2f(const float *data) {
        r0[0] = data[0];
        r0[1] = data[1];
        r1[0] = data[2];
        r1[1] = data[3];
        r2[0] = data[4];
        r2[1] = data[5];
    }
    __host__ __device__ mat3x2f(float m00, float m01, float m10, float m11, float m20, float m21) {
        r0[0] = m00;
        r0[1] = m01;
        r1[0] = m10;
        r1[1] = m11;
        r2[0] = m20;
        r2[1] = m21;
    };

    __host__ __device__ vec2f &operator[](int i) { return (&r0)[i]; }
    __host__ __device__ vec2f operator[](int i) const { return (&r0)[i]; }
    __host__ __device__ inline mat2x3f T() {

        mat2x3f t;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                t[j][i] = (*this)[i][j];
            }
        }
        return t;
    }
};

__host__ __device__ inline mat3x2f mat2x3f::T() { return mat3x2f(r0[0], r1[0], r0[1], r1[1], r0[2], r1[2]); }

struct mat3x3f {
    vec3f r0;
    vec3f r1;
    vec3f r2;
    __host__ __device__ mat3x3f() {};
    __host__ __device__ mat3x3f(vec3f r0, vec3f r1, vec3f r2) : r0(r0), r1(r1), r2(r2) {};
    __host__ __device__ mat3x3f(const float *data) {
        r0[0] = data[0];
        r0[1] = data[1];
        r0[2] = data[2];
        r1[0] = data[3];
        r1[1] = data[4];
        r1[2] = data[5];
        r2[0] = data[6];
        r2[1] = data[7];
        r2[2] = data[8];
    }
    __host__ __device__ mat3x3f(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21,
                                float m22) {
        r0[0] = m00;
        r0[1] = m01;
        r0[2] = m02;
        r1[0] = m10;
        r1[1] = m11;
        r1[2] = m12;
        r2[0] = m20;
        r2[1] = m21;
        r2[2] = m22;
    }
    __host__ __device__ vec3f &operator[](int i) { return (&r0)[i]; }
    __host__ __device__ vec3f operator[](int i) const { return (&r0)[i]; }

    __host__ __device__ inline mat3x3f T() {

        mat3x3f t;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                t[i][j] = (*this)[j][i];
            }
        }
        return t;
    }
};

struct mat4x4f {
    vec4f r0;
    vec4f r1;
    vec4f r2;
    vec4f r3;
    __host__ __device__ mat4x4f() {};
    __host__ __device__ mat4x4f(const float *data) {
        r0[0] = data[0];
        r0[1] = data[1];
        r0[2] = data[2];
        r0[3] = data[3];

        r1[0] = data[4];
        r1[1] = data[5];
        r1[2] = data[6];
        r1[3] = data[7];

        r2[0] = data[8];
        r2[1] = data[9];
        r2[2] = data[10];
        r2[3] = data[11];

        r3[0] = data[12];
        r3[1] = data[13];
        r3[2] = data[14];
        r3[3] = data[15];
    }
    __host__ __device__ vec4f &operator[](int i) { return (&r0)[i]; }
    __host__ __device__ vec4f operator[](int i) const { return (&r0)[i]; }
};

// add
__host__ __device__ inline vec2f operator+(const vec2f &a, const vec2f &b) { return {a.x + b.x, a.y + b.y}; }
__host__ __device__ inline vec2f operator+(const vec2f &a, const float b) { return {a.x + b, a.y + b}; }
__host__ __device__ inline vec2f operator+(const float b, const vec2f &a) { return a + b; }
__host__ __device__ inline vec3f operator+(const vec3f &a, const vec3f &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ inline vec3f operator+(const vec3f &a, const float b) { return {a.x + b, a.y + b, a.z + b}; }
__host__ __device__ inline vec3f operator+(const float b, const vec3f &a) { return a + b; }

__host__ __device__ inline void operator+=(vec3f &a, const vec3f &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__host__ __device__ inline void operator+=(vec3f &a, const float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

// substract
__host__ __device__ inline vec2f operator-(const vec2f &a, const vec2f &b) { return {a.x - b.x, a.y - b.y}; }
__host__ __device__ inline vec2f operator-(const vec2f &a, const float b) { return {a.x - b, a.y - b}; }
__host__ __device__ inline vec2f operator-(const float b, const vec2f &a) { return a - b; }
__host__ __device__ inline vec3f operator-(const vec3f &a, const vec3f &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ inline vec3f operator-(const vec3f &a, const float b) { return {a.x - b, a.y - b, a.z - b}; }
__host__ __device__ inline vec3f operator-(const float b, const vec3f &a) { return a - b; }

// multiply
__host__ __device__ inline vec2f operator*(const vec2f &a, const vec2f &b) { return {a.x * b.x, a.y * b.y}; }
__host__ __device__ inline vec2f operator*(const vec2f &a, const float b) { return {a.x * b, a.y * b}; }
__host__ __device__ inline vec2f operator*(const float b, const vec2f &a) { return a * b; }
__host__ __device__ inline vec3f operator*(const vec3f &a, const vec3f &b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
__host__ __device__ inline vec3f operator*(const vec3f &a, const float b) { return {a.x * b, a.y * b, a.z * b}; }
__host__ __device__ inline vec3f operator*(const float b, const vec3f &a) { return a * b; }

__host__ __device__ inline mat3x2f operator*(const mat3x3f &a, const mat3x2f &b) {
    return {a.r0.x * b.r0.x + a.r0.y * b.r1.x + a.r0.z * b.r2.x, a.r0.x * b.r0.y + a.r0.y * b.r1.y + a.r0.z * b.r2.y,
            a.r1.x * b.r0.x + a.r1.y * b.r1.x + a.r1.z * b.r2.x, a.r1.x * b.r0.y + a.r1.y * b.r1.y + a.r1.z * b.r2.y,
            a.r2.x * b.r0.x + a.r2.y * b.r1.x + a.r2.z * b.r2.x, a.r2.x * b.r0.y + a.r2.y * b.r1.y + a.r2.z * b.r2.y};
}
__host__ __device__ inline mat2x2f operator*(const mat2x3f &a, const mat3x2f &b) {
    return {a.r0.x * b.r0.x + a.r0.y * b.r1.x + a.r0.z * b.r2.x, a.r0.x * b.r0.y + a.r0.y * b.r1.y + a.r0.z * b.r2.y,
            a.r1.x * b.r0.x + a.r1.y * b.r1.x + a.r1.z * b.r2.x, a.r1.x * b.r0.y + a.r1.y * b.r1.y + a.r1.z * b.r2.y};
}

__host__ __device__ inline mat3x3f operator*(const mat3x3f &a, const mat3x3f &b) {
    mat3x3f r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return r;
}

__host__ __device__ inline vec4f operator*(const mat4x4f &m, const vec4f &v) {
    vec4f r;
    for (int i = 0; i < 4; ++i) {
        r[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            r[i] += m[i][j] * v[j];
        }
    }
    return r;
}

// division
__host__ __device__ inline vec3f operator/(const vec3f &v, const float w) { return {v.x / w, v.y / w, v.z / w}; }
__host__ __device__ inline vec4f operator/(const vec4f &v, const float w) {
    return {v.x / w, v.y / w, v.z / w, v.w / w};
}

// length
__host__ __device__ inline float length(const vec3f &v) {
    const float squared = v.x * v.x + v.y * v.y + v.z * v.z;
#ifdef __CUDA_ARCH__
    return sqrtf(squared);
#else
    return std::sqrt(squared);
#endif
}

__host__ __device__ inline float length(const vec4f &v) {
    const float squared = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#ifdef __CUDA_ARCH__
    return sqrtf(squared);
#else
    return std::sqrt(squared);
#endif
}

// normalize
__host__ __device__ inline vec3f normalize(const vec3f &v) { return v / length(v); }

__host__ __device__ inline vec4f normalize(const vec4f &v) { return v / length(v); }

// max
__host__ __device__ inline vec3f max(const vec3f &a, const vec3f &b) {
#ifdef __CUDA_ARCH__
    return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
#else
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
#endif
}

// print
// clang-format off
__host__ inline void print_vec4f(const vec4f &v) {
  printf("vec4f(\n%.4f, %.4f, %.4f, %.4f)\n", v[0], v[1], v[2], v[3]);
}

__host__ inline void print_mat4x4f(const mat4x4f &m) {
  printf("mat4x4f(\n"
         "%.4f, %.4f, %.4f, %.4f,\n"
         "%.4f, %.4f, %.4f, %.4f,\n"
         "%.4f, %.4f, %.4f, %.4f,\n"
         "%.4f, %.4f, %.4f, %.4f)\n",
         m[0][0], m[0][1], m[0][2], m[0][3],
         m[1][0], m[1][1], m[1][2], m[1][3],
         m[2][0], m[2][1], m[2][2], m[2][3],
         m[3][0], m[3][1], m[3][2], m[3][3]);
}

// clang-format on

} // namespace gpu
} // namespace matrix
} // namespace gsplat
