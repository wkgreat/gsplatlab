#pragma once

#include <cmath>

namespace gsplat {
namespace matrix {
namespace cpu {

template <typename T> struct vec3 {
  T x;
  T y;
  T z;
  const T &operator[](int i) const { return (&x)[i]; }
};

using vec3f = vec3<float>;
using vec3d = vec3<double>;

template <typename T> vec3<T> operator+(const vec3<T> &a, const vec3<T> &b) {
  vec3<T> r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  r.z = a.z + b.z;
  return r;
}

template <typename T> vec3<T> operator-(const vec3<T> &a, const vec3<T> &b) {
  vec3<T> r;
  r.x = a.x - b.x;
  r.y = a.y - b.y;
  r.z = a.z - b.z;
  return r;
}

template <typename T> T length(const vec3<T> &a) {
  T s = a.x * a.x + a.y * a.y + a.z * a.z;
  return std::sqrt(s);
}

template <typename T> vec3<T> normalize(const vec3<T> &a) {
  T len = length(a);
  vec3<T> r;
  r.x = a.x / len;
  r.y = a.y / len;
  r.z = a.z / len;
  return r;
}

template <typename T> const T dot(const vec3<T> &a, const vec3<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T> const vec3<T> cross(const vec3<T> &a, const vec3<T> &b) {
  vec3<T> c;
  c.x = a.y * b.z - a.z * b.y;
  c.y = a.z * b.x - a.x * b.z;
  c.z = a.x * b.y - a.y * b.x;
  return c;
}

template <typename T> struct mat3 {
  T m00, m01, m02;
  T m10, m11, m12;
  T m20, m21, m22;
  const T &operator[](int i) const { return (&m00)[i]; }
};

using mat3f = mat3<float>;
using mat3d = mat3<double>;

template <typename T> struct mat4 {
  T m00, m01, m02, m03;
  T m10, m11, m12, m13;
  T m20, m21, m22, m23;
  T m30, m31, m32, m33;
  const T &operator[](int i) const { return (&m00)[i]; }
};

using mat4f = mat4<float>;
using mat4d = mat4<double>;

template <typename T>
mat4<T> lookAt(const vec3<T> &eye, const vec3<T> &center, const vec3<T> up) {
  vec3<T> f = normalize(center - eye);
  vec3<T> s = normalize(cross(f, up));
  vec3<T> u = cross(s, f);
  mat4<T> m;
  m.m00 = s.x;
  m.m10 = s.y;
  m.m20 = s.z;

  m.m01 = u.x;
  m.m11 = u.y;
  m.m21 = u.z;

  m.m02 = -f.x;
  m.m12 = -f.y;
  m.m22 = -f.z;

  m.m30 = -dot(s, eye);
  m.m31 = -dot(u, eye);
  m.m32 = dot(f, eye);
  m.m03 = static_cast<T>(0);
  m.m13 = static_cast<T>(0);
  m.m23 = static_cast<T>(0);
  m.m33 = static_cast<T>(1);
  return m;
}

} // namespace cpu
} // namespace matrix
} // namespace gsplat
