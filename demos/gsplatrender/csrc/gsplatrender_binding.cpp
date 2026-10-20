#define PY_SSIZE_T_CLEAN
#include "gsplat_rasterizer.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *add_one(PyObject *self, PyObject *args) {

  PyObject *input_obj;

  if (!PyArg_ParseTuple(args, "O", &input_obj)) {
    return nullptr;
  }

  PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(input_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  if (!arr) {
    return nullptr;
  }

  float *data = static_cast<float *>(PyArray_DATA(arr));
  int ndim = PyArray_NDIM(arr);
  npy_intp *shape = PyArray_SHAPE(arr);
  npy_intp size = PyArray_SIZE(arr);
  float *outdata = static_cast<float *>(malloc(size));

  for (int i = 0; i < size; ++i) {
    outdata[i] = data[i] + 1;
  }

  PyObject *output_obj =
      PyArray_SimpleNewFromData(ndim, shape, NPY_FLOAT32, outdata);

  if (!output_obj) {
    free(outdata);
    Py_DECREF(arr);
    return nullptr;
  }

  PyObject *capsule = PyCapsule_New(outdata, nullptr, [](PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, nullptr);
    free(ptr);
  });

  if (!capsule) {
    free(outdata);
    Py_DECREF(arr);
    return nullptr;
  }

  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(output_obj), capsule);

  return output_obj;
}

static PyObject *gsplat_rasterize(PyObject *self, PyObject *args) {
  PyObject *scene = nullptr;
  PyObject *pos3d_obj = nullptr;
  PyObject *scale_obj = nullptr;
  PyObject *rot_obj = nullptr;
  PyObject *opacity_obj = nullptr;
  PyObject *sh_obj = nullptr;
  PyObject *ret_obj = nullptr;

  if (!PyArg_ParseTuple(args, "OOOOOO", &scene, &pos3d_obj, &scale_obj,
                        &rot_obj, &opacity_obj, &sh_obj)) {
    PySys_WriteStderr("PyArg_ParseTuple error\n");
    return nullptr;
  }

  PyObject *bgcolor_obj = PyObject_GetAttrString(scene, "bgcolor");
  if (!bgcolor_obj) {
    // 缺属性时 PyArray_FROM_OTF(NULL) 会直接崩，这里必须提前返回
    return nullptr;
  }
  PyArrayObject *bgcolor_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(bgcolor_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  PyObject *viewmtx_obj = PyObject_GetAttrString(scene, "viewmtx");
  PyArrayObject *viewmtx_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(viewmtx_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  PyObject *projmtx_obj = PyObject_GetAttrString(scene, "projmtx");
  PyArrayObject *projmtx_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(projmtx_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  PyObject *eye_obj = PyObject_GetAttrString(scene, "eye");
  PyObject *eye_x_obj = PyObject_GetAttrString(eye_obj, "x");
  PyObject *eye_y_obj = PyObject_GetAttrString(eye_obj, "y");
  PyObject *eye_z_obj = PyObject_GetAttrString(eye_obj, "z");
  float eye[3] = {static_cast<float>(PyFloat_AsDouble(eye_x_obj)),
                  static_cast<float>(PyFloat_AsDouble(eye_y_obj)),
                  static_cast<float>(PyFloat_AsDouble(eye_z_obj))};
  Py_XDECREF(eye_x_obj);
  Py_XDECREF(eye_y_obj);
  Py_XDECREF(eye_z_obj);
  Py_XDECREF(eye_obj);

  PyObject *near_obj = PyObject_GetAttrString(scene, "near");
  float znear = PyFloat_AsDouble(near_obj);
  PyObject *far_obj = PyObject_GetAttrString(scene, "far");
  float zfar = PyFloat_AsDouble(far_obj);
  PyObject *fovy_obj = PyObject_GetAttrString(scene, "fovy");
  float fovy = PyFloat_AsDouble(fovy_obj);
  PyObject *width_obj = PyObject_GetAttrString(scene, "width");
  int width = PyLong_AsLong(width_obj);
  PyObject *height_obj = PyObject_GetAttrString(scene, "height");
  int height = PyLong_AsLong(height_obj);

  int viewmtx_ndim = PyArray_NDIM(viewmtx_arr);
  npy_intp *viewmtx_shape = PyArray_SHAPE(viewmtx_arr);

  PySys_WriteStdout("Scene, viewmtx ndim=%d shape=(%d, %d)\n", viewmtx_ndim,
                    viewmtx_shape[0], viewmtx_shape[1]);
  PySys_WriteStdout("Scene, near=%f, far=%f, fovy=%f, width=%d, height=%d\n",
                    znear, zfar, fovy, width, height);

  PyArrayObject *pos3d_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(pos3d_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *scale_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(scale_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *rot_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(rot_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *opacity_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(opacity_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *sh_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(sh_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  if (!viewmtx_arr || !projmtx_arr || !pos3d_arr || !scale_arr || !rot_arr ||
      !opacity_arr || !sh_arr) {
    PySys_WriteStderr("arr is null!");
    Py_XDECREF(bgcolor_arr);
    Py_XDECREF(viewmtx_arr);
    Py_XDECREF(projmtx_arr);
    Py_XDECREF(pos3d_arr);
    Py_XDECREF(scale_arr);
    Py_XDECREF(rot_arr);
    Py_XDECREF(opacity_arr);
    Py_XDECREF(sh_arr);
    return nullptr;
  }

  int ndim = PyArray_NDIM(pos3d_arr);
  npy_intp *shape = PyArray_SHAPE(pos3d_arr);

  int nsplats = shape[0];

  PySys_WriteStdout("pos3d ndim=%d, shape=(%d, %d)\n", ndim, shape[0],
                    shape[1]);

  ndim = PyArray_NDIM(scale_arr);
  shape = PyArray_SHAPE(scale_arr);
  PySys_WriteStdout("scale ndim=%d, shape=(%d, %d)\n", ndim, shape[0],
                    shape[1]);

  ndim = PyArray_NDIM(rot_arr);
  shape = PyArray_SHAPE(rot_arr);
  PySys_WriteStdout("rot ndim=%d, shape=(%d, %d)\n", ndim, shape[0], shape[1]);

  ndim = PyArray_NDIM(opacity_arr);
  shape = PyArray_SHAPE(opacity_arr);
  PySys_WriteStdout("opacity ndim=%d, shape=(%d, %d)\n", ndim, shape[0],
                    shape[1]);

  ndim = PyArray_NDIM(sh_arr);
  shape = PyArray_SHAPE(sh_arr);
  PySys_WriteStdout("sh ndim=%d, shape=(%d, %d)\n", ndim, shape[0], shape[1]);

  // float *image_dat = (float *)malloc(width * height * 4 * sizeof(float));

  gsplat::SplatScene splatScene;
  splatScene.viewmtx = static_cast<float *>(PyArray_DATA(viewmtx_arr));
  splatScene.projmtx = static_cast<float *>(PyArray_DATA(projmtx_arr));
  splatScene.bgcolor = static_cast<float *>(PyArray_DATA(bgcolor_arr));
  splatScene.eye = eye;
  splatScene.znear = znear;
  splatScene.zfar = zfar;
  splatScene.fovy = fovy;
  splatScene.width = width;
  splatScene.height = height;

  gsplat::SplatData splats;
  splats.nsplats = nsplats;
  splats.pos3d = static_cast<float *>(PyArray_DATA(pos3d_arr));
  splats.scale = static_cast<float *>(PyArray_DATA(scale_arr));
  splats.rot = static_cast<float *>(PyArray_DATA(rot_arr));
  splats.opacity = static_cast<float *>(PyArray_DATA(opacity_arr));
  splats.sh = static_cast<float *>(PyArray_DATA(sh_arr));

  float *image;
  gsplat_rasterize_gpu(splatScene, splats, &image, PySys_WriteStdout);

  if (image) {

    npy_intp image_size[3]{height, width, 4};
    PyObject *image_obj =
        PyArray_SimpleNewFromData(3, image_size, NPY_FLOAT32, image);

    if (image_obj) {
      PyObject *capsule = PyCapsule_New(image, nullptr, [](PyObject *capsule) {
        void *ptr = PyCapsule_GetPointer(capsule, nullptr);
        free(ptr);
      });

      if (capsule) {
        if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(image_obj),
                                  capsule) < 0) {
          Py_DECREF(capsule);
          Py_DECREF(image_obj);
        } else {
          ret_obj = image_obj;
        }
      } else {
        Py_DECREF(image_obj);
        free(image);
      }
    } else {
      free(image);
    }
  } else {
    PyErr_NoMemory();
  }

  Py_XDECREF(bgcolor_arr);
  Py_XDECREF(viewmtx_arr);
  Py_XDECREF(pos3d_arr);
  Py_XDECREF(scale_arr);
  Py_XDECREF(rot_arr);
  Py_XDECREF(opacity_arr);
  Py_XDECREF(sh_arr);

  return ret_obj;
}

static PyObject *gsplat_compute_debug(PyObject *self, PyObject *args) {
  PyObject *scene = nullptr;
  PyObject *pos3d_obj = nullptr;
  PyObject *scale_obj = nullptr;
  PyObject *rot_obj = nullptr;
  PyObject *opacity_obj = nullptr;
  PyObject *sh_obj = nullptr;

  if (!PyArg_ParseTuple(args, "OOOOOO", &scene, &pos3d_obj, &scale_obj,
                        &rot_obj, &opacity_obj, &sh_obj)) {
    return nullptr;
  }

  // --- scene
  PyObject *bgcolor_obj = PyObject_GetAttrString(scene, "bgcolor");
  if (!bgcolor_obj) {
    // 缺属性时 PyArray_FROM_OTF(NULL) 会直接崩，这里必须提前返回
    return nullptr;
  }
  PyArrayObject *bgcolor_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(bgcolor_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyObject *viewmtx_obj = PyObject_GetAttrString(scene, "viewmtx");
  PyArrayObject *viewmtx_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(viewmtx_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyObject *projmtx_obj = PyObject_GetAttrString(scene, "projmtx");
  PyArrayObject *projmtx_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(projmtx_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  PyObject *eye_obj = PyObject_GetAttrString(scene, "eye");
  PyObject *eye_x_obj = PyObject_GetAttrString(eye_obj, "x");
  PyObject *eye_y_obj = PyObject_GetAttrString(eye_obj, "y");
  PyObject *eye_z_obj = PyObject_GetAttrString(eye_obj, "z");
  float eye[3] = {static_cast<float>(PyFloat_AsDouble(eye_x_obj)),
                  static_cast<float>(PyFloat_AsDouble(eye_y_obj)),
                  static_cast<float>(PyFloat_AsDouble(eye_z_obj))};
  Py_XDECREF(eye_x_obj);
  Py_XDECREF(eye_y_obj);
  Py_XDECREF(eye_z_obj);
  Py_XDECREF(eye_obj);

  float znear = PyFloat_AsDouble(PyObject_GetAttrString(scene, "near"));
  float zfar = PyFloat_AsDouble(PyObject_GetAttrString(scene, "far"));
  float fovy = PyFloat_AsDouble(PyObject_GetAttrString(scene, "fovy"));
  int width = PyLong_AsLong(PyObject_GetAttrString(scene, "width"));
  int height = PyLong_AsLong(PyObject_GetAttrString(scene, "height"));

  // --- splat data
  PyArrayObject *pos3d_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(pos3d_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *scale_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(scale_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *rot_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(rot_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *opacity_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(opacity_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  PyArrayObject *sh_arr = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(sh_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));

  int nsplats = PyArray_SHAPE(pos3d_arr)[0];

  gsplat::SplatScene splatScene;
  splatScene.viewmtx = static_cast<float *>(PyArray_DATA(viewmtx_arr));
  splatScene.projmtx = static_cast<float *>(PyArray_DATA(projmtx_arr));
  splatScene.bgcolor = static_cast<float *>(PyArray_DATA(bgcolor_arr));
  splatScene.eye = eye;
  splatScene.znear = znear;
  splatScene.zfar = zfar;
  splatScene.fovy = fovy;
  splatScene.width = width;
  splatScene.height = height;

  gsplat::SplatData splats;
  splats.nsplats = nsplats;
  splats.pos3d = static_cast<float *>(PyArray_DATA(pos3d_arr));
  splats.scale = static_cast<float *>(PyArray_DATA(scale_arr));
  splats.rot = static_cast<float *>(PyArray_DATA(rot_arr));
  splats.opacity = static_cast<float *>(PyArray_DATA(opacity_arr));
  splats.sh = static_cast<float *>(PyArray_DATA(sh_arr));

  // --- call debug compute
  gsplat::SplatData2DGPU splats2d_host;

  // --- return as tuple of numpy arrays
  npy_intp dims2[2] = {nsplats, 2};
  npy_intp dims1[1] = {nsplats};
  npy_intp dims4[2] = {nsplats, 4};

  PyObject *py_pos2d =
      PyArray_SimpleNewFromData(2, dims2, NPY_FLOAT32, splats2d_host.pos2d);
  PyObject *py_depth =
      PyArray_SimpleNewFromData(1, dims1, NPY_FLOAT32, splats2d_host.depth);
  PyObject *py_cov2d =
      PyArray_SimpleNewFromData(2, dims4, NPY_FLOAT32, splats2d_host.cov2d);
  PyObject *py_rect =
      PyArray_SimpleNewFromData(2, dims4, NPY_FLOAT32, splats2d_host.rect);
  PyObject *py_color =
      PyArray_SimpleNewFromData(2, dims4, NPY_FLOAT32, splats2d_host.color);

  auto make_capsule = [](void *p) {
    return PyCapsule_New(p, nullptr, [](PyObject *c) {
      free(PyCapsule_GetPointer(c, nullptr));
    });
  };
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(py_pos2d),
                        make_capsule(splats2d_host.pos2d));
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(py_depth),
                        make_capsule(splats2d_host.depth));
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(py_cov2d),
                        make_capsule(splats2d_host.cov2d));
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(py_rect),
                        make_capsule(splats2d_host.rect));
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(py_color),
                        make_capsule(splats2d_host.color));

  PyObject *ret =
      PyTuple_Pack(5, py_pos2d, py_depth, py_cov2d, py_rect, py_color);
  Py_DECREF(py_pos2d);
  Py_DECREF(py_depth);
  Py_DECREF(py_cov2d);
  Py_DECREF(py_rect);
  Py_DECREF(py_color);

  Py_XDECREF(bgcolor_arr);
  Py_XDECREF(viewmtx_arr);
  Py_XDECREF(projmtx_arr);
  Py_XDECREF(pos3d_arr);
  Py_XDECREF(scale_arr);
  Py_XDECREF(rot_arr);
  Py_XDECREF(opacity_arr);
  Py_XDECREF(sh_arr);
  return ret;
}

static PyMethodDef Methods[] = {
    {"add_one", add_one, METH_VARARGS, "Example NumPy C++ function"},
    {"gsplat_rasterize", gsplat_rasterize, METH_VARARGS,
     "Gaussian Splats Rasterizer"},
    {"gsplat_compute_debug", gsplat_compute_debug, METH_VARARGS,
     "Debug: returns (pos2d, depth, cov2d, rect, color) from compute kernel"},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1,
                                    Methods};

PyMODINIT_FUNC PyInit__C(void) {
  import_array();
  return PyModule_Create(&module);
}
