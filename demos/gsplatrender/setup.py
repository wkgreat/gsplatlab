import os
import subprocess

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home is None:
    raise RuntimeError("CUDA toolkit not found. Please set CUDA_HOME or CUDA_PATH.")

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dir = (
    os.path.join(cuda_home, "lib", "x64")
    if os.name == "nt"
    else os.path.join(cuda_home, "lib64")
)
nvcc = os.path.join(cuda_home, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")


class CUDAExtension(Extension):
    pass


class BuildExtension(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        super().build_extensions()

    def build_extension(self, ext):
        objects = []

        for source in ext.sources:
            if not source.endswith(".cu"):
                objects.extend(
                    self.compiler.compile(
                        [source],
                        output_dir=self.build_temp,
                        macros=ext.define_macros,
                        include_dirs=ext.include_dirs,
                        debug=self.debug,
                        extra_postargs=ext.extra_compile_args,
                        depends=ext.depends,
                    )
                )
                continue

            obj_ext = ".obj" if os.name == "nt" else ".o"
            obj = os.path.join(
                self.build_temp,
                os.path.splitext(source)[0].replace("/", os.sep) + obj_ext,
            )
            os.makedirs(os.path.dirname(obj), exist_ok=True)

            cmd = [
                nvcc,
                "-c",
                source,
                "-o",
                obj,
                "-O2",
                "--compiler-options",
                "/MD,/utf-8" if os.name == "nt" else "-fPIC",
            ]
            cmd.extend(f"-I{include_dir}" for include_dir in ext.include_dirs)
            subprocess.check_call(cmd)
            objects.append(obj)

        ext.extra_objects = list(ext.extra_objects or []) + objects
        ext.sources = []
        super().build_extension(ext)


setup(
    name="gsplatrender",
    packages=["gsplatrender"],
    ext_modules=[
        CUDAExtension(
            name="gsplatrender._C",
            sources=[
                "csrc/gsplat_tile_index.cu",
                "csrc/gsplat_rasterizer.cu",
                "csrc/cuda_utils.cu",
                "csrc/gsplatrender_binding.cpp",
            ],
            include_dirs=[np.get_include(), "csrc", cuda_include_dir],
            library_dirs=[cuda_library_dir],
            libraries=["cudart"],
            extra_compile_args=["/O2", "/utf-8"] if os.name == "nt" else ["-O2"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
