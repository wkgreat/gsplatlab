"""pygsplat: cugsplat 的 Python 绑定。

cugsplat（C++/CUDA 实现）由根目录 CMakeLists.txt 编译成动态库；
这里只编译 binding.cpp 并链接该动态库，不再重复编译 .cu。
"""

import os
import shutil
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CUGSPLAT_DIR = REPO_ROOT / "cugsplat"

cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home is None:
    raise RuntimeError("CUDA toolkit not found. Please set CUDA_HOME or CUDA_PATH.")

cuda_root = Path(cuda_home)
cuda_include_dir = cuda_root / "include"
cuda_library_dir = (
    cuda_root / "lib" / "x64" if os.name == "nt" else cuda_root / "lib64"
)

# 与根 CMakeLists.txt 的产物约定保持一致：<build>/lib 放导入库，<build>/bin 放运行时库。
# 换构建目录时用环境变量覆盖，不写死任何机器路径。
cugsplat_build_dir = Path(
    os.environ.get("CUGSPLAT_BUILD_DIR", REPO_ROOT / "build" / "cugsplat")
)
cugsplat_lib_dir = Path(
    os.environ.get("CUGSPLAT_LIB_DIR", cugsplat_build_dir / "lib")
)
cugsplat_bin_dir = Path(
    os.environ.get("CUGSPLAT_BIN_DIR", cugsplat_build_dir / "bin")
)

cugsplat_import_lib = "cugsplat.lib" if os.name == "nt" else "libcugsplat.so"
cugsplat_runtime_lib = "cugsplat.dll" if os.name == "nt" else "libcugsplat.so"


def find_first(dirs, name):
    for directory in dirs:
        path = Path(directory) / name
        if path.is_file():
            return path
    return None


def require_cugsplat_runtime() -> Path:
    runtime = find_first((cugsplat_bin_dir, cugsplat_lib_dir), cugsplat_runtime_lib)
    if runtime is None:
        raise RuntimeError(
            f"找不到 cugsplat 动态库：{cugsplat_bin_dir / cugsplat_runtime_lib}\n"
            "请先构建 cugsplat：\n"
            f'  cmake -S "{REPO_ROOT}" -B "{cugsplat_build_dir}" -DCMAKE_BUILD_TYPE=Release\n'
            f'  cmake --build "{cugsplat_build_dir}"\n'
            "或用 CUGSPLAT_BUILD_DIR / CUGSPLAT_LIB_DIR 指向已有构建目录。"
        )
    return runtime


class BuildExtension(build_ext):
    def build_extensions(self):
        if find_first((cugsplat_lib_dir, cugsplat_bin_dir), cugsplat_import_lib) is None:
            raise RuntimeError(
                f"找不到 cugsplat 导入库：{cugsplat_lib_dir / cugsplat_import_lib}\n"
                "请先执行 cmake --build 构建 cugsplat。"
            )

        super().build_extensions()

        # 把 cugsplat 运行时库放到 _C 扩展旁边：
        #   - Windows 下 _C.pyd 加载时需要能找到 cugsplat.dll
        #   - Linux 下配合 $ORIGIN 的 rpath 解析 libcugsplat.so
        runtime = require_cugsplat_runtime()
        ext_path = Path(self.get_ext_fullpath("pygsplat._C"))
        for target_dir in {ext_path.parent, HERE / "pygsplat"}:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime, target_dir / runtime.name)


extra_compile_args = (
    ["/O2", "/utf-8", "/Zc:preprocessor", "/std:c++17"]
    if os.name == "nt"
    else ["-O2", "-std=c++17", "-fPIC"]
)
extra_link_args = [] if os.name == "nt" else ["-Wl,-rpath,$ORIGIN"]

setup(
    name="pygsplat",
    version="0.0.0",
    packages=["pygsplat"],
    package_data={"pygsplat": [cugsplat_runtime_lib]},
    ext_modules=[
        Extension(
            name="pygsplat._C",
            sources=["csrc/gsplatrender_binding.cpp"],
            include_dirs=[
                np.get_include(),
                str(CUGSPLAT_DIR),
                str(cuda_include_dir),
            ],
            library_dirs=[str(cugsplat_lib_dir), str(cuda_library_dir)],
            libraries=["cugsplat", "cudart"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
