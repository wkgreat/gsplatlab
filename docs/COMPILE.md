# 编译与测试

本仓库分为两部分：

- `cugsplat/`：3DGS 的 C++/CUDA 实现，由 CMake 编成动态库（`cugsplat.dll` / `libcugsplat.so`）。
- `pygsplat/`：`cugsplat` 的 Python 绑定，`setup.py` 只编译 `csrc/gsplatrender_binding.cpp`
  并链接上一步生成的动态库，不再重复编译 `.cu`。

构建产物约定（`pygsplat/setup.py` 按此查找）：

```text
build/cugsplat/lib/    cugsplat.lib / libcugsplat.so（导入库）
build/cugsplat/bin/    cugsplat.dll（Windows 运行时库）
```

换构建目录时可用环境变量覆盖：`CUGSPLAT_BUILD_DIR`、`CUGSPLAT_LIB_DIR`、`CUGSPLAT_BIN_DIR`。

## 环境要求

- conda 环境 `gsplatlab`（Python 3.11，含 `numpy`、`pillow`、`plyfile`、`pyglm`）
- CMake >= 3.24、Ninja
- CUDA Toolkit（`CUDA_HOME` 或 `CUDA_PATH` 指向安装目录）
- Visual Studio 2022（Windows）
- 可用的 NVIDIA GPU（CMake 默认按当前 GPU 生成 `native` 架构）

## Windows

> 第 1 步需要在 **x64 Native Tools Command Prompt for VS 2022**
> （或已导入 `vcvars64.bat` 的终端）中执行，保证 `cl.exe` 和标准库头可用。

```bat
:: 1. 构建 cugsplat
cmake -S . -B build\cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build\cugsplat
```

产物：

```text
build\cugsplat\bin\cugsplat.dll
build\cugsplat\lib\cugsplat.lib
```

```powershell
# 2. 安装 pygsplat（-e 可编辑安装，链接上面的 cugsplat.lib）
conda run -n gsplatlab python -m pip install -e .\pygsplat --no-build-isolation

# 3. 运行测试，输出 output.png
conda run -n gsplatlab python .\pygsplat\gsplat_rasterize_test.py
```

`setup.py` 会把 `cugsplat.dll` 复制到 `pygsplat\pygsplat\_C.cpXX-win_amd64.pyd` 旁边，
`import pygsplat` 时即可正常加载。

### 无 GPU / 交叉编译

默认 `CMAKE_CUDA_ARCHITECTURES=native`，按当前显卡生成。需要固定架构时：

```bat
cmake -S . -B build\cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
```

## Linux

```shell
# 1. 构建 cugsplat
cmake -S . -B build/cugsplat -DCMAKE_BUILD_TYPE=Release
cmake --build build/cugsplat -j

# 2. 安装 pygsplat
conda run -n gsplatlab python -m pip install -e ./pygsplat --no-build-isolation

# 3. 运行测试
conda run -n gsplatlab python ./pygsplat/gsplat_rasterize_test.py
```

`libcugsplat.so` 会被复制到扩展旁边，扩展通过 `-Wl,-rpath,$ORIGIN` 加载。

## 验证结果

测试脚本正常时应输出：

```text
Scene, viewmtx ndim=2 shape=(4, 4)
Scene, near=0.010000, far=1000.000000, fovy=1.570796, width=500, height=500
pos3d ndim=2, shape=(139410, 4)
scale ndim=2, shape=(139410, 4)
rot ndim=2, shape=(139410, 4)
opacity ndim=2, shape=(139410, 1)
sh ndim=2, shape=(139410, 48)
scene bgcolor=(0.000000,0.000000,0.000000,1.000000)
```

并在当前目录生成 `output.png`。与基线对比：

```powershell
conda run -n gsplatlab python -c "import numpy as np; from PIL import Image; a=np.asarray(Image.open('output.png')).astype(np.int32); b=np.asarray(Image.open('pygsplat/baseline_20261017.png')).astype(np.int32); d=np.abs(a-b); print('maxdiff', d.max(), 'ndiff', np.count_nonzero(d), '/', d.size)"
```

预期 `maxdiff=1`，仅极少数通道存在 1/255 的浮点舍入差异（CUDA/GPU 版本不同会产生
几十个这样的通道），视觉上完全一致。

## 常见问题

- **找不到 cugsplat 导入库**：先执行上面的 `cmake --build`，或设置
  `CUGSPLAT_BUILD_DIR` 指向已有构建目录。
- **CUDA 版本**：`setup.py` 从 `CUDA_HOME` / `CUDA_PATH` 查找 CUDA；Windows 上
  未设置时会沿用当前 PATH 中的 `nvcc`，请确保 CUDA 的 `bin` 也在 `PATH` 中。
- **CRT 不一致**：`cugsplat` 与 Python 扩展之间的 `malloc/free` 跨 DLL，
  Windows 下请始终用 `Release`（`/MD`）构建 `cugsplat`，不要用 `Debug`（`/MDd`）。

## 清理重建

```powershell
Remove-Item -Recurse -Force build\cugsplat
Remove-Item -Force pygsplat\pygsplat\_C.*.pyd, pygsplat\pygsplat\cugsplat.dll
```

```shell
rm -rf build/cugsplat pygsplat/pygsplat/_C.*.so pygsplat/pygsplat/libcugsplat.so
```
