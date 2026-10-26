
# 编译安装
## windows
```powershell
. "E:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = "D:\Program Files\NVIDIA\CUDA\v13.4"
$env:CUDA_PATH = "D:\Program Files\NVIDIA\CUDA\v13.4"
$env:Path = "D:\Program Files\NVIDIA\CUDA\v13.4\bin;$env:Path"
python -m pip install -v -e . --no-build-isolation
```

## Linux
```shell
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
python -m pip install -v -e . --no-build-isolation
```

# IDE（clangd / VSCode）

clangd 配置不再手工维护：`.clangd` 和 `.vscode/settings.json` 由
`scripts/clangd/gen_clangd.py` 按当前机器生成（两个文件都已 gitignore）。

- 工具链来自 pixi（clang 23 / clangd），同时支持 Windows 和 Linux：
  ```shell
  pixi install             # 首次：按 pixi.toml 建环境（含 python/numpy 供 clangd 用）
  pixi run clangd-config   # 需要时手动重新生成（换机器 / 换 CUDA 后）
  ```
- **自动配置**：pixi 激活时会自动跑一次生成器（`pixi shell` / `pixi run`
  都会触发），幂等，无变化时打印“无变化”且不动文件。
- 生成内容里机器相关的只有两处来源，都会自动推导：
  - python / numpy 头文件 → 当前 pixi 环境的 `site-packages`（不再写死某台机器的 conda 路径）
  - CUDA → `CUDA_HOME` / `CUDA_PATH`（Windows 未设时会扫 `C:\Program Files\NVIDIA*\CUDA\v*`）
- VSCode 用 conda-forge 的 clangd：`clangd.path` 由脚本写成
  `${workspaceFolder}/.pixi/envs/default/...`，`pixi install` 之后即可用。
- 想换 clangd 用的 GPU arch（默认 `sm_120`）：设环境变量 `GSPLAT_CUDA_ARCH` 再重跑生成器。
