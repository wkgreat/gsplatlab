#Requires -Version 5.1
# 与 build_and_test.sh 对应的 Windows 版：只按顺序执行命令，不做环境探测。
# 运行前先进入 VS 开发者环境（cl.exe / rc.exe / mt.exe 在 PATH 中）。

$ErrorActionPreference = 'Stop'

# 以脚本所在目录（仓库根目录）为工作目录。
Set-Location -LiteralPath $PSScriptRoot

# CUDA 路径与本机一致；换机器时改这里。
# $env:CUDA_HOME = 'D:\Program Files\NVIDIA\CUDA\v13.4'
# $env:CUDA_PATH = $env:CUDA_HOME

# 统一 .pyc 输出目录，避免源码树里生成 __pycache__。
$env:PYTHONPYCACHEPREFIX = Join-Path $PSScriptRoot '.cache\pycache'

cmake -S . -B build/cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cugsplat
python -m pip install -e ./pygsplat --no-build-isolation
python ./pygsplat/test/test_cugsplat.py