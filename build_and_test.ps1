#Requires -Version 5.1
<#
.SYNOPSIS
    构建 cugsplat，把 pygsplat 装成可编辑包，并跑一次光栅化测试。

.DESCRIPTION
    等价于依次执行：
        cmake -S . -B build\cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release
        cmake --build build\cugsplat
        python -m pip install -e .\pygsplat --no-build-isolation
        python .\pygsplat\gsplat_rasterize_test.py
    区别只在两处：
      * 先进入 MSVC / Windows SDK 的开发环境，链接阶段才找得到 rc.exe / mt.exe；
      * python 直接用 conda 环境里的解释器，不依赖当前 shell 是否 conda activate 过。

.PARAMETER CondaEnv
    用来装 pygsplat、跑测试的 conda 环境名。

.PARAMETER BuildDir
    cugsplat 的 CMake 构建目录，必须和 pygsplat\setup.py 的默认查找路径一致。

.PARAMETER Configuration
    CMake 构建类型。

.EXAMPLE
    .\build_and_test.ps1
#>
[CmdletBinding()]
param(
    [string]$CondaEnv      = 'gsplatlab',
    [string]$BuildDir      = 'build\cugsplat',
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'

# 以脚本所在目录（项目根目录）为工作目录，不依赖调用者当前的 cwd。
if ($PSScriptRoot) { Set-Location -LiteralPath $PSScriptRoot }

# ---------------------------------------------------------------------------
# 1. 初始化 MSVC / Windows SDK 环境
#    只把 cl.exe 放进 PATH 是不够的：CMake 链接时要用 rc.exe / mt.exe 生成
#    清单，还要 SDK 的 INCLUDE / LIB，这些只有 vcvars / Enter-VsDevShell 才配。
#    缺了它们的典型报错是：
#      RC Pass 1: command "rc /fo .../manifest.res ..." failed ... no such file
# ---------------------------------------------------------------------------
if (-not $env:VCToolsInstallDir) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "找不到 vswhere.exe，请先安装 Visual Studio（勾选 C++ 桌面开发工作负载）。"
    }

    # 注意：这里不能把 vswhere 的输出直接接 Select-Object -First 1——
    # 提前截断会让 PowerShell 终止 vswhere 进程，$LASTEXITCODE 变成被杀的退出码。
    $vsPaths = @(& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
    $vsExit = $LASTEXITCODE
    if ($vsExit -eq 0 -and -not $vsPaths) {
        # 组件 ID 对不上时（不同 VS 版本 / 工作负载命名），退一步只要最新实例。
        $vsPaths = @(& $vswhere -latest -products * -property installationPath)
        $vsExit = $LASTEXITCODE
    }
    $vsPath = $vsPaths | Where-Object { $_.Trim() } | Select-Object -First 1
    if ($vsExit -ne 0 -or -not $vsPath) {
        throw "vswhere 没有找到装了 MSVC C++ 工具的 Visual Studio。"
    }
    $vsPath = $vsPath.Trim()

    $devShell = Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    if (-not (Test-Path -LiteralPath $devShell)) {
        throw "找不到 VS 的 DevShell 模块：$devShell"
    }

    Import-Module $devShell
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation `
                     -DevCmdArguments '-arch=x64 -host_arch=x64' -ErrorAction Stop | Out-Null
}

foreach ($tool in 'cl.exe', 'rc.exe', 'mt.exe') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "开发环境里找不到 $tool。若缺 rc.exe / mt.exe，请在 Visual Studio 安装器里补装 Windows SDK。"
    }
}

# CUDA：setup.py 只认 CUDA_HOME / CUDA_PATH，两个都没设时用 nvcc 的位置兜底。
if (-not ($env:CUDA_HOME -or $env:CUDA_PATH)) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) { $env:CUDA_PATH = Split-Path -Parent (Split-Path -Parent $nvcc.Source) }
}

# ---------------------------------------------------------------------------
# 2. 配置并构建 cugsplat
# ---------------------------------------------------------------------------
Write-Host "==> cmake configure ($Configuration)"
$cmakeArgs = @('-S', '.', '-B', $BuildDir, '-G', 'Ninja', "-DCMAKE_BUILD_TYPE=$Configuration")

# CMake 的 CUDA_ARCHITECTURES=native 靠跑一个探测程序枚举 GPU 来实现，在远程/
# 受限环境里会直接失败（CUDA_ARCHITECTURES is set to "native", but no NVIDIA
# GPU was detected）。先用 nvidia-smi 查算力显式传进去；查不到就沿用 native。
if (-not $env:CMAKE_CUDA_ARCHITECTURES) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $caps = & $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $caps) {
            $arch = @($caps |
                ForEach-Object { $_.Trim() -replace '\.', '' } |
                Where-Object { $_ -match '^\d+$' } |
                Sort-Object -Unique) -join ';'
            if ($arch) {
                $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$arch"
                Write-Host "==> CUDA arch: $arch"
            }
        }
    }
}

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake 配置失败，退出码 $LASTEXITCODE。" }

Write-Host '==> cmake build'
& cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cugsplat 构建失败，退出码 $LASTEXITCODE。" }

$importLib = Join-Path $BuildDir 'lib\cugsplat.lib'
if (-not (Test-Path -LiteralPath $importLib)) {
    throw "构建结束但没有生成导入库：$importLib"
}

# ---------------------------------------------------------------------------
# 3. 找到 conda 环境里的 python
# ---------------------------------------------------------------------------
$conda = Get-Command conda -CommandType Application -ErrorAction SilentlyContinue |
         Select-Object -First 1
if (-not $conda) { throw '在 PATH 中找不到 conda。' }

$envListJson = & $conda.Source env list --json | Out-String
if ($LASTEXITCODE -ne 0) { throw "conda env list 失败，退出码 $LASTEXITCODE。" }
$envRoots = @(($envListJson | ConvertFrom-Json).envs)

$envRoot = $envRoots |
    Where-Object { [System.IO.Path]::GetFileName($_) -eq $CondaEnv } |
    Select-Object -First 1
if (-not $envRoot) {
    throw "找不到名为 '$CondaEnv' 的 conda 环境。已有环境：$($envRoots -join ', ')"
}

$python = @(
    (Join-Path $envRoot 'python.exe')
    (Join-Path $envRoot 'bin/python')
    (Join-Path $envRoot 'bin/python3')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $python) { throw "conda 环境 '$CondaEnv' 中找不到 python 解释器：$envRoot" }
Write-Host "==> python: $python"

# ---------------------------------------------------------------------------
# 4. 安装 pygsplat（可编辑），再跑测试
# ---------------------------------------------------------------------------
Write-Host '==> pip install -e pygsplat'
& $python -m pip install -e .\pygsplat --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "pip install 失败，退出码 $LASTEXITCODE。" }

Write-Host '==> run gsplat_rasterize_test.py'
& $python .\pygsplat\gsplat_rasterize_test.py
if ($LASTEXITCODE -ne 0) { throw "测试脚本失败，退出码 $LASTEXITCODE。" }

Write-Host '==> done'
