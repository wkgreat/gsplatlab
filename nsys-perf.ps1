#Requires -Version 5.1
<#
.SYNOPSIS
    用 Nsight Systems 采集 pygsplat/gsplat_rasterize_test.py 的性能数据，并打开采集结果。

.DESCRIPTION
    nsys-perf.sh 的 PowerShell 版本，输出文件为 profile\profile_<yyyyMMdd-HHmm>.nsys-rep。

.PARAMETER CondaEnv
    conda 环境名。

.PARAMETER Script
    被采集的 python 脚本，相对于本脚本所在目录（项目根目录）。

.PARAMETER ProfileDir
    profile 输出目录。

.PARAMETER NoUi
    只采集，不打开 nsys-ui。

.EXAMPLE
    .\nsys-perf.ps1

.EXAMPLE
    .\nsys-perf.ps1 -CondaEnv gsplatlab -ProfileDir profile -NoUi
#>
[CmdletBinding()]
param(
    [string]$CondaEnv   = 'gsplatlab',
    [string]$Script     = 'pygsplat\gsplat_rasterize_test.py',
    [string]$ProfileDir = 'profile',
    [switch]$NoUi
)

$ErrorActionPreference = 'Stop'

# 以脚本所在目录（项目根目录）为工作目录，不依赖调用者当前的 cwd。
if ($PSScriptRoot) { Set-Location -LiteralPath $PSScriptRoot }

# 1. 确保输出目录存在
if (-not (Test-Path -LiteralPath $ProfileDir -PathType Container)) {
    New-Item -ItemType Directory -Path $ProfileDir -Force | Out-Null
    Write-Host "created $ProfileDir"
}

# 2. 输出文件名：profile\profile_<yyyyMMdd-HHmm>
$ProfileFile = Join-Path $ProfileDir ('profile_' + (Get-Date -Format 'yyyyMMdd-HHmm'))
Write-Host $ProfileFile

# 3. 解析 conda 环境中的 python.exe，让 nsys 直接拉起解释器，
#    避免 conda run / cmd 的包装进程出现在调用栈里。
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

$target = Join-Path $PSScriptRoot $Script
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "待采集的脚本不存在：$target"
}

# 4. 采集：nsys profile -o <输出> <python> <脚本>
$nsys = Get-Command nsys -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
if (-not $nsys) { throw '在 PATH 中找不到 nsys，请把 Nsight Systems 的 bin 目录加入 PATH。' }

Write-Host "nsys profile -o $ProfileFile $python $target"
& $nsys.Source profile -o $ProfileFile $python $target
if ($LASTEXITCODE -ne 0) { throw "nsys profile 失败，退出码 $LASTEXITCODE。" }

$report = "$ProfileFile.nsys-rep"
if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
    throw "没有生成报告文件：$report"
}
# GUI 进程不共享本进程的 cwd，打开报告要用绝对路径。
$reportFull = (Resolve-Path -LiteralPath $report).ProviderPath
Write-Host "report: $reportFull"

# 5. 打开结果：nsys-ui <输出>.nsys-rep
if ($NoUi) { return }

$nsysUi = Get-Command nsys-ui -CommandType Application -ErrorAction SilentlyContinue |
          Select-Object -First 1
if ($nsysUi) {
    $uiExe = $nsysUi.Source
} else {
    # Windows 安装包里 nsys-ui.exe 在 host-windows-x64，和 PATH 里的
    # target-windows-x64\nsys.exe 是兄弟目录，需要自己找。
    $uiExe = Join-Path (Split-Path -Parent (Split-Path -Parent $nsys.Source)) 'host-windows-x64\nsys-ui.exe'
    if (-not (Test-Path -LiteralPath $uiExe -PathType Leaf)) {
        throw '在 PATH 和 Nsight Systems 安装目录中都找不到 nsys-ui。'
    }
}

Start-Process -FilePath $uiExe -ArgumentList "`"$reportFull`""
