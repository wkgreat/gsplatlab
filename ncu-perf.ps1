$ProfileDir = 'profile'
$ProfileFile = "./$ProfileDir/ncu_$(Get-Date -Format 'yyyyMMdd-HHmm')"
$PythonExe = 'D:\conda\envs\gsplatlab\python.exe'
$KernelName = 'gsplatTileAlphaCacheKernel'
$LaunchSkip = 0

if (-not (Test-Path $ProfileDir)) { New-Item -ItemType Directory $ProfileDir | Out-Null; Write-Host "created $ProfileDir" }
Write-Host $ProfileFile

# gsplatTileAlphaCacheKernel 每个 tile 调用一次（几千次），必须用 launch-count 只取一次调用
# application replay 避免在 pass 之间备份/恢复显存
ncu --replay-mode application --kernel-name $KernelName --launch-skip $LaunchSkip --launch-count 1 `
    --set basic -o $ProfileFile $PythonExe ./pygsplat/test/test_cugsplat.py
