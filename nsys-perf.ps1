$ProfileDir = 'profile'
$ProfileFile = "./$ProfileDir/profile_$(Get-Date -Format 'yyyyMMdd-HHmm')"

if (-not (Test-Path $ProfileDir)) { New-Item -ItemType Directory $ProfileDir | Out-Null; Write-Host "created $ProfileDir" }
Write-Host $ProfileFile

# CPU sampling and context-switch tracing need an elevated (Administrator) shell.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$cpuTrace = @()
if (-not $isAdmin) {
    Write-Host 'not elevated: CPU sampling / context-switch trace disabled (run as Administrator for CPU-side data)'
    $cpuTrace = @('--sample=none', '--cpuctxsw=none')
}

nsys profile @cpuTrace --trace="cuda,nvtx" -o $ProfileFile conda run -n gsplatlab python ./pygsplat/gsplat_rasterize_test.py

nsys-ui "$ProfileFile.nsys-rep"