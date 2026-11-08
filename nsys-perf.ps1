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

$Duration = 60

nsys profile @cpuTrace --trace="cuda,nvtx" --duration=$Duration -o $ProfileFile conda run -n gsplatlab python ./pygsplat/test/test_cugsplat.py

nsys-ui "$ProfileFile.nsys-rep"