param(
    [Parameter(Mandatory = $true)]
    [string]$CuptiRoot,
    [Parameter(Mandatory = $true)]
    [string]$CudaRoot,
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $RepoRoot ".native-build\cupti-capture"
$CuptiRoot = (Resolve-Path -LiteralPath $CuptiRoot).Path
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path

if (-not (Test-Path -LiteralPath (Join-Path $CuptiRoot "include\cupti.h"))) {
    throw "CUPTI root must contain include\cupti.h"
}
if (-not (Test-Path -LiteralPath (Join-Path $CudaRoot "include\cuda.h"))) {
    throw "CUDA root must contain include\cuda.h"
}

$CMake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $CMake) { throw "cmake is required." }

& $CMake -S (Join-Path $RepoRoot "tools\cupti_capture") -B $BuildRoot -A x64 `
    "-DCUPTI_ROOT=$CuptiRoot" "-DCUDA_ROOT=$CudaRoot"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }

& $CMake --build $BuildRoot --config $Configuration
if ($LASTEXITCODE -ne 0) { throw "CUPTI capture build failed." }

$Capture = Join-Path $BuildRoot "bin\dlss5_cupti_capture.dll"
if (-not (Test-Path -LiteralPath $Capture)) {
    throw "Capture DLL not found: $Capture"
}
Write-Host "Built CUPTI capture: $Capture" -ForegroundColor Green
