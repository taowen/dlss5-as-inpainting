param(
    [string]$Configuration = "Release",
    [string]$StageTo = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$HarnessRoot = Join-Path $RepoRoot "third_party\DLSS5-Image-Converter"
$SdkRoot = Join-Path $RepoRoot "third_party\NVIDIA-DLSS"
$NativeRoot = Join-Path $HarnessRoot "native"
$BuildRoot = Join-Path $RepoRoot ".native-build\dlss5-image-converter"

if (-not (Test-Path -LiteralPath (Join-Path $SdkRoot "include\nvsdk_ngx.h"))) {
    throw "NVIDIA DLSS SDK submodule is not initialized. Run: git submodule update --init --recursive"
}
if (-not (Test-Path -LiteralPath (Join-Path $HarnessRoot "native\dlss5_eval\CMakeLists.txt"))) {
    throw "DLSS5-Image-Converter submodule is not initialized. Run: git submodule update --init --recursive"
}

$CMake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $CMake) {
    throw "cmake is required. Install CMake or the Visual Studio C++ workload."
}

& $CMake -S (Join-Path $NativeRoot "dlss5_eval") -B $BuildRoot -A x64 `
    "-DDLSS_SDK=$SdkRoot"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }

& $CMake --build $BuildRoot --config $Configuration
if ($LASTEXITCODE -ne 0) { throw "Native harness build failed." }

$Harness = Join-Path $NativeRoot "bin\dlss5_eval.exe"
if (-not (Test-Path -LiteralPath $Harness)) { throw "Built harness not found: $Harness" }

if ($StageTo) {
    $Stage = (Resolve-Path -LiteralPath $StageTo -ErrorAction Stop).Path
    Copy-Item -LiteralPath $Harness -Destination (Join-Path $Stage "dlss5_eval.exe") -Force
    Write-Host "Staged harness: $(Join-Path $Stage 'dlss5_eval.exe')" -ForegroundColor Green
}

Write-Host "Built harness: $Harness" -ForegroundColor Green
Write-Host "Run tools\dlss5_fp16_harness_probe.py from a runtime folder containing the user's ReShade/DLSS DLLs."
