param(
    [string]$Configuration = "Release",
    [string]$StageTo = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ReshadeRoot = Join-Path $RepoRoot "third_party\reshade"
$BuildRoot = Join-Path $RepoRoot ".native-build\reshade-capture"

if (-not (Test-Path -LiteralPath (Join-Path $ReshadeRoot "include\reshade.hpp"))) {
    throw "ReShade submodule is not initialized. Run: git submodule update --init third_party/reshade"
}
$CMake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $CMake) { throw "cmake is required. Install CMake or the Visual Studio C++ workload." }

& $CMake -S (Join-Path $RepoRoot "tools\reshade_capture") -B $BuildRoot -A x64
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }
& $CMake --build $BuildRoot --config $Configuration
if ($LASTEXITCODE -ne 0) { throw "ReShade capture add-on build failed." }

$Addon = Join-Path $BuildRoot "$Configuration\dlss5_reshade_capture.addon64"
if (-not (Test-Path -LiteralPath $Addon)) { throw "Built add-on not found: $Addon" }
if ($StageTo) {
    $Stage = (Resolve-Path -LiteralPath $StageTo -ErrorAction Stop).Path
    Copy-Item -LiteralPath $Addon -Destination (Join-Path $Stage "dlss5_reshade_capture.addon64") -Force
    Write-Host "Staged ReShade capture add-on: $(Join-Path $Stage 'dlss5_reshade_capture.addon64')" -ForegroundColor Green
} else {
    Write-Host "Built ReShade capture add-on: $Addon" -ForegroundColor Green
}
