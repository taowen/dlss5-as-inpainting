param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$CubitRoot = Join-Path $RepoRoot "third_party\cubit"

$Cargo = (Get-Command cargo -ErrorAction SilentlyContinue).Source
if (-not $Cargo) {
    $UserCargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
    if (Test-Path -LiteralPath $UserCargo) { $Cargo = $UserCargo }
}
if (-not $Cargo) {
    throw "Rust cargo is required. Install Rust 1.87+ or add .cargo\bin to PATH."
}
if (-not (Test-Path -LiteralPath (Join-Path $CubitRoot "Cargo.toml"))) {
    throw "cubit submodule is not initialized. Run: git submodule update --init third_party/cubit"
}

Push-Location $CubitRoot
try {
    & $Cargo build --release --locked
    if ($LASTEXITCODE -ne 0) { throw "cubit build failed." }
}
finally {
    Pop-Location
}

$Cubit = Join-Path $CubitRoot "target\release\cubit.exe"
if (-not (Test-Path -LiteralPath $Cubit)) { throw "cubit executable not found: $Cubit" }
Write-Host "Built cubit: $Cubit" -ForegroundColor Green
