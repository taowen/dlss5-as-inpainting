param(
    [string]$Cubit = "",
    [string]$Cubin = "cubins\fatbin_00\fatbin_00_0xdf0e0.4.sm_120.cubin",
    [string]$Kernel = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8",
    [string]$Output = "runtime_probe_output\cubit_roundtrip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Cubit) { $Cubit = Join-Path $RepoRoot "third_party\cubit\target\release\cubit.exe" }
$Cubit = (Resolve-Path -LiteralPath $Cubit -ErrorAction Stop).Path
$Cubin = (Resolve-Path -LiteralPath (Join-Path $RepoRoot $Cubin) -ErrorAction Stop).Path
$Output = Join-Path $RepoRoot $Output
New-Item -ItemType Directory -Force $Output | Out-Null

$Sass = Join-Path $Output "pre.frozen.sass"
$Roundtrip = Join-Path $Output "pre.roundtrip.cubin"
$Table = Join-Path (Split-Path -Parent $Cubit) "..\..\tables\sm120.json"
$Table = (Resolve-Path -LiteralPath $Table -ErrorAction Stop).Path
$CubitWorkdir = Split-Path -Parent (Split-Path -Parent $Cubit)

Push-Location $CubitWorkdir
try {
    & $Cubit disassemble $Cubin --table $Table --kernel $Kernel --frozen --output $Sass
    if ($LASTEXITCODE -ne 0) { throw "cubit disassemble failed." }
    & $Cubit asm $Sass --table $Table --template $Cubin --kernel $Kernel --output $Roundtrip
    if ($LASTEXITCODE -ne 0) { throw "cubit assemble failed." }
}
finally {
    Pop-Location
}

$OriginalHash = (Get-FileHash -LiteralPath $Cubin -Algorithm SHA256).Hash
$RoundtripHash = (Get-FileHash -LiteralPath $Roundtrip -Algorithm SHA256).Hash
$Report = [ordered]@{
    cubit = $Cubit
    table = $Table
    input = $Cubin
    kernel = $Kernel
    input_bytes = (Get-Item -LiteralPath $Cubin).Length
    roundtrip_bytes = (Get-Item -LiteralPath $Roundtrip).Length
    input_sha256 = $OriginalHash
    roundtrip_sha256 = $RoundtripHash
    bit_exact = ($OriginalHash -eq $RoundtripHash)
}
$Report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "manifest.json") -Encoding utf8
$Report | ConvertTo-Json
if (-not $Report.bit_exact) { throw "cubit roundtrip changed the CUBIN bytes." }
