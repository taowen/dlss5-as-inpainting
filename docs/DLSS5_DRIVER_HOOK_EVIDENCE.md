# DLSS5 driver-boundary hook evidence

This records the first successful driver-side hook for the RTX 5080 native
carrier. It is intentionally separate from the ReShade API trace: ReShade
sees resource creation, while the neural submission crosses into NVIDIA's
driver implementation.

## Reproduction

Build the add-on and place it beside the working D3D12 carrier:

```powershell
powershell -File tools\build_dlss5_reshade_capture.ps1
Copy-Item .native-build\reshade-capture\Release\dlss5_reshade_capture.addon64 <runtime>
python tools\dlss5_fp16_harness_probe.py --harness <runtime>\dlss5_eval.exe `
  --width 256 --height 256 --output runtime_probe_output\native_driver_hook
```

The add-on uses MinHook to observe `LoadLibrary*` and `GetProcAddress`, then
proxies the undocumented CUDA export table returned by `cuGetExportTable`.
Each table entry is replaced with a no-op machine-code recorder that restores
all observed integer arguments and tail-jumps to the original entry. This is
why the native image remains an unchanged control while the call arguments are
recorded.

## RTX 5080 result

On the local driver/runtime pair:

| observation | result |
|---|---|
| device | NVIDIA GeForce RTX 5080, PCI ID `0x2c02` |
| loaded CUDA module | `nvcuda64.dll` |
| CUDA entry used | `cuGetExportTable` |
| export-table UUID | `7f9212d6261ddd4d8af638dd1aeb10ae` |
| first table word | `0x1d8` = 472 bytes |
| table slots | 59 (`slot 0` is the size word) |
| ordinary `cuModuleLoadData` hook | no calls |
| ordinary `cuLaunchKernel` hook | no calls |
| actual caller of private slots | `nvwgf2umx.dll` |

One transparent 256² run hit these private slots:

```text
3, 12, 13, 14, 39, 40, 44, 50, 52, 53, 54
```

The high-frequency slots in that run were slot 50 (`7448` calls), slot 44
(`1064`), slot 52 (`1064`), and slot 53 (`2128`). The recorded return sites
all resolved to `nvwgf2umx.dll`; this is the dynamic proof that the missing
launch boundary is below NGX/ReShade and inside the display driver.

The native output metrics were unchanged from the clean control:

```text
history MAE 0.04204458522144705, RMSE 0.06093890106119408
motion  MAE 0.007987282471731305, RMSE 0.016925965247247328
```

## What this solves

The earlier statement “CUPTI sees no DLSS5 module” was incomplete. DLSS5 does
load real device code—the embedded CUBIN extractor already proves that—but
this carrier does not expose it through the ordinary CUDA launch API. The
private table hook now provides a stable place to identify the driver calls,
their caller sites, GPU/resource arguments, and eventually the private module
or command-buffer payload.

## Remaining capture work

The current recorder stores one argument snapshot per private slot. The next
instrumentation step is to retain a bounded ring of snapshots and copy pointed
host structures with `ReadProcessMemory` while `nvwgf2umx.dll` is still alive.
The useful targets are the high-frequency slots 44/50/52/53 and the table
entry whose argument contains the CUBIN or its driver command description.
Only after that payload is identified should a pre-kernel telemetry store be
constructed; the previous direct `STS -> STG` mutation had no valid driver
binding and correctly resulted in a device hang.

The table mechanism is undocumented by NVIDIA; the public CUDA API documents
the normal `cuLaunchKernel` route, not this table. The implementation is
therefore pinned to the observed driver build and remains a diagnostic hook,
not a production dependency.
