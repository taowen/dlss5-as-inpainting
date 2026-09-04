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
Each table entry is replaced with a machine-code recorder that restores all
observed integer arguments and tail-jumps to the original entry. The recorder
writes a fixed-layout, process-specific `dlss5_driver_trace_<pid>.bin` mapping;
the first and latest argument snapshots are therefore available even when the
ReShade unload callback is not. Read one with:

```powershell
python tools\read_dlss5_driver_trace.py <runtime>\dlss5_driver_trace_<pid>.bin
```

Set `DLSS5_DARK_SCAN=1` to dump CUBIN-like pointed arguments. Add
`DLSS5_DARK_SCAN_ALL=1` to scan every called private slot instead of the
high-value slots 12 and 50.

Set `DLSS5_D3D12_CAPTURE_NEURAL=1` to enable the optional GPU readback. It
copies the two root-0 SRV textures after the `Neural` composition dispatch and
writes `dlss5_d3d12_capture_<pid>_<n>.rgba16f.bin` beside the harness. The
capture uses a fence and restores the source resource state before returning;
the clean-control metrics are a required regression check when using it.

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

In a current four-process 256² run, one process recorded slot 50 (`3724`
calls), slot 44 (`532`), slot 52 (`532`), and slot 53 (`1064`), with the other
called slots being 3/12/13/14/39/40/54. The recorded return sites all resolved
to `nvwgf2umx.dll`; this is the dynamic proof that the missing launch boundary
is below NGX/ReShade and inside the display driver. The counts are per process
and depend on the number of frames sent to that process.

The native output metrics were unchanged from the clean control:

```text
history MAE 0.04204458522144705, RMSE 0.06093890106119408
motion  MAE 0.007987282471731305, RMSE 0.016925965247247328
```

## Native D3D12 graph evidence

The same add-on also hooks the native D3D12 device and command-list vtables.
The hook is observation-only and the clean-control metrics above remain
unchanged. A 256² process produced:

```text
CreateComputePipelineState: 2 PSOs
  4900 bytes: Original(t2) -> Output(u0)
  6404 bytes: Neural(t2) + OutputOriginal(t3) -> Output(u0)
Dispatch: (16, 16, 1)
```

The two PSO bytecodes are written beside the harness as
`dlss5_d3d12_cs_<pid>_<n>.dxil` (the carrier currently returns DXBC-format
shader blobs despite the historical `.dxil` filename). Descriptor-heap and
view hooks record the CPU/GPU handle mapping. For the representative process,
the six-descriptor shader-visible heap had a 32-byte stride; root table 0 used
the heap base, while the output table used descriptor 4 or 5. This matches the
two shader resource layouts above and identifies the final Original/Neural
composition pass.

The optional readback was run once per input with a 256x256 fence-synchronized
probe. The root-0 descriptor 0 capture matched the input ramp/checker, while
the second source consumed by the `Neural` shader had these means:

```text
color   original [0.499838, 0.499837, 0.249923]
color   neural   [0.626684, 0.626682, 0.491251]
checker original [0.449284, 0.404327, 0.359444]
checker neural   [0.537337, 0.511911, 0.484781]
```

This is the first direct image-space readback of the live Neural result in the
carrier. It is not an internal DLSS5 activation/tensor dump: the driver-owned
pre-front feature producer and its tensor layout remain opaque.

The four raw RGBA16F planes and their hashes are committed under
`evidence/dlss5_d3d12_capture_256/` for offline comparison.

## Driver-side CUBIN evidence

With `DLSS5_DARK_SCAN=1`, slot 12's `r8` argument is a repeatable 15,656-byte
blob:

```text
magic       50 ed 55 ba
declared    0x3d28 bytes
ELF offset  0x50
raw ELF     15,576 bytes
SHA-256     D3DD26C7F14C1E701256372C3B96F30A54D2C17FF3F6C96CEE7FE349A723F216
```

The raw ELF is detected as `sm_75` and disassembles to the entry
`cuda_clear_buffer_kernel`; it performs a guarded byte clear followed by an
`STG.E` store. This is a real driver-side CUBIN passed through the private
boundary, but it is a utility/initialization kernel, not the DLSS5 neural
model. The large embedded `sm_120` pre CUBIN and the driver-side clear CUBIN
are therefore different artifacts.

## What this solves

The earlier statement “CUPTI sees no DLSS5 module” was incomplete. DLSS5 does
load real device code—the embedded CUBIN extractor already proves that—but
this carrier does not expose it through the ordinary CUDA launch API. The
private table hook now provides a stable place to identify the driver calls,
their caller sites, GPU/resource arguments, and eventually the private module
or command-buffer payload.

## Remaining capture work

The current recorder stores first/latest snapshots and can dump pointed CUBIN
containers. The optional D3D12 readback exposes the live Neural image, but not
the driver-owned pre-front tensor or its internal command description. The
D3D12 evidence proves the visible carrier composition pass; the private table
evidence proves the lower driver boundary. The remaining blocker for a
bit-exact PyTorch model is the missing pre-front tensor producer and its driver
resource bindings. The previous direct
`STS -> STG` mutation had no valid driver binding and correctly resulted in a
device hang, so the next safe step is to decode the high-frequency private
slot structures/resource handles rather than mutate another store.

The table mechanism is undocumented by NVIDIA; the public CUDA API documents
the normal `cuLaunchKernel` route, not this table. The implementation is
therefore pinned to the observed driver build and remains a diagnostic hook,
not a production dependency.
