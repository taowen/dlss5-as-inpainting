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

Set `DLSS5_DARK_DUMP_STRUCTS=1` to write 4 KiB snapshots for first/latest
private-slot pointer arguments. A current scan found driver engine-name tables
(`cuda_dldn_engine_*` and `hiluma_engine_*`) but no additional neural CUBIN or
activation pointer beyond the clear utility described below.

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

The composition root signature also receives 16 root constants at parameter 2.
For the 256x256 SDR probe the observed words are:

```text
0x100, 0x100, 0x100, 0x100, 0x0, 0x0, 0x100, 0x100,
0x434b0000, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
```

The first two `uint2` fields are the output size and the `0x434b0000` word is
the observed `203.0` floating-point diffuse-white value; the following `1` is
the HDR mode in this carrier configuration.

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

The current recorder stores ordered per-dispatch snapshots and can dump pointed
CUBIN containers. The optional D3D12 readback exposes the live Neural image and
the exact 15,711,232-byte driver-owned arena before Neural dispatch, but not the
arena's tensor layout or its internal command description. The D3D12 evidence
proves the visible carrier composition pass; the private table evidence proves
the lower driver boundary. The remaining blocker for a bit-exact PyTorch model
is the missing pre-front tensor producer and its driver resource bindings. The previous direct
`STS -> STG` mutation had no valid driver binding and correctly resulted in a
device hang, so the next safe step is to decode the high-frequency private
slot structures/resource handles rather than mutate another store.

The 147 MiB model UAV is now separately capturable. Its snapshot maps all 153
serialized `WEIGHTS_HT` records byte-for-byte after native alignment, including
block0 front-tile offsets `0x2010` and `0x2210`; the coordinate mutation leaves
this buffer unchanged. The dynamic 15.7 MiB UAV is therefore the remaining
activation/command-memory target for exact intermediate recovery.

The table mechanism is undocumented by NVIDIA; the public CUDA API documents
the normal `cuLaunchKernel` route, not this table. The implementation is
therefore pinned to the observed driver build and remains a diagnostic hook,
not a production dependency.

## Complete private-call ring

The recorder now keeps a 1024-entry per-slot call ring in its extended trace
record. `tools/read_dlss5_driver_trace.py` accepts both the legacy record and
the extended record; add `--samples` to print the per-call arguments. A fresh
native 256x256 run recorded the complete high-frequency sequence:

| slot | calls | caller return site | evidence |
|---:|---:|---|---|
| 44 | 532 | `nvwgf2umx+0x1d8ffb` | second argument advances by `0x100` |
| 52 | 532 | `nvwgf2umx+0x1d8fba` | paired metadata call |
| 53 | 1064 | `nvwgf2umx+0x1d8dc2` and `+0x1d8eab` | two descriptor/resource calls per slot-44 item |

This was the first dynamic record of the private resource boundary. A later
NVAPI wrapper resolved the remaining launch boundary: the carrier calls
`NvAPI_D3D12_LaunchCuKernelChain` directly, rather than ordinary
`cuLaunchKernel`. The older ring remains useful because it independently shows
the driver-side resource traffic, while the NVAPI record supplies the exact
function handle, grid/block geometry, parameter pointer, and parameter bytes.

## NVAPI CUBIN launch chain

The optional NVAPI wrapper is enabled with:

```powershell
$env:DLSS5_NVAPI_WRAP_RESULTS = "1"
$env:DLSS5_NVAPI_CAPTURE_LAUNCH = "1"
```

The wrapper identifies these private-interface IDs from the installed NVAPI
interface table:

| ID | observed operation |
|---:|---|
| `0xad1a677d` | `NvAPI_D3D12_CreateCuModule` |
| `0xe2436e22` | `NvAPI_D3D12_CreateCuFunction` |
| `0x24973538` | `NvAPI_D3D12_LaunchCuKernelChain` |
| `0x329fe6e0` | `NvAPI_D3D12_GetCudaMergedTextureSamplerObject` |
| `0x0ddac234` | `NvAPI_D3D12_GetCudaIndependentDescriptorObject` |

On the RTX 5080 256x256 carrier, the wrapper recorded 176 named CUDA
functions and 532 single-kernel launch-chain calls. The three temporal passes
each begin with:

```text
cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8  grid=40,40,1 block=32,1,1 param_size=264
cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8 grid=20,20,1 block=32,1,1 param_size=96
...
cc_tinlayout_fused_post_block_swin_1h_32_fp8     grid=41,41,1 block=32,1,1 param_size=184
cg2r_copy_kernel                                grid=16,16,1 block=16,16,1 param_size=72
```

The exact first pre-block payload contains the live bindings and constants:

```text
offset 0xd0: 0x00000100, 0x00000100, GPU VA 0x1ba16c00
offset 0xe0: GPU VA 0x009e00000                         # 147 MiB model UAV
offset 0xf0: 0x00000140, 0x00000140, GPU VA 0x1bd36c00
offset 0x100: 0x000000a0, 0x000000a0
```

The first eight bytes are also the exact CUDA object handles used by the
pre-front texture path (`0x80200009801` in the 256x256 control). The wrapper
maps those handles back to the D3D12 CPU descriptors and records the resource
dimensions and raw 32-byte descriptor payloads. The later temporal pass adds
the handles `0x80000009805` and `0x80200009806` for the temporal resources.

The capture can be joined into a replay manifest without interpreting or
rounding any parameter data:

```powershell
python tools/analyze_dlss5_nvapi_launch.py `
  --runtime <capture-runtime>
```

The manifest retains the exact parameter bytes, SHA-256, little-endian words,
function names, launch geometry, descriptor-object handles, and raw descriptor
hashes. Capture outputs stay in the ignored `.native-build` tree because their
addresses are process-specific; the parser and the add-on are the reproducible
artifacts committed to the repository.

## Bit-exact PyTorch-facing carrier

Ordinary PyTorch layers remain a semantic translation and are explicitly not
bit exact. The repository now also exposes the native CUBIN carrier through an
inference-only `torch.nn.Module`. This is the honest way to provide a PyTorch
call boundary while preserving the proprietary SM120 FP8/QMMA/SASS execution:

```powershell
python tools/verify_dlss5_bit_exact.py --runtime <prepared-runtime>
```

The verifier runs an independent native two-frame golden process and a second
process through `DLSS5BitExactCarrier`, then compares the raw RGBA16F bytes.
The local RTX 5080 result is:

```text
bytes       524288
native SHA  1fe38ab7fe6b85b8352fd11a48b15b32c2713029785baa7ee9a9ba934f38f1e3
carrier SHA 1fe38ab7fe6b85b8352fd11a48b15b32c2713029785baa7ee9a9ba934f38f1e3
byte_equal  true
```

This proves bit equality for the pinned native carrier and input contract. It
does not claim that `tools/dlss5_pytorch.py` has become a portable translation
of every proprietary instruction; replacing the native CUBIN with ordinary
PyTorch arithmetic would require a separate numerical-equivalence proof.
