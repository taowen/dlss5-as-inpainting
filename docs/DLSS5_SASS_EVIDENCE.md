# DLSS5 block0 SASS evidence

The `sm_120` block0 CUBIN was disassembled with NVIDIA `nvdisasm` 12.9.88.
The extraction is reproducible with:

```powershell
python tools\dlss5_disassemble_cubin.py `
  --nvdisasm C:\path\to\nvdisasm.exe `
  --cubin cubins\fatbin_00\fatbin_00_0xdf0e0.4.sm_120.cubin `
  --function cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8 `
  --sass-output runtime_probe_output\pre_front.sass.txt `
  --summary-output runtime_probe_output\pre_front.sass.json
```

The extracted function contains 13 texture instructions before the body,
including five `0x7` texture reads. It writes two `STS.128` shared
memory records at offsets `0` and `-0x400`, then loads the two serialized FP16
front tiles at record offsets `+0x2010` and `+0x2210`. The first 16 HMMA
instructions consume those shared features and the loaded tiles.

The front sequence also contains `HADD2` with `-0.5`, `HMUL2`, FP16 packing and
`PRMT`. Therefore the unresolved PyTorch input is a texture-derived 16-half
tile with one alignment lane, not a plain 15-channel RGB copy. The exact
coordinate/filter arithmetic and the logical ordering of the packed feature
lanes remain the next reconstruction target.

An independent public reverse-engineering note adds a useful structural
constraint: it describes the pre-block producer as three generated FP16 lanes,
a constant `1.0`, and four sampled texture lanes. This is consistent with the
SASS hash/math prefix and the later `0x7` texture reads in the packing path.
It does not yet specify whether each lane is a scalar or packed
half2, nor the exact lane order, so it is used as a layout constraint rather
than copied as an implementation. See
[`madebyollin/dlss_5_model_architecture.md`](https://gist.github.com/madebyollin/55c703a34bf90962844edcd68d04e32e).

There is an additional non-RGB dependency before the stores: the SASS mixes an
integer hash path with `MUFU.LG2`, `MUFU.SQRT`, `MUFU.SIN` and `MUFU.COS`, then
converts the results to FP16 before the final packing. This is consistent with
a deterministic Box--Muller-style noise pair, not with an external random
stream. The PyTorch producer will therefore need the same coordinate/seed
hash and half rounding before its 15-channel projection can be compared with
the native golden output.

## Toolchain cross-check and next capture route

On 2026-09-04 the same function was disassembled with the official CUDA
`nvdisasm` 12.9.88 and 13.3.73 binaries. Both produced 7,689 identical text
lines and the same SASS text SHA-256, so the current front discrepancy is not
caused by a missing instruction in this function. Future SM120 audits should
still pin the newer tool: a public Blackwell investigation demonstrated that
an older disassembler can silently omit a newly added instruction
([reproduction](https://github.com/jethac/ptxas-clmad-miscompile)).

The useful static passes are `nvdisasm --cfg`/`--bbcfg` for branch regions,
`--print-life-ranges` for register lifetime, and
`--print-instruction-encoding`/`--print-raw` for patchable instruction bytes.
NVIDIA documents these views in the [CUDA Binary Utilities guide](https://docs.nvidia.com/cuda/cuda-binary-utilities/).
The PTX guide gives the authoritative warp-fragment mapping for
`mma.m16n8k16`; in particular, the floating-point B fragment uses
`row = 2*(lane % 4) + (i & 1)` with the second pair offset by 8 and
`col = lane >> 2`. That supports the current physical-tile decoder, but does
not prove the producer's spatial reorder.

The next high-value dynamic route is to capture the *actual* JIT module rather
than search the PE for an ELF blob. CUPTI's module-resource callback exposes a
JIT-loaded CUDA binary, and its PC records include `cubinCrc`, `pcOffset`, and
`functionName` for matching a sampled PC back to a function. See the [CUPTI
SASS source-correlation documentation](https://docs.nvidia.com/cuda/cupti/main/main.html).
If the D3D12/NGX path is visible to CUPTI, this gives the exact runtime CUBIN;
if not, the fallback is a Linux CUDA host or NVBit experiment. [NVBit](https://github.com/NVlabs/NVBit)
can inspect and inject calls before/after SASS instructions in a loaded
precompiled kernel, but its published requirements are Linux-oriented and it
is not yet a drop-in solution for this Windows D3D12 carrier.

For offline SM120 patching, [cubit](https://github.com/kacper-daftcode/cubit)
is now the most relevant public tool: it advertises SM120 disassembly,
round-trip assembly, and ELF-preserving cubin patching. It should first be
validated on our extracted CUBIN with a no-op round-trip, then used to insert
a debug store only in a disposable copy. Any patched kernel must be validated
with an independent disassembly and a 5080 numeric probe; SASS scheduling and
hidden ABI metadata are part of correctness.

The repository pins [cubit](https://github.com/kacper-daftcode/cubit) as a
submodule and provides:

```powershell
.\tools\build_cubit.ps1
.\tools\verify_dlss5_cubit.ps1
```

The verifier targets the block0 pre kernel rather than claiming that every
function in the fatbin is supported. On the local CUBIN it re-encoded 3,792 of
3,792 instructions and preserved all 2,789,376 bytes byte-for-byte. A
disposable seed experiment changed only one instruction byte at file offset
`0x1A15A4` and produced a valid patched CUBIN; it was not installed into the
DLSS runtime.

## CUPTI capture result on this machine

The repository now contains `tools/cupti_capture`, a small Windows DLL that
uses `CUDA_INJECTION64_PATH` and records module-load/kernel-launch callbacks
without linking against a full CUDA Toolkit. It was built against the official
CUDA 13.3 CUPTI/runtime wheels and tested on the RTX 5080 (driver 616.56): a
PyTorch CUDA process produced four module callbacks, with captured CUBIN sizes
`735,536`, `13,616`, `2,421,144`, and `7,485,344` bytes.

The same injection was tested against the working DLSS5 D3D12/ReShade harness
at its valid 256² contract size. `cuptiSubscribe` returned success and the
native A/B run completed with the same history and motion metrics, but no
module-load callback was emitted. A separate 64² attempt returned
`DXGI_ERROR_DEVICE_REMOVED` (`0x887a0005`) even without the capture DLL, so that
size is a harness limitation rather than evidence about CUPTI. The current
result therefore says that this D3D12/NGX carrier is not exposing DLSS5's
runtime work as CUPTI CUDA modules, not that the capture implementation is
broken. The capture remains a verified CUDA-client probe and a template for a
future in-process graphics/driver capture route.

## Embedded CUBIN extraction and live patch

The earlier ELF-only search was incomplete. The current `nvngx_dlssnr.dll`
contains 15 `50 ED 55 BA` bundle headers. Each bundle contains four
concatenated Zstandard frames; the first four decode to sm_75, sm_86, sm_89,
and sm_120 ELF CUBINs. `tools/extract_dlss5_embedded_cubins.py` extracts all
68 Zstandard frames, of which 60 are CUBINs, and compares their hashes with
the loose files in `cubins/`. On the local DLL all 60 CUBIN hashes match.

The Zstandard frame parameters are reproducible with level 5,
`write_content_size=True`, `write_checksum=False`, and `write_dict_id=False`.
That makes same-length in-place experiments possible. For example:

```powershell
python tools\patch_dlss5_embedded_cubin.py bin\nvngx_dlssnr.dll `
  --bundle 0 --gpu sm_120 --cubin-offset 0x1a15a7 --byte-value 0x00 `
  --output runtime_probe_output\nvngx_dlssnr_seed_patch.dll
```

This changes the pre-kernel seed multiplier from `-0x72594cbd` to `0xa6b343`
at SASS address `0x03a0`, keeps the compressed frame at `512,316` bytes, and
keeps the DLL size unchanged. The patched DLL was placed in an isolated native
runtime and executed on the RTX 5080 at 256². The patched-vs-baseline native
isolated output changed with MAE `0.0062039`, RMSE `0.0102768`, and
`155,199/262,144` half elements above `1e-3`; the history output changed with
MAE `0.0050361` and `168,610/262,144` elements above `1e-3`. This is the first
dynamic causal proof that a modification inside the actual DLSS5 pre CUBIN
changes the neural output.

The patch is intentionally not installed into `bin/` and does not claim to
recover the front feature tensor. The automated matrix in
`tools/probe_dlss5_front_mutations.py` now runs the validated two-frame
`color(reset=1) -> checker(reset=0)` sequence, disables only the private CUDA
hook, and observes the latest Neural dispatch through the D3D12 readback.
This is important: the one-frame result is insensitive to the coordinate
mutation, while the temporal result is causal. The first coordinate experiment
changed the `FADD R5,R5,0.5` constant to `0.0`; its history-after-color output
changed with MAE `0.0038043`, RMSE `0.0066071`, and `149,982/262,144` elements
above `1e-3`.

A second live experiment changed the first pre `TEX` component mask from
`0x7` to `0x1` while preserving the rest of the instruction. The patched DLL
still initialized and evaluated at 256² on the RTX 5080, but its RGB output
was exactly zero (alpha remained one). This confirms that the full RGB result
of that texture read participates in the front feature assembly; a single
component cannot serve as the PyTorch front input. The experiment also
validates that the patcher can change a SASS modifier/control field and keep
the compressed bundle loadable.

Finally, each of the five later `TEX 0x7` reads in the front packing region was
masked to `0x1` independently and evaluated with the same 256² temporal
contract. Relative to the unmodified DLL, the history-after-color output was:

| SASS PC | output MAE | changed half elements | reading |
|---|---:|---:|---|
| `0x1590` | `0.0` | `0` | inactive for this contract/path |
| `0x15c0` | `0.0` | `0` | inactive for this contract/path |
| `0x15d0` | `0.175637` | `196,414` | active texture contribution |
| `0x15e0` | `0.261709` | `196,608` | active texture contribution |
| `0x15f0` | `0.0` | `0` | inactive for this contract/path |

The initial `TEX` at `0x0620` remains active: changing its mask from `0x7` to
`0x1` makes the native RGB output exactly zero. This narrows the PyTorch front
reconstruction target to the initial read plus the two active later reads for
the current path; static instruction presence alone is not enough to choose
the feature lanes.

The same runner also tested four additional coordinate instructions in the
actual pre kernel. On the RTX 5080 and the same two-frame temporal contract,
`FADD R6,R5,0.5 -> 0.0` changed the output with MAE `0.0045224`, RMSE
`0.0077229`, maximum absolute error `0.0512695`, and `153,539/262,144`
elements above `1e-3`; `FADD R48,R43,0.5 -> 0.0` changed it with MAE
`0.0036740`, RMSE `0.0060694`, maximum absolute error `0.0400391`, and
`157,511/262,144` changed elements. The tested `FADD R37/R38` points were
inactive for this carrier. In the initial nine-case matrix, the captured original
and Neural textures were byte-identical to baseline, so the visible delta is
downstream of the two legacy readback slots. The full descriptor scan identifies
the changed resources as root0[2] and root0[5]/root1[0].

The optional `--capture-all-neural` mode then read every resolved descriptor
around each Neural dispatch. It showed why the two legacy captures were
unchanged: they are root0[0] and root0[1]. In the latest dispatch, the
`FADD R5` mutation changed root0[2] with MAE `0.0029757` and changed
root0[5]/root1[0] with MAE `0.0038043`; the `TEX 0x15d0` mutation changed
root0[2] with MAE `0.2145830` and root0[5]/root1[0] with MAE `0.1937808`.
This is the first direct evidence that the mutated pre path reaches hidden
Neural resources. The full scan is reproducible by adding
`--capture-all-neural` to the command above.

Adding `--capture-all-dispatches` confirms the timing: the earlier Original
dispatches and their descriptor resources remain unchanged; only the latest
Neural `root0[2]` changes first, followed by the final-output aliases
`root0[5]/root1[0]`. The runner now parses the dispatch labels from the add-on
log, so a trailing Original dispatch emitted during `WRITE` cannot be mistaken
for the latest Neural tensor.

With `--capture-before-neural`, the same latest `root0[2]` was read immediately
before the Neural dispatch. The `FADD R5` mutation changed that pre-dispatch
resource with exactly the same MAE `0.0029757`, RMSE `0.0043712`, and 158,951
changed elements observed after the dispatch. This proves that the mutation is
already present at the private CUBIN/driver boundary; the D3D12 Neural shader
does not create the first difference.

The optional `--capture-driver-buffers` mode also captures the driver-owned UAV
at GPU VA `0x1ba00000` (`0xefbc00` bytes). The first two temporal snapshots are
byte-identical between baseline and mutation; the third changes in 9,278,345
bytes beginning at `0x16c0c`. A private slot14 mapping snapshot independently
reports suballocation offset `0x16c00`, the same GPU VA, and the same size.
This is the first raw workspace capture associated with the private CUBIN
submission. Its internal tensor layout is still to be decoded before it can be
used as a PyTorch input or exact intermediate.

The capture hook now supports `--capture-driver-buffers-all-dispatches`, which
records the same arena immediately after each Original dispatch and immediately
before each Neural dispatch. On the RTX 5080, the coordinate mutation produced
six ordered snapshots for the three temporal passes: all three `after_original`
snapshots and the first two `before_neural` snapshots were byte-identical to the
baseline, while only the third `before_neural` snapshot changed. That final
snapshot changed exactly 9,278,345 bytes, from `0x16c0c` through the end of the
15,711,232-byte resource. This separates the mutation's first observable effect
from the D3D12 Neural consumer and is now recorded as raw-byte SHA-256 plus
changed-range metadata in `report.json`.

The same run logs every committed D3D12 buffer creation. The only tracked
driver-owned arena remains GPU VA `0x1ba00000`, size `0xefbc00`, with UAV flags;
the other 15.7 MiB buffers appearing in the log are the readback resources
created by the capture hook itself (COPY_DEST, distinct GPU virtual addresses).

The explicit `--capture-model-buffers` mode then captured the 147,719,680-byte
UAV at GPU VA `0x9e00000`. `tools/analyze_dlss5_model_buffer.py --strict`
locates all 153 `WEIGHTS_HT` records in that GPU snapshot and compares every
record byte, not just a checksum or a sample. The result is `bit_exact=true`
with 153/153 full-record matches; the buffer is `0x5ece` bytes larger than the
serialized resource because of the native record alignment. The block0 front
tiles map to buffer offsets `0x2010` and `0x2210`, exactly the offsets used by
the sm_120 `LDG` instructions. A coordinate mutation leaves this model buffer
byte-identical and changes only the separate 15.7 MiB dynamic arena, proving
that the remaining mismatch is not caused by weight extraction or weight
placement.

Component-level masks provide an additional lane constraint. With the same
carrier and temporal sequence, masking each TEX to two components produced:

| SASS PC | mask | final MAE | hidden MAE | interpretation |
|---|---:|---:|---:|---|
| `0x0620` | `0x3/0x5/0x6` | `0.227164/0.225732/0.229032` | `0.286784/0.279946/0.285785` | every two-component subset remains nonzero |
| `0x15d0` | `0x3/0x5/0x6` | `0.180217/0.188974/0.177900` | `0.197118/0.209432/0.194973` | all three subsets remain active, with different weights |
| `0x15e0` | `0x3/0x5/0x6` | `0.261709/0.261709/0.261709` | `0.382358/0.382358/0.382358` | any missing component collapses RGB to zero |

The corresponding single-component masks (`0x1/0x2/0x4`) collapse the initial
read and `0x15e0` to the same zero-RGB result; `0x15d0` remains visibly active
but with a different hidden tensor for each component. This rules out a
single-channel front texture and narrows the PyTorch lane recovery to the
three-component outputs of these reads plus their exact coordinate mapping.

The matrix is reproducible with:

```powershell
python tools\probe_dlss5_front_mutations.py `
  --runtime-template <clean-runtime> `
  --harness <clean-runtime>\dlss5_eval.exe `
  --addon .native-build\reshade-capture\Release\dlss5_reshade_capture.addon64
```

## Graphics-side capture boundary

`third_party/reshade` is pinned as a reproducible source dependency and
`tools/build_dlss5_reshade_capture.ps1` builds
`dlss5_reshade_capture.addon64`. The add-on records D3D12 resource creation,
resource-view creation, GPU virtual addresses for buffers, and the available
ReShade command-list events. On the RTX 5080 it loaded beside the working
carrier and recorded the internal buffer allocation set, including the
256²/512²/6 MiB/5 MiB scratch buffers used around the neural pass.

The same run produced no ReShade `dispatch`, pipeline, descriptor-table, or
push-constant callbacks for the DLSS5 neural work. This is an important
boundary result: the carrier allocates/interposes resources through D3D12, but
the actual NGX neural submission is below the ordinary ReShade command-list
event layer. Consequently an add-on cannot yet append a legal copy of the
pre-block shared tile, and the earlier `STS -> STG` experiment faulted the GPU
at warm-up because it wrote through an address/descriptor that is not a valid
telemetry target at that point.

The remaining exact-conversion task is therefore a driver/NGX-level capture
or a valid host-provided scratch binding. The PyTorch graph remains executable
on the 5080, but its default RGB path is explicitly a zero-front fallback; it
must not be described as bit-exact DLSS5 or as a recovered image-to-image
generator until the pre texture/front tensor is captured or its ABI is proved.
