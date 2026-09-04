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
recover the front feature tensor. The next safe experiments are same-length
or metadata-adjusted patches to the pre `TEX` coordinate/math path, followed by
output A/B runs; the extractor and patcher now provide the required reversible
boundary. A first coordinate experiment changed the `FADD R5,R5,0.5` constant
to `0.0`. Its patched DLL ran successfully; the isolated first checker output
was unchanged, while the history-after-color output changed with MAE
`0.0038043`, RMSE `0.0066071`, and `149,982/262,144` elements above `1e-3`.
This is consistent with a temporal/state-dependent coordinate path and is not
yet a recovered texture-lane formula.
