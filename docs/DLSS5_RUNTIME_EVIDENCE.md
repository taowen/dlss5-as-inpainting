# DLSS 5 runtime evidence

This note records a reproducible black-box run of the native DLSS 5 worker on
the local RTX 5080. It validates the runtime call chain and input sensitivity;
it does not by itself expose the internal shader dispatch list or prove the
static 71-block graph.

## Run

- Date: 2026-09-03 21:14–21:16 (Asia/Shanghai; repeated run matched exactly)
- GPU: NVIDIA GeForce RTX 5080, driver 616.56
- Worker: `DLSS.5.Visual.Enhancer.v5.0` `bin/runtime/host/nvngx.dll`
- Model: `nvngx_dlssnr.dll`, SHA-256
  `6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927`
- Dimensions: 256x256 input and output
- Dependencies: Python standard library only; no PyTorch or CUDA toolkit

The worker was started with `nvngx.dll --video`. Its setup response was
successful (`ok=1`, `ngx_result=1`) and negotiated 256x256 output with an
optimal render range of 253x253..256x256. The host log reports an RTX 5080
adapter, a hidden D3D12 carrier swapchain, successful `NVSDK_NGX_D3D12_Init`,
and three carrier evaluations for two delivered frames.

## Behavioral A/B results

The probe uses deterministic RGBA8 patterns and writes every returned frame.
The values below are mean absolute byte differences over the complete RGBA8
payload (262,144 bytes per frame).

| Test | Difference | Interpretation |
| --- | ---: | --- |
| Two-frame history vs isolated current frame | 7.2837 | Previous-frame state affects frame 2 |
| Zero Motion Vector vs shifted Motion Vector | 3.7315 | Motion Vector input affects frame 2 |
| Intensity 1.0 vs Intensity 0.0 | 7.5029 | Intensity control affects output |
| Model output vs current input | 7.6917 | Output is not a byte copy of the input |

These are dynamic behavioral observations of the complete worker/NGX/D3D12
path. They are evidence that temporal state, motion vectors, and the exposed
intensity control reach the runtime, but they are not a per-layer GPU trace.

## Reproduce

```powershell
python tools\dlss5_worker_probe.py `
  --worker C:\path\to\DLSS.5.Visual.Enhancer.v5.0\bin\runtime\host\nvngx.dll
```

The generated manifest, stderr logs, and raw output frames are in
`runtime_probe_output/worker_probe/`. The native worker and its host-side DLLs
must remain together in the extracted runtime directory.

## Independent native harness cross-check

On 2026-09-04, the public C++ harness from
[`criso2hd-alt/DLSS5-Image-Converter`](https://github.com/criso2hd-alt/DLSS5-Image-Converter)
was built locally against the public NVIDIA DLSS SDK headers and run with the
same locally available runtime files. Its probe reported:

- RTX 5080, driver 616.56
- `dlss_available=1`
- `reshade_proxy_loaded=1`
- `neural_addon_loaded=1`
- `dlssnr_module_loaded=1`
- `test_evaluation=ok`

The harness also returned a full 256x256 `RGBA16F` output. Feeding it the same
deterministic pattern used by the worker, after converting the RGBA8 values to
linear RGBA16F, produced a linear-domain correlation of `0.9979` against the
worker's RGBA8 result and mean absolute difference `0.0120`. This independently
cross-checks the native worker path and color contract; it does not expose
per-layer tensors.

An explicit `DLSSNR.ControlMask` resource added to the ordinary DLSS parameter
block produced identical mask=0 and mask=255 outputs in this harness. That
means the field was not propagated through this ordinary-DLSS hook setup; it is
not evidence that the native Feature 18 implementation lacks ControlMask.

## Full-precision temporal A/B

The same independent harness was then driven with its raw `RGBA16F` protocol.
The first frame was a deterministic linear test image, followed by a checker
pattern. The final checker output was compared with an isolated checker run
whose first frame was reset. The temporal-history effect was:

- mean absolute difference: `0.026169`
- RMSE: `0.061115`
- changed values above `1e-3`: `160,773 / 524,288`

Repeating that two-frame experiment with a constant `(+4, 0)` pixel motion
vector on the second frame instead of zero motion produced a distinct output:

- mean absolute difference versus zero-MV: `0.008876`
- RMSE: `0.024945`
- changed values above `1e-3`: `138,016 / 524,288`

Both results are before RGBA8 quantisation and the alpha channel remained 1.0.
They are stronger dynamic evidence than the worker's byte-level result: the
native path's temporal state and motion-vector inputs affect the full-precision
neural output, not only the presentation conversion.

The experiment is now automated by the standard-library-only
`tools/dlss5_fp16_harness_probe.py` driver. For example, with the locally built
external harness:

```powershell
python tools\dlss5_fp16_harness_probe.py `
  --harness C:\path\to\dlss5_eval.exe `
  --output runtime_probe_output\fp16_harness
```

The 2026-09-04 run of this fixed contract reported history MAE/RMSE
`0.042045/0.060939` and shifted-MV-vs-zero-MV MAE/RMSE `0.007987/0.016926`.
The raw frames and manifest are ignored local evidence; the harness and all
runtime DLLs remain external.

The temporary build, SDK headers, and locally staged runtime are outside this
repository. NVIDIA runtime files and third-party binaries are not committed.
