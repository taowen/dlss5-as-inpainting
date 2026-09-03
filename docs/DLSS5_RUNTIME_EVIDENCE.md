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
