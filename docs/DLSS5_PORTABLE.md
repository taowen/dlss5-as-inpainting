# DLSS5 portable PyTorch model

The repository now has two different PyTorch-facing implementations:

* `DLSS5Graph` is the 71-block, weight-backed semantic translation. It is
  useful for studying the recovered graph and can run on ordinary PyTorch
  devices, but its private TEX/cat front producer is not fully recovered.
* `DLSS5PortableModel` is the deployable cross-device approximation. It uses
  only `Conv2d`, `SiLU`, tensor concatenation, interpolation, and clamping.
  It has no CUDA, FP8, native DLL, or custom operator dependency.

The portable model starts as an identity residual model. This is intentional:
an unresolved front feature must not be replaced by a random tensor that turns
valid RGB into saturated or negative pixels. Native RGBA16F input/output pairs
can be used to train its small residual head:

```powershell
$n = 'C:\path\to\native\bin'
python tools\distill_dlss5_portable.py `
  --pair "$n\full_checker.rgba16f.bin=$n\full_checker.out.rgba16f.bin" `
  --pair "$n\full_red_ramp.rgba16f.bin=$n\full_red_ramp.out.rgba16f.bin" `
  --pair "$n\full_green_ramp.rgba16f.bin=$n\full_green_ramp.out.rgba16f.bin" `
  --pair "$n\full_blue_ramp.rgba16f.bin=$n\full_blue_ramp.out.rgba16f.bin" `
  --pair "$n\worker_like_linear.rgba16f.bin=$n\worker_like_linear_out.rgba16f.bin" `
  --output DLSS5-extracted\dlss5_pytorch_portable_v1.pt `
  --device cuda --steps 400
```

The checkpoint contains only the portable residual head. It does not contain
or redistribute NVIDIA binaries or `WEIGHTS_HT.bin`.

Run it on CPU, NVIDIA CUDA, AMD ROCm, Intel, or another backend supported by
the installed PyTorch build:

```powershell
python tools\run_dlss5_portable.py `
  DLSS5-extracted\dlss5_pytorch_portable_v1.pt `
  input.rgba16f.bin --width 256 --height 256 --device cpu `
  --output portable.rgba16f.bin
```

Or use it directly:

```python
from tools.dlss5_portable import load_portable_checkpoint

model = load_portable_checkpoint("DLSS5-extracted/dlss5_pytorch_portable_v1.pt", device="cuda")
rgb = rgb_nchw.float()                 # [N, 3, H, W]
output = model(rgb=rgb)                # [N, 3, H, W], finite RGB in [0, 1]
output = model(rgb, depth=depth, history=previous, motion=mv, control_mask=mask)
```

Depth, history, motion, and `control_mask` are optional and are resized or
zero-filled to the RGB contract. The model preserves spatial dimensions. A
zero mask returns the source; a one mask enables the learned residual.

Validation on the local RTX 5080 used ten 256x256 native pairs. The identity
baseline RGB MAE was `0.02931`; the 250-step portable residual reached
`0.01593`. The exported checkpoint was loaded and executed on both CPU and an
RTX 5080, with finite `[1, 3, 256, 256]` output in `[0, 1]`. These are quality
and portability checks, not a bit-exact claim.

For strict equality on the original NVIDIA path, keep using
`DLSS5BitExactModel`; the portable model is deliberately a separate
approximation and does not call that native carrier.
