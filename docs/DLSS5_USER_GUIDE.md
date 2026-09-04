# DLSS5 model usage guide

## Which model should be used?

| Path | Intended use | Requirements | Exactness |
|---|---|---|---|
| `DLSS5BitExactModel` | NVIDIA control and byte-level verification | native runtime + compatible NVIDIA GPU | bit exact for the pinned contract |
| `DLSS5Graph` | graph/layout research and future front recovery | PyTorch + extracted local weights | semantic/non-exact |
| `DLSS5PortableModel` | deployment on other GPU vendors or CPU | ordinary PyTorch only | native-distilled approximation |

The portable model is the safe default for a general application. Its final
residual layer is initialized as identity, and the checked-in checkpoint is a
small native-distilled residual head. It never consumes the unresolved SASS
front tensor, CUDA descriptors, or NVIDIA DLLs.

## Portable API

```python
from dlss5.portable import load_portable_checkpoint

model = load_portable_checkpoint(
    "models/dlss5_pytorch_portable_v1.pt",
    device="cuda",                 # or "cpu", "cuda:1", ROCm, etc.
)
rgb = rgb_nchw.float()             # [N, 3, H, W], normally in [0, 1]
output = model(rgb=rgb)            # [N, 3, H, W], finite and clamped to [0, 1]
```

Optional conditions are accepted:

```python
output = model(
    color=rgb,
    depth=depth,                    # optional [N, C, H', W']
    history=previous_rgb,            # optional previous output
    motion=motion_vectors,           # optional [N, C, H', W']
    control_mask=mask,               # 0=source, 1=learned residual
)
```

Conditions are resized to the RGB rectangle and missing channels are zero
filled. The portable model preserves spatial dimensions. For an image export,
convert the returned RGB tensor to the desired file color space; the model
output itself is numeric RGB, not a PNG color-management promise.

To train a new portable head from local native pairs:

```powershell
python tools\distill_dlss5_portable.py `
  --pair native\input.rgba16f.bin=native\output.rgba16f.bin `
  --output models\my_portable_v1.pt `
  --device cuda --steps 400
```

The pair format is tightly packed little-endian RGBA16F. The distiller uses
the first three channels, clamps training values to `[0,1]`, and stores only
ordinary convolution weights.

## Native input/output contract

The native harness path is a different interface:

| Plane | Format | Meaning |
|---|---|---|
| color | RGBA16F | linear RGB plus alpha; the current experiments use alpha=1 |
| depth | R32F | reversed-Z-style depth plane in the host contract |
| motion | RG16F | per-pixel motion vector in the host's pixel/coordinate convention |
| output | RGBA16F | linear RGB plus alpha=1 |

The native path is temporal. `reset=1` seeds or clears the history state;
`reset=0` evaluates with the previous frame. A motion vector is not an image
enhancement knob: it tells the temporal resolver where the previous-frame
sample came from, so it affects reprojection, history rejection, and
disocclusion handling. Its sign and scale must be calibrated against the host
renderer; do not assume that a vector copied from a different API has the same
direction convention.

For display previews, linear native RGB must be converted to sRGB. The image
case runner does this conversion when writing PNGs, while all numeric metrics
remain in linear RGB.

## Semantic graph caveat

`DLSS5Graph` loads the recovered 71-block body and can execute on CPU or
non-NVIDIA backends with the pure-tensor E4M3 fallback. Its private pre-block
texture/cat producer is still unresolved. Feeding the current SASS candidate
is therefore an experiment, not the default application path. Use the portable
checkpoint when the requirement is a correct, stable image on another GPU.
