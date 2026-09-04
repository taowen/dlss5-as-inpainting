# DLSS5 as Inpainting

Research repository for understanding the DLSS5/Feature18 neural-rendering
contract, translating the recovered arithmetic into PyTorch, and preparing a
portable image-to-image/inpainting pipeline.

This repository contains three deliberately separate execution paths:

1. **Native carrier** — the original NVIDIA CUBIN/runtime path exposed through
   `DLSS5BitExactModel`. It is the only path with a bit-exact claim and needs a
   locally prepared native runtime on a compatible NVIDIA GPU.
2. **Semantic graph** — the 71-block `DLSS5Graph` translation. It loads the
   recovered `WEIGHTS_HT.bin` layout and is useful for research, but the
   private texture/front-feature producer is not fully recovered.
3. **Portable model** — `DLSS5PortableModel` and
   `models/dlss5_pytorch_portable_v1.pt`. It uses ordinary PyTorch operators and
   runs on CPU, CUDA, ROCm, Intel, or another supported backend. It is a
   native-distilled approximation, not a CUBIN clone.

## Quick start

Install the source package for normal imports:

```powershell
python -m pip install -e .
```

For the image experiment tools, install the optional NumPy dependency too:

```powershell
python -m pip install -e ".[research]"
```

For Distill-Any-Depth small depth inference, add the depth extra:

```powershell
python -m pip install -e ".[research,depth]"
```

Run the portable checkpoint on an RGBA16F contract image:

```powershell
python tools\run_dlss5_portable.py `
  models\dlss5_pytorch_portable_v1.pt `
  input.rgba16f.bin --width 256 --height 256 --device cpu `
  --output portable.rgba16f.bin
```

Run the semantic graph smoke test:

```powershell
python tools\dlss5_pytorch.py --self-test
```

The full model contract and the observed native behavior are documented in
[`docs/DLSS5_USER_GUIDE.md`](docs/DLSS5_USER_GUIDE.md) and
[`docs/DLSS5_EXPERIMENTS.md`](docs/DLSS5_EXPERIMENTS.md). The stereo/inpainting
experiment is documented in [`docs/DLSS5_STEREO_EXPERIMENTS.md`](docs/DLSS5_STEREO_EXPERIMENTS.md).

## Repository layout

```text
src/dlss5/                 reusable model source package
  graph.py                 71-block semantic graph and forward path
  blocks.py                Swin, Split-Swin, ViT, and attention blocks
  loaders.py               WEIGHTS_HT-to-module bindings and audit report
  weights.py               outer weight parser and tensor decoders
  layouts.py               recovered offsets, shapes, and block maps
  ops.py                   E4M3, front candidates, and layout operations
  portable.py              backend-independent distilled model

tools/                     development and reverse-engineering tools
  dlss5_pytorch.py         backwards-compatible source/CLI facade
  dlss5_portable.py        backwards-compatible portable facade
  experiments/             public image download and native case runners
  analyze_*.py              capture, SASS, storage, and launch analysis
  probe_*.py                dynamic probes and regression checks
  run_*.py / verify_*.py   execution and validation utilities

models/                    small portable checkpoint suitable for distribution
examples/
  assets/input/            downloaded source images
  assets/normalized/       256x256 display inputs
  assets/conditions/       controlled depth proxy previews
  assets/depth/            Distill-Any-Depth small relative depth outputs
  cases/native/            native output PNGs and JSON metrics
  cases/stereo_inpainting/ stereo warp/fill/DLSS5 comparison PNGs and metrics
docs/                      user guide, experiment report, and evidence notes
third_party/               optional source repositories as git submodules
DLSS5-extracted/           local NVIDIA-derived extraction; intentionally ignored
runtime_probe_output/      local probe output; intentionally ignored
```

`src/dlss5` is the code intended for reuse. `tools` contains experiments and
development instrumentation; it is not part of the model's numerical API.

## Reproducing the image cases

The fixtures come from Wikimedia Commons sources with public-domain or CC0
status. Re-download them and record hashes with:

```powershell
python tools\experiments\download_public_examples.py
```

A native harness is not redistributed. With a locally prepared
`dlss5_eval.exe`, regenerate the PNGs and manifest:

```powershell
python tools\experiments\run_dlss5_image_cases.py `
  --harness C:\path\to\dlss5_eval.exe --width 256 --height 256
```

Generate relative depth and run the stereo experiment:

```powershell
python tools\experiments\run_distill_any_depth_small.py --device cuda
python tools\experiments\run_stereo_inpainting_cases.py `
  --harness C:\path\to\dlss5_eval.exe --max-disparity 16 --plane-shift 16
```

The runner uses linear-light RGBA16F inputs, controlled R32F depth proxies,
and RG16F motion vectors. Native binary contracts are written below
`examples/.work/` and are ignored; reviewable PNGs and the relative-path
manifest are kept under `examples/cases/native/`.

## Future work

The next layers are intentionally not hidden behind the current model API:

- replace the synthetic depth proxy with a depth estimator and calibrate its
  reversed-Z mapping;
- add a documented temporal-state object and motion-vector reprojection tests;
- validate ControlMask and inpainting-specific masks against native outputs;
- validate the no-op ControlMask binding with launch/resource telemetry, then
  repeat the stereo test with the actual HoleMask;
- expand native-pair distillation beyond the current public fixtures;
- recover the remaining private front producer before making any stronger
  native-equivalence claim.
