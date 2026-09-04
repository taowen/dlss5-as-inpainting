# Example data

`assets/input/` contains the downloaded public-domain/CC0 source images.
`assets/normalized/` contains the 256x256 sRGB inputs used by the experiment.
`assets/conditions/` contains controlled depth previews.
`cases/native/` contains native output PNGs and the numerical manifest.

The raw RGBA16F/R32F/RG16F files are generated under `examples/.work/` and
ignored because they are intermediate contract data. Regenerate everything
with a local native harness:

```powershell
python tools\experiments\run_dlss5_image_cases.py `
  --harness C:\path\to\dlss5_eval.exe --width 256 --height 256
```

See [`docs/DLSS5_EXPERIMENTS.md`](../docs/DLSS5_EXPERIMENTS.md) for the
interpretation of the image, depth, history, and motion cases.
