# Experiment tools

These scripts are development-only and are separate from the reusable model
package in `src/dlss5/`.

| script | purpose |
|---|---|
| `download_public_examples.py` | fetch the four public-domain/CC0 fixtures and record hashes |
| `run_dlss5_image_cases.py` | build color/depth/motion contracts, invoke a local native harness, and write PNG/JSON evidence |
| `run_distill_any_depth_small.py` | run the official 24.8M-parameter Distill-Any-Depth small checkpoint and write relative R32F depth |
| `run_stereo_inpainting_cases.py` | forward-splat a left eye, make a simple fill, run native DLSS5, and compare hole/valid regions |

The image runner requires a locally prepared `dlss5_eval.exe` and `ffmpeg`.
Native binaries are intentionally not stored in this repository. Temporary
contract planes go under ignored `examples/.work/`; reviewable outputs go under
`examples/cases/native/`.

The depth script downloads the checkpoint through Hugging Face on first use;
the 99 MB model is cached outside the repository. The stereo report includes
both the stock constant-mask result and a separate spatial-mask probe; the
temporary spatial harness still produced a mask0/mask255 no-op.
