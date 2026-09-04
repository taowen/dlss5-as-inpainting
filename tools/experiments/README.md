# Experiment tools

These scripts are development-only and are separate from the reusable model
package in `src/dlss5/`.

| script | purpose |
|---|---|
| `download_public_examples.py` | fetch the four public-domain/CC0 fixtures and record hashes |
| `run_dlss5_image_cases.py` | build color/depth/motion contracts, invoke a local native harness, and write PNG/JSON evidence |

The image runner requires a locally prepared `dlss5_eval.exe` and `ffmpeg`.
Native binaries are intentionally not stored in this repository. Temporary
contract planes go under ignored `examples/.work/`; reviewable outputs go under
`examples/cases/native/`.
