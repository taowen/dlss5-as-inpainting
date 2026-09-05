# DLSS5 for Stereo Disocclusion Repair: Findings

## What DLSS5 does

DLSS5 Neural Rendering modifies an existing rendered image to enhance its appearance. In our tested pipeline, image content, temporal history, and motion vectors affect the output.

It is not a dedicated inpainting model. We have not established that it can reliably reconstruct background surfaces hidden behind foreground objects.

Our experiments use a local NVIDIA runtime on an RTX 5080 through a D3D12/DLAA–ReShade–RenoDX carrier. Consequently, measured effects belong to that complete pipeline; we have not yet isolated the contribution of Neural Rendering from the surrounding processing.

## Why test it for stereo generation?

Generating a second eye from one image and a depth map exposes background regions absent from the original view. These *disocclusion holes* commonly appear beside foreground silhouettes.

Simple pixel shifting and nearest-neighbor filling produce stretched textures and visible seams. We investigated whether DLSS5 could improve those regions by treating the left eye as the previous frame and the reconstructed right eye as the current frame.

The experimental flow was:

```text
Source image + depth
    → right-eye reprojection and visibility testing
    → hole mask and geometric motion vectors
    → background prefill
    → DLSS5 candidate
    → external composition restricted to the hole region
```

Distill Any Depth Small was used for the photographic examples. For quantitative evaluation, we also constructed layered scenes with known geometry and independently rendered right-eye ground truth.

## What we learned about the inputs

| Input | Finding |
|---|---|
| Color | Affects output. |
| History / reset | Affects output. |
| Motion vectors | Affect output; correct geometric correspondence matters. |
| Depth | The analyzed DLL’s neural core does not consume it. We use depth externally for stereo geometry. |
| Native ControlMask | No observable effect through our tested carrier, including constant and spatial masks. Parameter propagation to Feature18 remains unresolved. |

We therefore use an explicit host-side mask to preserve valid reprojected pixels and apply the DLSS5 candidate only inside selected holes. This is application-level composition, not proof that native ControlMask works.

Updating the driver from **616.56 to 616.64** did not change the tested outputs.

## Quantitative results

Our latest benchmark contains an opaque foreground rectangle over three background materials, with **8- and 16-pixel internal disocclusion bands**. Hidden background pixels are available only to the evaluator.

The table reports hole-region mean absolute error in linear RGB; lower is better.

| Background / hole width | Nearest-neighbor fill | Background-side fill | Background-side fill + DLSS5 with left-eye history |
|---|---:|---:|---:|
| Smooth / 8 px | 0.16848 | **0.00244** | 0.00829 |
| Smooth / 16 px | 0.16939 | **0.00460** | 0.00893 |
| Stripes / 8 px | 0.10667 | **0.10000** | 0.10418 |
| Stripes / 16 px | 0.15667 | **0.11667** | 0.12285 |
| Stone / 8 px | 0.20708 | 0.20344 | **0.18949** |
| Stone / 16 px | 0.21013 | 0.20382 | **0.18821** |

Two findings stand out:

- **Choosing background pixels instead of foreground pixels is the largest improvement**, especially on smooth surfaces.
- DLSS5 reduced error by approximately **6.9% and 7.7%** on the stone cases, but increased error on all four smooth and striped cases.

Visual inspection also showed that stretched horizontal artifacts remained in the stone examples. A lower pixel error therefore does not establish convincing texture reconstruction.

A separate motion-vector test shifted content left by eight pixels. Supplying a current-to-previous horizontal vector of **+8 pixels** produced overlap MAE **0.03029**, compared with **0.07981** for zero motion and **0.07936** for the opposite sign. This supports the geometric convention used in our experiments, although appearance changes prevent treating this metric alone as a complete validation of native sampling behavior.

## Conclusion

**DLSS5 currently shows limited, material-dependent value as a local refinement stage—not a reliable general solution for stereo disocclusion filling.**

The evidence favors improving geometry and background reconstruction first. DLSS5 may then be useful for selected textured regions, provided that external composition protects valid pixels and a quality check can reject harmful changes.

The next design will reconstruct a **shared background representation before rendering both eyes**. This should help keep generated textures consistent between eyes. We will then compare:

1. Background completion without DLSS5.
2. DLSS5 enhancement before splitting into two eyes.
3. Local DLSS5 refinement after stereo rendering.

Further work must evaluate independent scenes, fine silhouettes, real video motion, binocular consistency, temporal flicker, and runtime cost. The present six-case benchmark supports targeted experimentation, but not production-wide enablement.

The separate small PyTorch surrogate in the repository should not be confused with the native DLSS5 model; its general image quality and inpainting equivalence have not been established.

## Supporting evidence

- [Full experiment report and images](STEREO_V2_RESULTS.md)
- [Layered-scene metrics](../examples/cases/stereo_v2/layered/metrics.json)
- [Motion-vector comparison](../examples/cases/stereo_v2/layered/motion_calibration.json)
- [Driver update regression](DLSS5_DRIVER_61664_REGRESSION.md)
- [Stereo Pipeline V2 design](STEREO_PIPELINE_V2.md)

Experiment baseline: commit `e051ea4`.
