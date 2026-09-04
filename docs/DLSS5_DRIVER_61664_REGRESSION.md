# NVIDIA driver 616.64 regression

The local RTX 5080 was upgraded from driver `616.56` to `616.64`. The control
environment was recorded before the update in
`runtime_probe_output/pre_driver_update_616_56.json`.

## Results

| check | 616.56 baseline | 616.64 result |
|---|---|---|
| bit-exact carrier SHA | `1fe38ab7...f38f1e3` | `1fe38ab7...f38f1e3` |
| Depth luma vs flat output | not different | MAE/RMSE/max = `0/0/0`; raw SHA identical |
| spatial ControlMask vs constant mask | MAE/RMSE/max = `0/0/0` | MAE/RMSE/max = `0/0/0` |
| UseAutoMask 0 vs 1 | not different in prior probe | MAE/RMSE/max = `0/0/0` |
| stereo plane oracle raw DLSS5 | `20e421...ad18` | `20e421...ad18` |

The post-update stereo run still reports a 6.25% oracle hole region, raw
DLSS5 hole MAE `0.137969`, and host-side ControlMask-composited all-pixel MAE
`0.008623` versus simple-fill all-pixel MAE `0.009508`. The Blue Marble and
Portrait predicted-depth cases also retain their previous hole fractions:
`5.20%` and `10.73%`.

## Interpretation

For this pinned `nvngx_dlssnr.dll`, ReShade/add-on, harness, and 5080, the NBA
2K27 driver update did not change the observed neural output or make Depth,
ControlMask, or UseAutoMask effective. The result does not prove that future
drivers or other DLSS5 builds behave the same; it establishes that the current
ControlMask no-op is not caused by the 616.56 driver alone.

