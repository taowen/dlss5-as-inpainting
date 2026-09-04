# DLSS5 block0 SASS evidence

The `sm_120` block0 CUBIN was disassembled with NVIDIA `nvdisasm` 12.9.88.
The extraction is reproducible with:

```powershell
python tools\dlss5_disassemble_cubin.py `
  --nvdisasm C:\path\to\nvdisasm.exe `
  --cubin cubins\fatbin_00\fatbin_00_0xdf0e0.4.sm_120.cubin `
  --function cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8 `
  --sass-output runtime_probe_output\pre_front.sass.txt `
  --summary-output runtime_probe_output\pre_front.sass.json
```

The extracted function contains 13 texture instructions before the body,
including five `0x7` texture reads. It writes two `STS.128` shared
memory records at offsets `0` and `-0x400`, then loads the two serialized FP16
front tiles at record offsets `+0x2010` and `+0x2210`. The first 16 HMMA
instructions consume those shared features and the loaded tiles.

The front sequence also contains `HADD2` with `-0.5`, `HMUL2`, FP16 packing and
`PRMT`. Therefore the unresolved PyTorch input is a texture-derived 16-half
tile with one alignment lane, not a plain 15-channel RGB copy. The exact
coordinate/filter arithmetic and the logical ordering of the packed feature
lanes remain the next reconstruction target.

There is an additional non-RGB dependency before the stores: the SASS mixes an
integer hash path with `MUFU.LG2`, `MUFU.SQRT`, `MUFU.SIN` and `MUFU.COS`, then
converts the results to FP16 before the final packing. This is consistent with
a deterministic Box--Muller-style noise pair, not with an external random
stream. The PyTorch producer will therefore need the same coordinate/seed
hash and half rounding before its 15-channel projection can be compared with
the native golden output.
