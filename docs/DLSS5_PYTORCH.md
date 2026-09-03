# DLSS-NR PyTorch 图翻译

`tools/dlss5_pytorch.py` 是针对仓库中固定 DLL/cubin 的可读 PyTorch 翻译。它对应 `compute_graph.json` 的 71 个 block：

```text
RGB core input
  → Pre texture front-end + 32 → Swin 32/64/128/256
  → Split-Swin 512
  → ViT 1024 (8 blocks)
  → 对称 decoder
  → Post(32 + enc0 skip 32 → RGB) + ControlMask blend
```

Swin、Split-Swin 和 ViT block 的残差顺序都是 `FFN → ffn_cos_skip → QKV/attention → projection → attn_cos_skip`；这点来自 host constructor 的 child 注册顺序和对应 fused op 列表。它们没有额外的 LayerNorm：FFN 直接吃输入，attention 只对 Q/K 做向量归一化。

FFN 中也不是 GELU，而是 cubin 模板命名的 `MpCubicSiluActivation`：

```python
t = x.clamp(-4, 4)
p = 0.447265625 - 0.055908203125 * t.abs()
y = x * (0.89453125 + t * p)
```

Split-Swin 的 `layer0` 是两条并行的 `512 → 512` 支路，实际为 `cubic_silu(ffwd(x)) * ffwd_gate(x)`，再送入 `layer1` 的投影；不是三层串联 Linear。

普通层级 Swin 的注意力可读形式是：

```python
q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
score = (q @ k.transpose(-2, -1)) * attn_scale + attn_bias
prob = softmax(score, dim=-1)
```

这与 DLL 主机侧保留的 `convolution`、`linalg_vector_norm`、`clamp_min`、`div`、`bmm`、`softmax` 以及 Split/ViT kernel 名称相对应。

ViT 使用另一套独立的 attention kernel：Q/K 归一化后各乘 `sqrt(32)=5.65625`，再做 `bmm`、`layer3.layer` 提供的 score scalar、自定义 half2 `exp`、行和、倒数和第二次 `bmm`；cubin 中没有 ViT attention bias。PyTorch 代码用稳定的 `score - detached_row_max` 加 `torch.exp` 表达同一逻辑。

Split-Swin 最后一个 encoder block 的 `ProjPool` 同时保留两个输出：

```python
full = x + projection(attention) * attn_cos_skip
vit_input = FinalHead(avg_pool2d(full, kernel_size=2, stride=2))
decoder_skip = full
```

因此 block30 的 output0 是低一倍空间、1024 通道的 ViT 输入，output1 是全分辨率、512 通道的 decoder skip。

## 运行

```bash
python tools/extract_dlssnr_resources.py bin/nvngx_dlssnr.dll
python3 tools/dlss5_pytorch.py --self-test
python3 tools/dlss5_pytorch.py --weights DLSS5-extracted/WEIGHTS_HT.bin
python tools/probe_dlss5_pytorch.py --weights DLSS5-extracted --device cuda --dtype float16 --size 256
python tools/export_dlss5_pytorch.py --weights DLSS5-extracted
python tools/run_dlss5_pytorch.py DLSS5-extracted/dlss5_pytorch_reference_fp16.pt --device cuda --size 256
```

`WEIGHTS_HT.bin` 不需要另行寻找：它是 `nvngx_dlssnr.dll` 中类型 10、名称
`WEIGHTS_HT` 的 PE 资源。提取后大小为 147,695,410 bytes，共 153 条记录。

量化 CUBIN 在矩阵乘法中使用 `QMMA.*.F16.E4M3.E4M3`，并以
`F2FP.SATFINITE.E4M3.F16` 写出中间 activation。加载真实权重运行时必须保留
这个边界：

```python
net, weight_map = DLSS5Graph.with_weight_map("DLSS5-extracted", load_known=True)
net.enable_fp8_emulation()
net = net.half().cuda().eval()
```

如果关闭 FP8 emulation，让融合层的中间结果一直以不受限 FP32 传播，gated
Split-Swin 会逐层放大并产生非有限值；这不是原始量化 CUBIN 的执行方式。

本机 RTX 5080（PyTorch 2.11.0+cu128，原生 `sm_120`）验证结果：完整 71-block
图以 FP16 计算、E4M3 activation round-trip 执行 256×256 输入，输出
196,608/196,608 元素全部有限，单次约 0.70 秒，峰值显存约 616 MiB。导出的
checkpoint 含 145,754,963 个参数，约 291.8 MB；从磁盘重新加载后可独立执行。

这仍是“可执行参考翻译”，不是 bit-exact 克隆。把与 native worker 相同的
256×256 deterministic RGB 输入送入两条路径后，当前 PyTorch 输出裁剪到 `[0,1]`
的 RGB MAE 为 `0.48527`（RMSE `0.54778`）。这个较大的差距符合尚未恢复的 pre
texture feature assembly、tensor-core storage permutation、transition 空间重排和
post output swizzle；后续校准应以这个 golden pair 为基线，而不能把“全有限”误当成
已经复现 DLL 数值。

```python
from tools.dlss5_pytorch import DLSS5Graph

net = DLSS5Graph()
out = net(rgb=rgb, control_mask=mask)

# 如果已经在外部重建了 pre kernel 的 texture/cat 前端，可把它生成的
# NCHW 32-channel feature tile 显式送入完整的 32×32 prefix adapter：
out = net(rgb=rgb, pre_features=pre_features, control_mask=mask)

# 如果要把外围 temporal feature 先拼好再送入参考图，显式指定输入宽度；
# 这部分不属于 block0 的已确认 RGB core contract。
temporal_net = DLSS5Graph(color_channels=4, history_channels=4, motion_channels=2)
out = temporal_net(color, history=previous_output, motion=motion, control_mask=mask)
```

## 权重边界

`WEIGHTS_HT.bin` 的外层记录可以严格解析，主体字节流由 cubin 的 signed E4M3 解码路径确认。外层的两字节元素只是容器；FP8 权重按字节存放，部分尾部/头部是 FP16 参数。

目前已经严格切出的布局如下：

| family | section | 已确认的内部布局 |
|---|---|---|
| ViT | layer0 | FP8 `(4096,1024)` + 16 bytes padding |
| ViT | layer1 | FP8 `(1024,4096)` + FP16 `ffn_cos_skip(1024,)` |
| ViT | layer2 | 64-half header + FP8 `(3072,1024)` |
| ViT | layer3 | FP16 score scalar |
| ViT | layer4 | FP8 `(1024,1024)` + FP16 `attn_cos_skip(1024,)` |
| Split-Swin | layer0 | 两个 FP8 `(512,512)` FFN 矩阵 |
| Split-Swin | layer1/3 | FP8 `(512,512)` + FP16 `ffn_cos_skip/attn_cos_skip(512,)` |
| Split-Swin | layer2 | FP8 `(1536,512)` + FP16 `attn_bias(16,64,64)` + FP32 `attn_scale(16,)` |
| Split FinalHead | layer4 | FP8 `(1024,512)` pointwise weight + 16 bytes padding；FinalHead 作用于 `ProjPool` 的 2×2 pooled 512-channel output；host 的第二 convolution slot 在 cubin/record 中无独立权重，reference 按 identity 表达 |
| 普通/transition Swin | layer0 | `weight1/2` FP8，FFN 宽度 `128/224/384/704`；`qkv/projection` FP8；per-head `attn_scale` FP32；两组 `cos_skip` 和 `attn_bias` FP16。ordinary 总长 `20672/61760/197184/689232`，down/up body 及 transition matrix 也已接入 |
| Post block70 | layer0 | 前置 `dw_weight(32,)` 与 `inp_upsample_input_scale(32,)` FP16、C32 body 的 `weight1/2/qkv/projection/cos_skip/attn_bias`、FP32 `attn_scale`，以及按注册顺序拆出的 `out_gain(8×FP16)` + padded FP16 `out_conv_weight(16×32)` 已接入；`out_gain` 当前 8 个值全为 0，静态只确认它被注册、未确认独立 consumer；默认保留 raw fallback，也可用 `post_output_layout="tensor_core_candidate"` 试验由有效行列掩码推导的 4×32 tile；动态 `sin`、正式输出 tile swizzle 仍 unresolved |
| Pre block0 | layer0 | 1024-byte `32×32` E4M3 raw tile + C32 body；完整 raw payload 装入 `pre_texture_adapter` 作为探索入口，但 texture front-end 和 tensor-core storage permutation 仍未解出；没有 front-end 时保留一个明确标注为 fallback 的 RGB 1×1 projection；block0 output0 是无参数 2×2 pool；`weight2` 的 4 个非法槽置零作为运行 fallback |
| Decoder input block39 | layer0 | FP8 `(512,1024)` projection + FP16 `(512,)` channel scale；`sin/tile` 插值路径用 bilinear reference 表达 |

普通 Swin 的 `attn_bias` 起始偏移已校正为 `11360/41120/147744/557600`（C=32/64/128/256）；`projection` 起始偏移为 `19568/57520/180528/623168`。bias 结束到 projection 开始的 `16/16/16/32` 字节不是 opaque gap，而是 `1/2/4/8` 个 FP32 per-head `attn_scale`，其余字节为对齐 padding。up body、pre 和 post 也按相同规则接入了 scale；up body 的前置 `2*C²` 区和 QKV 前的 `sin/opaque` 区仍只记录在 report 中。

可加载的部分现在可以这样调用：

```python
from tools.dlss5_pytorch import DLSS5Graph

net, weight_map = DLSS5Graph.with_weight_map(
    "DLSS5-extracted", load_known=True
)
print(net.weight_report)
```

`load_known=True` 会加载普通 Swin、pre body、四个 downsample-Swin body、四个 upsample-Swin body、ViT、Split-Swin、decoder-input block39 以及 block70 post body 的已确认 FP8 矩阵、window bias、FP32 attention scale、ViT score scalar、pre 的完整 32×32 raw input-prefix payload、post 输入的两个 FP16 channel vector 和 `*_cos_skip` 残差缩放；down/up transition matrix、Split FinalHead 的 `1024×512` pointwise weight，以及 post 的 padded FP16 `out_conv_weight` 也会加载。post tail 现在按注册顺序拆成 `out_gain(8×FP16)` + `out_conv_weight(16×32×FP16)`；`out_gain` 已解码且 8 个值全为 0，但目前只能确认它被注册，无法仅凭 index-based binding 证明是 padding 或没有 consumer；输出 tile 的 tensor-core row/lane swizzle 仍未证明。尚未确定语义的 header、transition 空间重排、pre texture front-end 的 `cat/ones_like/detach` 数值实现及其 prefix storage permutation、block0 `weight2`、ViT half2 exp 的 bit-exact 实现、block39 的动态 `sin/tile` 会放进 report，不会猜测 reshape。FinalHead 的 host 第二 convolution slot 已按 cubin 证据落为 identity。修正 attention-bias 起点后，所有已加载的 bias 都是有限 FP16；`block70.layer0.blend_scale` 已验证为 `0.73974609375`。真实整网执行还必须启用 CUBIN 已确认的 E4M3 SATFINITE activation 边界；关闭它会把本应量化存储的中间结果错误地以无限制 FP32 传播并最终溢出。

剩余的 pre texture front-end 数值组装、downsample/upsample 的精确空间重排、block39 的 `sin/tile` 插值、ViT half2 exp 的 bit-exact 实现、post output tile 的 tensor-core swizzle/out_gain consumer 仍是 fused layout；pre `weight2` 的 4 个非法槽目前只能用零 fallback。要达到 bit-exact，还需要继续从对应 kernel 的 load stride 和 DLL golden 输出校准这些布局及每层量化参数。
