# Depth3D + DLSS 5 右眼遮挡修复方案

> 后续实施以 [立体视觉方案 V2](STEREO_PIPELINE_V2.md) 为准。本文保留为跨眼虚拟历史的 V1 方案及实验记录。

## 目标与结论

统一以 ReShade/Add-on 方式运行，不再单独估计跨时间光流，也不使用 NVOF。

把同一游戏帧的左右眼建模成一个两步虚拟时间序列：

```text
虚拟上一帧 = 左眼
虚拟当前帧 = 右眼
虚拟相机运动 = 左眼相机位置 → 右眼相机位置
```

Depth3D 用游戏深度把左眼重投影到右眼，并在重投影时同时生成右眼预填充图、右眼深度、Hole Mask，以及“右眼当前像素指向左眼历史像素”的 Motion Vector。随后先用左眼初始化 DLSS 5 历史，再把右眼作为下一帧提交，利用 DLSS 5 的时间网络减轻右眼空洞预填充产生的拉伸、裂纹、halo 和模糊块。

这里的 DLSS 5 指 DLSS Neural Rendering（NGX Feature 18），不是 Frame Generation。它仍然是受当前图像和历史图像约束的神经增强器，不保证恢复左眼中从未可见的真实遮挡背景。

## 总体流程

```text
游戏 Color + Raw Depth
  → 构造左眼 Color / Depth
  → 左眼 Depth 反投影到三维
  → 左眼相机 → 右眼相机变换
  → 投影并做右眼 Z-buffer
       ├─ Right Warped Color
       ├─ Right Warped Depth
       ├─ Right-to-Left Source Map
       ├─ Right Motion Vector
       └─ Right Hole Mask
  → 对右眼空洞做最小规则预填充
  → Hole Mask 膨胀、深度边缘约束、羽化
  → DLSS 5：左眼 Reset=1，初始化历史
  → DLSS 5：右眼 Reset=0，作为虚拟下一帧处理
  → 左眼原图 + DLSS 右眼输出
  → SBS/TAB 打包
```

## 模块及输入输出

| 模块 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| ReShade 捕获层 | 游戏 Backbuffer、Raw Depth | 游戏颜色、线性化深度 | 复用现有 ReShade/Depth3D 捕获能力。 |
| 左眼构造 | 游戏颜色、深度、立体参数 | `LeftColor`、`LeftDepth` | 左眼是 DLSS 虚拟历史帧。 |
| 深度重投影 | `LeftColor`、`LeftDepth`、左右眼内外参 | `RightWarpedColor`、`RightDepth`、`SourceMap`、`HoleMask` | 正向投影到右眼并做 Z-buffer；每个有效右眼像素保留其左眼来源坐标。 |
| 几何 MV 生成 | `SourceMap`、右眼像素坐标 | `RightMVec` | 不做光流估计；直接由已知深度和相机位姿产生确定的双目对应关系。 |
| 最小预填充 | `RightWarpedColor`、`RightDepth`、`HoleMask` | 完整的 `RightPrefilledColor` | DLSS 输入纹理不能含未定义像素；预填充尽量简单，避免先引入复杂规则纹理。 |
| Control Mask 生成 | `HoleMask`、`RightDepth` | `RightControlMask` | 洞区膨胀、遮挡边缘约束、羽化后转换为 R8_UNORM。 |
| DLSS 5 历史初始化 | `LeftColor`、零 MV、Reset=1 | 内部 `prev_output` | 每个游戏帧/双目对开始时清除旧历史，再把左眼写成虚拟上一帧。该更新行为需运行验证。 |
| DLSS 5 右眼修复 | `RightPrefilledColor`、`RightMVec`、`RightControlMask`、Reset=0 | `RightEnhancedColor` | 当前图像、运动对齐后的左眼历史在网络内融合；ControlMask 只控制最终神经结果的可见比例。 |
| Stereo Pack | `LeftColor`、`RightEnhancedColor` | SBS/TAB | DLSS 主要处理右眼；左眼保持真实渲染基准。 |
| 诊断层 | SourceMap、Depth、MV、Hole Mask、DLSS 输出 | Debug View、差分图 | 验证 MV 方向/尺度、遮挡边界和左右眼一致性。 |

## 从 Depth Map 生成“光流”

这一步产生的是由几何确定的双目 Motion Vector，不是图像匹配得到的 optical flow。

对左眼像素 `pL=(xL,yL)` 及其线性深度 `zL`：

```text
XL = zL × inverse(KL) × homogeneous(pL)
XR = R_RL × XL + t_RL
pR = project(KR × XR)
```

将 `LeftColor(pL)` 正向写入 `pR`，按右眼深度做 Z-buffer。获胜样本同时写入：

```text
RightWarpedColor(pR) = LeftColor(pL)
RightDepth(pR)       = XR.z
SourceMap(pR)        = pL
```

DLSS 需要的是当前像素查询上一虚拟帧的位置。首版采用 current-to-previous、像素单位的约定：

```text
previous_pixel = current_pixel + motion_vector
RightMVec(pR)  = pL - pR
```

对于经过校正的平行双目相机，水平视差的绝对值近似为：

```text
|disparity| = fx × baseline / depth
```

不能直接把 `pR-pL` 的正向视差写给 DLSS；提交的是定义在右眼像素上的反向查询矢量 `pL-pR`。由于未公开 ABI 仍可能存在符号、归一化或 Y 轴方向差异，必须用固定深度平面和已知水平位移做运行标定，再确定 `MVecScaleX/Y`。

正向投影后没有左眼来源的右眼像素为 disocclusion：

```text
SourceMap 无效 → HoleMask=1
```

这些像素不存在真实的左右眼对应关系，因此没有物理正确的 MV。首版将其 MV 置零，或从同一背景深度层的最近有效边界传播，并保留单独的无效标记供调试。不能把前景矢量跨越深度边界传播进背景洞区。

建议资源格式：

| 资源 | 格式 | 尺寸 | 内容 |
| --- | --- | --- | --- |
| `RightDepth` | R32_FLOAT | W×H | Z-buffer 后的右眼线性深度。 |
| `SourceMap` | RG32_FLOAT 或 RG16_FLOAT | W×H | 每个有效右眼像素对应的左眼像素坐标。 |
| `RightMVec` | RG16_FLOAT | W×H | `pL-pR`，待运行标定其方向和尺度。 |
| `HoleMask` | R8_UNORM | W×H | 0=有重投影来源，1=无来源。 |
| `RightControlMask` | R8_UNORM | W×H | Hole Mask 经膨胀、深度边缘约束和羽化后的 DLSS 最终混合权重。 |

## DLSS 5 的实际计算关系

对本地实际模型 DLL：

```text
DLSS.5.Visual.Enhancer.v3.0/bin/runtime/nvngx_dlssnr.dll
SHA256: 6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927
```

静态分析确认其神经网络直接使用当前 `Color`、内部上一输出 `prev_output` 和 `MVec`。参数解析器虽然读取 `DLSSNR.Depth`，但这个精确 DLL 构建的 `CG2RNetworkManager::Evaluate` 没有把 Depth 送入神经核心。因此本方案不依赖 DLSS 自己理解深度，而是在 DLSS 调用前把深度转换为右眼重投影、MV 和 Hole Mask。

概念关系为：

```text
NeuralRight = Network(
    current = RightPrefilledColor,
    history = previous DLSS output for LeftColor,
    motion  = RightMVec,
    controls)

RightOutput = PostBlend(
    base   = RightPrefilledColor,
    neural = NeuralRight,
    weight = saturate(RightControlMask.r × blend_scale))

// native simple/control-mask tail
RightOutput = neural + weight × (base - neural)
```

这不是左右图的简单 alpha 混合：网络在特征空间中根据 MV 融合当前右眼与左眼历史；最后的 post-block 才在基础结果与神经结果之间混合。

### ControlMask 的边界

Cubin SASS 静态分析确认：

```text
effective_blend = saturate(ControlMask.r × blend_scale)
Output = NeuralColor + effective_blend × (BaseColor - NeuralColor)
```

因此 `ControlMask`：

- 只读取 R 通道；首选 R8_UNORM。
- 0 表示保留神经结果，1 表示按 `blend_scale` 限制在基础结果与神经结果之间混合。
- 不进入 Encoder，不是显式的 inpainting condition。
- 适合把神经修改集中到洞区和遮挡边缘，但不能保证生成真实的未知背景。

## NGX Feature 18 调用

首版限定 D3D12。左右眼不是两个长期独立的时间实例，而是在同一个 feature/context 中按“左眼 → 右眼”顺序执行。每个新的游戏帧都重新从左眼开始并设置 Reset，避免上一游戏帧右眼污染下一对双目图。

### 创建

```cpp
NVSDK_NGX_D3D12_Init_with_ProjectID(..., device, ...);
NVSDK_NGX_D3D12_GetCapabilityParameters(&caps);
NVSDK_NGX_D3D12_AllocateParameters(&params);

params->Set("DLSSNR.Enabled", 1u);
params->Set("DLSSNR.Width", eye_width);
params->Set("DLSSNR.Height", eye_height);
params->Set("DLSSNR.Hint.Render.Preset", preset);
params->Set("DLSSNR.Intensity", intensity);
params->Set("DLSSNR.Style", style);
params->Set("DLSSNR.LocalStructureStrength", local_structure);
params->Set("DLSSNR.LocalToneStrength", local_tone);
params->Set("DLSSNR.SkinStructureStrength", skin_structure);
params->Set("DLSSNR.UseAutoMask", 0u);

NVSDK_NGX_D3D12_CreateFeature(
    cmd, (NVSDK_NGX_Feature)18, params, &feature);
```

创建时锁存的 preset、style、intensity 和 strength 参数发生变化时，应重建 feature。

### 第一次 Evaluate：左眼初始化历史

```cpp
SetResource(params, "DLSSNR.Color",       left_color);
SetResource(params, "DLSSNR.Depth",       left_depth);       // ABI 兼容；该构建的核心未使用
SetResource(params, "DLSSNR.MVec",        zero_mv);
SetResource(params, "DLSSNR.ControlMask", zero_mask);
SetResource(params, "DLSSNR.Output",      left_scratch_output);

SetUInt(params,  "DLSSNR.Reset", 1u);
SetFloat(params, "DLSSNR.MVecScaleX", calibrated_scale_x);
SetFloat(params, "DLSSNR.MVecScaleY", calibrated_scale_y);
SetFullSubrect(params, ...);

NVSDK_NGX_D3D12_EvaluateFeature(cmd, feature, params, nullptr);
```

`left_scratch_output` 可丢弃，最终显示仍使用 `LeftColor`。这里的目的，是让下一次 Evaluate 的内部 `prev_output` 对应左眼。标准时序语义应在 Reset 帧执行结束后写入新历史，但这是本方案成立的关键点，必须用捕获和受控测试确认。

### 第二次 Evaluate：右眼作为虚拟下一帧

```cpp
SetResource(params, "DLSSNR.Color",       right_prefilled_color);
SetResource(params, "DLSSNR.Depth",       right_depth);       // ABI 兼容；不依赖其参与网络
SetResource(params, "DLSSNR.MVec",        right_mv);
SetResource(params, "DLSSNR.ControlMask", right_control_mask);
SetResource(params, "DLSSNR.Output",      right_enhanced_color);

SetUInt(params,  "DLSSNR.Reset", 0u);
SetFloat(params, "DLSSNR.MVecScaleX", calibrated_scale_x);
SetFloat(params, "DLSSNR.MVecScaleY", calibrated_scale_y);
SetUInt(params,  "DLSSNR.UseAutoMask", 0u);
SetFullSubrect(params, ...);

NVSDK_NGX_D3D12_EvaluateFeature(cmd, feature, params, nullptr);
```

| 右眼 Evaluate 输入 | 格式/尺寸 | 内容 |
| --- | --- | --- |
| `DLSSNR.Color` | 与 Output 兼容，W×H | 深度重投影并完成最小预填充的右眼颜色。 |
| `DLSSNR.Depth` | R32F，W×H | 绑定以兼容 ABI；当前精确 DLL 的神经核心未使用。 |
| `DLSSNR.MVec` | RG16F，W×H | 定义在右眼坐标上的 `pL-pR`；符号和 scale 需运行标定。 |
| `DLSSNR.ControlMask` | R8_UNORM，W×H | 空洞及遮挡边缘的神经输出混合强度。 |
| `DLSSNR.Reset` | 0 | 使用紧邻的左眼 Evaluate 所建立的历史。 |
| 各资源 Subrect | uint×4 | BaseX/BaseY/Width/Height；首版使用完整单眼区域。 |

唯一外部图像输出是同尺寸的 `DLSSNR.Output`。模型不会输出深度、MV、mask 或另一只眼。

## ReShade/Add-on 执行顺序

```text
Capture Color/Depth
→ Depth3D Left-to-Right Reprojection + Z-buffer
→ Right SourceMap/MV/Hole Mask
→ Minimal Right Prefill
→ Hole Mask Dilate/Depth-Edge Constraint/Feather
→ DLSS Evaluate Left (Reset=1, seed history)
→ DLSS Evaluate Right (Reset=0, use geometric MV)
→ Pack LeftColor + RightEnhancedColor as SBS/TAB
```

无需切换到 UEVR，也无需 NVOF、图像光流、跨游戏帧 MV 或左右眼独立的 DLSS 历史。

## 验证顺序

1. 用固定深度平面和已知 baseline 验证 `SourceMap`、Z-buffer、洞区方向与视差大小。
2. 分别提交 `RightMVec=pL-pR`、反号矢量和零矢量，通过差分确认 DLSS 的方向、单位及 `MVecScaleX/Y`。
3. 比较“只执行右眼”“先 Reset 左眼再执行右眼”，确认第二次 Evaluate 确实读取刚建立的左眼 `prev_output`。
4. 使用全 0、全 1、半屏和 Hole Mask 四种 ControlMask，确认其 R 通道、空间范围和混合方向。
5. 比较规则预填充、DLSS 自动 mask、显式 Hole ControlMask，检查洞区瑕疵、双眼一致性和闪烁。
6. 每个游戏帧重复 `Left Reset=1 → Right Reset=0`，确认不存在上一帧右眼串入下一帧左眼的历史污染。

## 实施阶段

1. 改造 `SuperDepth3D.fx`：在打包前输出 `LeftColor/Depth`，并在左→右重投影时生成 `RightColor/Depth`、`SourceMap` 和 `HoleMask`。
2. 增加几何 MV pass：由 `SourceMap` 计算定义在右眼坐标上的 `pL-pR`，处理越界、Z-buffer 冲突和洞区无效矢量。
3. 将规则补洞缩减为最小预填充，并生成膨胀、深度边缘约束和羽化后的 R8 `DLSSNR.ControlMask`。
4. 扩展 Feature 18 wrapper：同一 feature 每游戏帧连续 Evaluate 左眼和右眼，绑定 ControlMask 及全部 Subrect。
5. 完成 MV 标定、Reset/history、mask 和左右眼一致性测试，再确定默认强度和失败回退。

## 主要风险

- `Reset=1` 的左眼 Evaluate 是否在结束时形成下一次调用可读的 `prev_output`，需要运行验证；静态分析只确认内部存在并读取 previous-output 资源。
- MV 的 current-to-previous 定义符合网络接线推断，但未公开 ABI 的实际符号、单位和坐标原点仍必须标定。
- ControlMask 只是最终 post-block 混合门控，不会把 Hole Mask 作为生成条件送进网络。
- 真正的 disocclusion 在左眼没有对应内容；DLSS 只能根据当前预填充、邻域和训练先验改善观感，不能恢复真实背景。
- 同一游戏帧需要两次 Feature 18 Evaluate，成本高于只处理一次右眼；但已去掉双路 NVOF 和跨时间双眼处理。
- DLSS 可能在右眼生成与左眼不一致的高频细节，必须以 stereo consistency 和观看舒适度作为验收指标。
- Feature 18 没有公开稳定 SDK；必须固定 DLL 哈希，升级 DLL、驱动或转发层后重新验证字段和行为。

## 依据

- Feature 18 的创建/Evaluate 方式参考本地 [`dlssnr_forwarder.cpp`](OptiScaler_DLSSNR/OptiScaler/dlssnr/forwarder/dlssnr_forwarder.cpp)。
- `Color/prev_output/MVec` 接线、Depth 未进入当前核心、ControlMask 的最终混合行为来自本地 [`nvngx_dlssnr.dll`](DLSS.5.Visual.Enhancer.v3.0/bin/runtime/nvngx_dlssnr.dll) 静态分析，详见 [`DLSS5_COMPUTE_GRAPH.md`](DLSS5_COMPUTE_GRAPH.md)。
- 本方案中的左右眼虚拟时序、深度几何 MV 和每对双目图 Reset 是工程设计，需要按上述验证顺序实测，不作为 NVIDIA 已公开保证的用途。

## 2026-09-05 实验状态

已使用 Distill-Any-Depth small 对公开图片生成相对深度，并完成了
`Left Reset=1 -> Right Reset=0` 的 stereo forward-splat 实验。实验结果与完整图片、
HoleMask、Motion Vector 和 JSON 指标见
[`DLSS5_STEREO_EXPERIMENTS.md`](DLSS5_STEREO_EXPERIMENTS.md)。

当前结论不是“DLSS5 已解决双目洞填补”：在可构造 ground truth 的平面纹理 oracle 中，
DLSS5 将洞区 MAE 从 `0.15212` 降到 `0.13797`，但全图 MAE 从 `0.00951` 升到 `0.05010`；
随后已用临时 spatial mask-file harness 提交 `valid=255/hole=0` 的 R8 mask，结果与
`mask=255` 仍逐值一致。说明当前 runtime/add-on 的 native ControlMask 绑定没有可观测
效果；因此已在 stereo 最终合成边界实现同语义的 host-side ControlMask：有效像素选择
简单重投影，HoleMask 像素选择 DLSS5 输出。这样 oracle 全图 MAE 从 `0.00951` 降到
`0.00862`，同时保持 native mask A/B 证据不变。

因此下一步应先用 launch/resource telemetry 定位 ControlMask 是否真正绑定到 native
执行链，再用有真实遮挡背景的 layered scene 做最终判定。
