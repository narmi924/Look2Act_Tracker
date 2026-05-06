# UnityEyes / Synthetic Data 集成计划

## 当前判断

UnityEyes 风格的 synthetic data 对 Look2Act 有价值，但不能盲目当成当前训练集的直接替代品，也不能作为实时追踪失败的第一修复手段。

当前标签审计已经暴露 deep 真实数据链路中至少两个未解决问题：

- 已存储的 head-pose Euler 角不能直接用于 head-local label regeneration。
- 当前 runtime 固定屏幕几何与 label generation geometry 不够一致。

因此，synthetic data 更适合被定位为“可控预训练”和“覆盖范围补充”工具，而不是马上修复实时 tracking 的主路径。

## 合成数据能帮什么

- 增加眼部外观多样性：眼睑形状、瞳孔位置、虹膜颜色、光照、阴影、局部遮挡、极端 gaze 等。
- 提供渲染时可控的密集 gaze direction label。
- 先预训练 eye-feature backbone，再用 Look2Act real data fine-tune。
- 支持数据覆盖消融实验：
  - real only；
  - synthetic only；
  - synthetic pretrain + real fine-tune；
  - synthetic low-weight mixed with real。

## 合成数据不应该怎么用

- 不要把 UnityEyes 的二维眼部参数直接和 Look2Act 的 screen-point calibration target 混在一起。
- 不要在实时几何和校准诊断通过前声称 synthetic data 解决了 live tracking failure。
- 在真实数据 head-pose 约定未解决前，不要用 synthetic data 训练 head-local label 路线。
- 不要在没有 real validation/test split 和 no-synthetic baseline 的情况下报告 synthetic gain。

## 推荐的 Label Contract

每个实验必须只使用一种明确的训练目标约定。

### 方案 A：Camera-Space Gaze Pretraining

- Synthetic adapter 输出左/右眼裁剪图，以及 camera-like coordinate system 下的 3D gaze vector。
- Real-data fine-tuning 使用当前 `gaze_x/y/z`，暂时把它视为 camera-space vector。
- Runtime deep backend 使用 `deep_gaze_space=camera`。
- Screen projection 的 ray origin 必须和 label contract 保持一致。

这是 synthetic data 的第一优先路线，因为它绕开了当前可疑的 Euler-angle 转换。

### 方案 B：Head-Local Gaze Pretraining

- Synthetic adapter 输出眼部裁剪、head pose 和 head-local gaze。
- Real labels 必须用经过验证的 rotation matrix 重新生成，不能直接使用当前 raw Euler values。
- Runtime deep backend 使用 `deep_gaze_space=head`。

这条路线应等待 head-pose audit 问题解决后再做。

## 数据 Adapter 形态

转换器应输出和 Look2Act processed dataset 一致的 schema：

```csv
eye_img_path,right_eye_img_path,gaze_x,gaze_y,gaze_z,head_yaw,head_pitch,head_roll,norm_target_x,norm_target_y,session_id,user_id
```

对于没有真实 screen target 的 synthetic row：

- 如果训练代码不依赖 `norm_target_x/norm_target_y`，可以留空或使用明确 sentinel。
- 如果训练/评估代码需要 screen target，则必须根据 synthetic gaze vector 和 synthetic camera geometry 生成一致的 virtual target plane。

synthetic 数据应写入独立目录，例如：

```text
dataset_synthetic_processed/
  train/
    images/
    labels.csv
  val/
    images/
    labels.csv
```

不要把 synthetic row 原地合并进 `dataset_processed`。

## 第一轮实验矩阵

在任何长时间 LOO 之前，先跑小规模有限实验：

| ID | 训练数据 | Label Space | Runtime Backend | 目的 |
| --- | --- | --- | --- | --- |
| R0 | Look2Act real only | 当前 camera-like labels | deep camera | baseline |
| S0 | synthetic only | camera-space | deep camera | 检查 domain gap |
| S1 | synthetic pretrain + real fine-tune | camera-space | deep camera | 检查覆盖提升 |
| S2 | synthetic low-weight mixed + real | camera-space | deep camera | 检查正则化效果 |

只有这些实验跑通后，才考虑 head-local variants。

## 成功标准

- real validation/test users 上的 offline angular error 改善，或至少不退化。
- live diagnostic CSV 显示 deep raw points 不塌缩，也不会在校准前大幅跳出屏幕。
- deep raw output 具备空间意义后，calibration residual 应降低。
- classic tracker 在 synthetic 实验期间继续作为可演示 fallback，不被移除。

## 下一步实现任务

1. 保持 `classic` 为默认 demo flow。
2. 在生成 head-local labels 前，先修正或替换 head-pose Euler 约定。
3. 只有在选定 target contract 后，再添加 synthetic dataset adapter。
4. 添加可选择 `real`、`synthetic`、`mixed` 数据源的训练配置，不要修改现有数据集目录。
5. synthetic 结果作为 ablation 报告，不作为核心系统依赖。
