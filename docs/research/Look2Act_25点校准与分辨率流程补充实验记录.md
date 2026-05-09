# Look2Act 25 点校准与分辨率流程补充实验记录

记录日期：2026-05-09

## 1. 当前执行范围

本轮没有修改论文 docx 正文、公式、题注、参考文献或目录。后续正式写入 docx 时再使用 Documents 插件执行。

本轮已完成两类非 docx 工作：

1. 生成校准 JSON 分析脚本，并分析当前可追溯的校准文件。
2. 生成扩展校准对比脚本，并在现有评估结果基础上补充 13 点、25 点 affine / polynomial 的离线 hold-out 对比。

注意：在用户暂停 docx 修改前，已生成或覆盖了若干论文引用图片资源，包括 `5-12.png`、`5-12.svg`、`6-6.png` 和附录 B 部分图片顺序资源。当前 docx 未继续写入；这些图片可在后续 Documents 阶段审核后使用，如需恢复也应在正式改 docx 前统一处理。

## 2. 数据集处理状态与取舍

`dataset_processed` 当前缺失或为空，但 `dataset_raw` 仍保留原始 session 数据，已确认多个 session 下存在 `labels.csv` 和 `meta.json`。因此，如后续需要完整重跑模型评估，可以重新执行预处理脚本生成 `dataset_processed`。

本次校准对比实验没有立即重跑 `scripts/preprocess.py`，原因如下：

1. 现有 `evaluation_results/error_analysis/per_sample_errors.csv` 已包含测试集的预测归一化坐标、目标归一化坐标和像素误差。
2. 使用该文件中 `split=test` 数据，并按 1536x864 屏幕尺寸还原像素误差，可以复现原有无校准基线 `314.5 px`。
3. 本次目标是比较不同校准点数和映射方式在同一预测结果上的校准后 hold-out 误差，不需要重新训练或重新推理模型。
4. 直接使用已生成的 per-sample 评估结果可以避免重新处理数据集带来的样本划分变化、运行时间和环境风险。

如后续确实需要重建 `dataset_processed`，建议先确认当前输出目录不含需要保留的人工文件，再执行：

```powershell
conda.exe run --no-capture-output -n gaze-env python scripts/preprocess.py --config configs/train_config.yaml
```

若需要清空并完全重建，再在确认无保留文件后增加 `--clean` 参数。

## 3. 已新增脚本与输出

已新增脚本：

| 脚本 | 功能 |
|---|---|
| `scripts/analyze_calibration_results.py` | 扫描运行期与项目内校准 JSON，重算每点拟合残差，输出校准摘要和 Classic 25 点残差图。 |
| `scripts/exp_calibration_compare_extended.py` | 基于现有测试集预测结果扩展校准策略对比，加入 13 点、25 点 affine / polynomial。 |

已执行命令：

```powershell
conda.exe run --no-capture-output -n gaze-env python scripts/analyze_calibration_results.py
conda.exe run --no-capture-output -n gaze-env python scripts/exp_calibration_compare_extended.py
```

已生成结果：

| 输出文件 | 说明 |
|---|---|
| `evaluation_results/calibration_final/calibration_summary.csv` | 校准文件摘要。 |
| `evaluation_results/calibration_final/calibration_summary.json` | 校准文件摘要 JSON。 |
| `evaluation_results/calibration_final/classic_25pt_residuals.csv` | Classic 25 点逐点拟合残差。 |
| `evaluation_results/calibration_final/classic_25pt_residuals.png` | Classic 25 点逐点残差图。 |
| `evaluation_results/calibration_final/deep_25pt_residuals.csv` | Deep 25 点逐点拟合残差。 |
| `evaluation_results/calibration_final/deep_25pt_residuals.png` | Deep 25 点逐点残差图。 |
| `evaluation_results/calibration_final/deep_25pt_manual_test_template.csv` | Deep 25 点人工实测记录模板。 |
| `evaluation_results/calibration_final/README_25pt_calibration_test.md` | Deep 25 点校准补测说明。 |
| `evaluation_results/calibration_compare_extended/results.csv` | 扩展校准对比结果。 |
| `evaluation_results/calibration_compare_extended/results.json` | 扩展校准对比结果 JSON。 |
| `evaluation_results/calibration_compare_extended/calibration_strategy_matrix.png` | 扩展校准策略对比图。 |

## 4. 最终校准配置核对

当前配置文件中，Classic 和 Deep 均显式配置为 25 点校准：

| 项目 | Classic | Deep | 代码或文件依据 |
|---|---:|---:|---|
| `calibration.num_points` | 25 | 25 | `configs/classic.yaml`、`configs/deep.yaml` |
| 摄像头请求分辨率 | 1280x720 | 1280x720 | `configs/classic.yaml`、`configs/deep.yaml` 的 `camera.width/height` |
| 有效校准方法 | polynomial | polynomial | `src/tracker/pipeline.py` 中 `effective_calibration_method`：点数 >= 25 时为 polynomial |
| 校准点网格 | 5x5 | 5x5 | `src/ui/calibration_page.py`：`num_points >= 25` 时使用 5x5 |
| 默认保存文件 | `calibration_classic.json` | `calibration_deep.json` | `configs/classic.yaml`、`configs/deep.yaml` |

结论：论文不能把 9 点 affine 写成最终系统配置。9 点 affine / 9 点 polynomial 应定位为少点校准策略对比实验；最终 Classic 和 Deep 配置均为 25 点 polynomial。

## 5. 当前校准文件分析结果

校准文件分析结果如下：

| 来源 | 路径类型 | 方法 | 点数 | 平均拟合残差 | 中位残差 | 最大残差 | 屏幕分辨率记录 |
|---|---|---|---:|---:|---:|---:|---|
| runtime Classic | 默认运行期路径 | polynomial | 25 | 73.58 px | 53.96 px | 212.54 px | 1440x900，由目标点坐标推断 |
| runtime Deep | 默认运行期路径 | polynomial | 25 | 118.69 px | 103.46 px | 251.97 px | 1440x900，由目标点坐标推断 |
| bin Classic | 非默认样本路径 | polynomial | 25 | 93.26 px | 87.62 px | 203.90 px | 1440x900，由目标点坐标推断 |
| legacy root | 旧根目录样本 | polynomial | 9 | 293.77 px | 248.08 px | 611.88 px | 1440x900，由目标点坐标推断 |

论文主结果建议采用默认运行期 `runtime Classic` 的 25 点 polynomial 结果。`bin/calibration_classic.json` 可作为旁证，不建议作为主表结果。`calibration.json` 是旧 9 点 polynomial 文件，不应作为最终系统配置依据。

Deep 运行期 `calibration_deep.json` 已于 2026-05-09 21:01:14 生成并备份，当前可以作为论文中 Deep 25 点实时校准拟合残差的数据来源。该结果仍属于校准样本拟合残差，不应写成 hold-out 泛化误差。

## 6. 扩展校准对比实验方法

扩展实验的数据源为 `evaluation_results/error_analysis/per_sample_errors.csv` 的测试集记录。该文件提供：

| 字段 | 用途 |
|---|---|
| `pred_nx`, `pred_ny` | 模型预测的归一化屏幕坐标 |
| `norm_target_x`, `norm_target_y` | 目标点归一化屏幕坐标 |
| `pixel_error` | 原始未校准像素误差 |
| `split` | 使用 `test` 子集 |

实验设置：

| 项目 | 设置 |
|---|---|
| 随机种子 | 42 |
| 重复次数 | 20 |
| 评价方式 | 从测试集中抽取校准点拟合映射，其余样本作为 hold-out 测试 |
| 屏幕尺寸 | 1536x864，用于复现原始像素误差 |
| affine 特征 | `[x, y, 1]` |
| polynomial 特征 | `[x, y, xy, x^2, y^2, 1]` |
| 对比配置 | 无校准、1/3/5/9/13/25 点 affine、9/13/25 点 polynomial |

本实验与运行期 `calibration_classic.json` 的 25 点拟合残差不是同一评价指标。扩展实验是 hold-out 泛化误差；运行期 JSON 是用户校准样本上的拟合残差。

## 7. 扩展校准对比结果

| 方法 | 校准点数 | hold-out 平均误差 | 标准差 | hold-out 中位误差 | P90 误差 | 校准点拟合误差 |
|---|---:|---:|---:|---:|---:|---:|
| 无校准 | 0 | 314.54 px | 0.00 px | 320.76 px | 479.57 px | 0.00 px |
| affine | 1 | 409.57 px | 58.86 px | 403.98 px | 664.72 px | 0.00 px |
| affine | 3 | 4608.39 px | 17849.18 px | 4137.57 px | 8485.77 px | 0.00 px |
| affine | 5 | 295.74 px | 165.24 px | 251.42 px | 569.50 px | 115.11 px |
| affine | 9 | 221.19 px | 28.59 px | 185.74 px | 416.47 px | 174.53 px |
| polynomial | 9 | 436.97 px | 222.27 px | 318.93 px | 979.13 px | 103.39 px |
| affine | 13 | 198.30 px | 24.83 px | 152.08 px | 400.86 px | 142.32 px |
| polynomial | 13 | 274.34 px | 86.94 px | 216.18 px | 561.27 px | 130.08 px |
| affine | 25 | 184.57 px | 12.52 px | 136.30 px | 381.10 px | 157.49 px |
| polynomial | 25 | 198.49 px | 16.73 px | 153.61 px | 398.37 px | 153.33 px |

关键结论：

1. 在本轮扩展 hold-out 实验中，25 点 affine 的平均误差最低，为 184.57 px。
2. 25 点 polynomial 的平均误差为 198.49 px，虽然不是本矩阵中的绝对最低值，但明显优于 9 点 affine 的 221.19 px，也显著优于 9 点 polynomial 的 436.97 px。
3. 9 点 polynomial 在少点条件下表现不稳定，说明 polynomial 需要足够多且覆盖充分的校准点。
4. 25 点策略整体优于 9 点策略，尤其在标准差和 P90 误差上更稳定。
5. 因此论文不应写成“25 点 polynomial 在所有实验中绝对最优”；更安全准确的表述是：扩展实验表明增加到 25 点后校准泛化误差和稳定性明显改善，最终系统结合屏幕覆盖完整性、运行期配置一致性和实时校准流程，采用 25 点 polynomial。

## 8. 论文第六章建议写法

第六章 6.5 建议改为“校准映射实验与最终 25 点校准验证”，但尽量不新增过多节号，以免破坏后续编号。

建议叙事顺序：

1. 先说明 9 点 affine / 9 点 polynomial 是少点校准策略对比，不是最终系统配置。
2. 再加入扩展实验，比较 13 点、25 点 affine / polynomial。
3. 明确 25 点结果来自离线 hold-out 扩展实验，Classic 运行期 25 点结果来自实时校准文件拟合残差。
4. 明确两类指标不能直接等同。
5. 最终结论写为“25 点提供更完整屏幕覆盖，并在扩展实验中显示更好的泛化稳定性；最终系统采用 25 点 polynomial 作为实时校准配置”。

可直接写入论文的核心表述建议：

> 原有 9 点 affine 与 9 点 polynomial 对比用于分析少点校准条件下不同映射函数的稳定性。实验结果表明，在校准点较少时，affine 映射的泛化误差低于 polynomial，说明二次多项式映射在样本覆盖不足时容易出现过拟合或局部外推不稳定。因此，9 点 affine 结果可作为少点校准策略的参考，但不能代表最终系统的实时校准配置。

> 为进一步解释最终系统采用 25 点校准的原因，本研究在相同测试集预测结果上扩展了校准点数和映射方式的对比。结果显示，25 点 affine 和 25 点 polynomial 均显著优于 9 点 polynomial，其中 25 点 polynomial 的 hold-out 平均误差为 198.49 px，低于 9 点 affine 的 221.19 px。该结果说明，当校准点覆盖屏幕中心、边缘和角落后，多项式映射的稳定性得到改善。

> 最终实时系统中 Classic 与 Deep 均配置为 25 点 polynomial 校准。当前可追溯的 Classic 运行期校准文件包含 25 个校准点，其平均拟合残差为 73.58 px，最大拟合残差为 212.54 px；Deep 运行期校准文件同样包含 25 个校准点，其平均拟合残差为 118.69 px，最大拟合残差为 251.97 px。该结果反映实时校准样本上的拟合情况，不等同于离线 hold-out 泛化误差。

## 9. 图 6-6 与表格建议

允许后续替换原图 6-6 为组合图“不同校准策略与最终 25 点校准结果对比”。组合图建议分为两部分：

| 分区 | 内容 | 指标 |
|---|---|---|
| A | 离线 hold-out 校准策略对比 | hold-out 平均像素误差 |
| B | 最终实时 25 点校准拟合结果 | 校准拟合残差 |

图中必须显式标注：

1. 9/13/25 点扩展实验为 hold-out 误差。
2. Classic 25 点运行期结果为校准拟合残差。
3. 两类指标不可直接等同。

第六章建议新增或替换核心表：

| 实验类型 | 模式 | 校准点数 | 映射方式 | 评价方式 | 平均误差/残差 | 最大误差/残差 | 数据来源 | 说明 |
|---|---|---:|---|---|---:|---:|---|---|
| 离线校准对比 | Deep/离线链路 | 0 | 无校准 | hold-out 像素误差 | 314.54 px | - | error_analysis 测试集 | 原始映射基线 |
| 离线校准对比 | Deep/离线链路 | 9 | affine | hold-out 像素误差 | 221.19 px | - | 扩展校准对比 | 少点条件下较稳定 |
| 离线校准对比 | Deep/离线链路 | 9 | polynomial | hold-out 像素误差 | 436.97 px | - | 扩展校准对比 | 少点条件下不稳定 |
| 离线校准对比 | Deep/离线链路 | 25 | affine | hold-out 像素误差 | 184.57 px | - | 扩展校准对比 | 本矩阵 hold-out 平均误差最低 |
| 离线校准对比 | Deep/离线链路 | 25 | polynomial | hold-out 像素误差 | 198.49 px | - | 扩展校准对比 | 最终 polynomial 配置的离线参考 |
| 最终实时校准 | Classic | 25 | polynomial | 校准拟合残差 | 73.58 px | 212.54 px | 运行期 calibration_classic.json | 不等同于 hold-out 误差 |
| 最终实时校准 | Deep | 25 | polynomial | 校准拟合残差 | 118.69 px | 251.97 px | 运行期 calibration_deep.json | 不等同于 hold-out 误差 |

## 10. 分辨率流程与论文使用流程建议

代码核对结论：

1. 校准点生成使用 `QApplication.primaryScreen()` 和 `screen.geometry()` 获取主屏幕尺寸。
2. 校准页面在无法获取屏幕时 fallback 为 1920x1080。
3. 主窗口和追踪页面也使用主屏幕几何尺寸设置全屏窗口。
4. 配置文件中的 1280x720 是摄像头请求分辨率，不是屏幕默认分辨率。
5. 摄像头模块会设置 `CAP_PROP_FRAME_WIDTH` 和 `CAP_PROP_FRAME_HEIGHT`，设置页也允许选择并检测实际摄像头分辨率。
6. 模型眼区输入尺寸 128x128 是眼区裁剪后的模型输入尺寸，不是屏幕分辨率，也不是摄像头原始分辨率。

论文流程建议补充：

> 用户选择 Classic 或 Deep 模式后，系统首先进入摄像头预览与设置确认阶段。该阶段用于确认摄像头请求分辨率、实际画面、人脸检测状态和眼区裁剪质量。与此同时，系统读取当前主屏幕几何尺寸，并在后续校准阶段根据主屏幕分辨率生成目标点坐标。屏幕分辨率直接影响校准点位置、目标点像素坐标和像素误差计算；摄像头分辨率则主要影响人脸检测、眼区裁剪质量、帧率和实时追踪稳定性。

图 5-12 后续应加入“分辨率与显示环境确认”节点，建议流程为：

```text
启动程序 -> 选择追踪模式 -> 摄像头预览 -> 分辨率与显示环境确认 -> 25 点校准 -> 实时追踪 -> 全屏验证 -> 注视交互
```

附录 B 建议顺序：

| 小节 | 内容 |
|---|---|
| B.1 | 运行环境 |
| B.2 | 启动与模式选择 |
| B.3 | 摄像头预览 |
| B.4 | 确认分辨率 |
| B.5 | 25 点校准 |
| B.6 | 实时追踪与全屏验证 |
| B.7 | 注视交互演示 |
| B.8 | 常见问题与处理方式 |

B.4 建议使用 `docs/final-project-paper/引用图片/设置页截图.png`，图题为“图 B-3 分辨率与系统设置界面”。后续附录图号需要顺延。

## 11. 后续 docx 更新计划

后续使用 Documents 插件修改 docx 时，建议按以下顺序执行：

1. 备份当前 docx。
2. 只修改第五章流程说明、第六章 6.5 校准实验小节、图 5-12、图 6-6、附录 B。
3. 不修改公式、摘要、参考文献和附录 A 程序清单主体。
4. 将第六章 6.5 标题或段落改为“校准映射实验与最终 25 点校准验证”。
5. 替换原图 6-6 为组合图，并更新图题为“不同校准策略与最终 25 点校准结果对比”。
6. 插入或替换校准结果表，采用第 9 节中的表格结构。
7. 第五章流程文字中加入分辨率检测/确认。
8. 图 5-12 中在“摄像头预览”后加入“分辨率与显示环境确认”。
9. 附录 B 在 B.3 后新增 B.4“确认分辨率”，并使用设置页截图。
10. 更新附录 B 后续图号和正文引用。
11. 渲染检查 docx 页面，重点检查图片位置、题注编号、表格分页、公式编号和参考文献编号。

## 12. 必须避免的论文写法

后续写入论文时应避免以下表述：

1. 不写“9 点 affine 是最终系统配置”。
2. 不写“25 点 polynomial 在所有实验中绝对最优”。
3. 不把 Classic 25 点校准拟合残差写成 hold-out 泛化误差。
4. 不使用近似值替代 Deep `calibration_deep.json` 的真实拟合残差；当前 Deep 数值必须来自 2026-05-09 生成的运行期校准文件。
5. 不把 1280x720 写成屏幕默认分辨率。
6. 不把 128x128 写成摄像头分辨率或屏幕分辨率。
7. 不说自建数据集、模型权重或 ONNX 模型公开发布。

## 13. 当前推荐结论

本轮实验支持如下论文结论：

1. 9 点 affine / polynomial 是少点校准策略对比实验，用于说明少点条件下 affine 更稳定。
2. 扩展到 25 点后，校准策略整体优于 9 点策略；25 点 affine 在本轮离线 hold-out 矩阵中平均误差最低。
3. 25 点 polynomial 虽不是本矩阵绝对最低，但相较 9 点 polynomial 大幅改善，并优于 9 点 affine；这说明 polynomial 映射需要更充分的屏幕覆盖。
4. 最终系统采用 25 点 polynomial，合理依据应写为“屏幕覆盖完整性、实时系统配置一致性和扩展实验稳定性共同支持”，而不是简单写成“25 点 polynomial 所有指标最好”。
5. Classic 与 Deep 运行期 25 点 polynomial 校准均已有可追溯拟合残差，可分别用于最终实时校准结果表；两者均不能表述为 hold-out 泛化误差。
