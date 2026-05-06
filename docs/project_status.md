# Look2Act 当前项目状态

更新时间：2026-05-06

## 当前结论

Look2Act 已经从“离线指标可讲但实时体验不稳定”的状态，恢复到可演示、可诊断、可继续研究的状态。项目现在保留两条可运行路线：

- **Classic Demo**：工程体验兜底线。使用 Eye_Touch 风格的经典图像处理链路，适合展示稳定的视线跟随、全屏验证、九宫格交互和 5x5 井字棋交互。
- **Deep Demo**：机器学习研究演示线。当前有效配置为 camera-space 3D gaze + zero pose + zero origin + 720mm screen plane + swapped eye input + EMA smoothing。该路线已经能在实时校准后产生较清晰的屏幕拓扑，但仍需要进一步研究头部运动鲁棒性和正式评估。

## 推荐演示命令

在 Git Bash 中运行：

```bash
conda activate gaze-env
cd /d/Projects/Look2Act_Tracker_Project
python main.py --config configs/experiments/system_classic_demo.yaml
```

Deep 演示：

```bash
conda activate gaze-env
cd /d/Projects/Look2Act_Tracker_Project
python main.py --config configs/experiments/system_deep_demo.yaml
```

等价的深度研究配置仍保留为：

```bash
python main.py --config configs/experiments/system_deep_camera_zero_720_pose_zero_swap_ema.yaml
```

## 已完成

- 引入 Classic backend，恢复可演示的视线追踪体验。
- 完成“校准 -> 全屏验证 -> 全屏交互”的独立窗口流程。
- 增加三种语言模式入口：中文、English、双语。
- 修正 Deep 3D runtime 契约的关键实验方向，确认旧链路中的 head-space/runtime 假设会造成拓扑折叠。
- 收敛 Deep Demo 配置，实时 raw topology 已能明显随注视点变化。
- 增加诊断、拓扑分析、3D projection variant 评估与研究日志。

## 仍需推进

- Deep 路线需要正式离线评估、实时多次重复校准记录和论文级对照实验。
- 当前 Deep Demo 对头部运动较敏感，演示时应保持头部稳定，主要移动眼睛。
- 后续可继续研究 screen distance、头部姿态补偿、个人少量数据微调与虚拟数据集增强。
- README、论文正文和最终答辩材料需要同步当前事实，避免继续使用旧的 head-local 叙事。

## 仓库约定

- `configs/experiments/system_classic_demo.yaml`：稳定演示配置。
- `configs/experiments/system_deep_demo.yaml`：当前最佳 Deep Demo 配置。
- `docs/research/3d_contract_runtime_repair_2026-05-06.md`：3D 契约修复实验日志。
- `docs/research/3d_gaze_to_screen_stage_summary_2026-05-06.md`：3D 路线阶段总结。
- `tools/manual_checks/`：手工 UI 检查脚本，不参与默认 pytest。
