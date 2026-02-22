# CPU 优化策略说明

## 概述

Look2Act Tracker 采用分阶段的优化策略，确保系统在各种硬件环境下都能高效运行。

## 优化策略

### 1. 训练阶段优化

**默认启用 IPEX 加速训练**

系统默认使用 Intel XPU (Arc GPU) + IPEX 加速模型训练，配置位于 `configs/train_config.yaml`：

```yaml
training:
  device: "xpu"      # 使用 Intel Arc GPU
  use_ipex: true     # 启用 IPEX 优化
```

训练脚本会自动检测硬件环境并应用优化：

```python
# scripts/train.py 自动处理
if device.type == "xpu" and use_ipex:
    import intel_extension_for_pytorch as ipex
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
```

**回退机制**：
- 如果 XPU 不可用，自动回退到 CPU
- 如果 IPEX 未安装，自动回退到标准 PyTorch

**手动切换到 CPU 训练**：

```yaml
training:
  device: "cpu"
  use_ipex: false
```

**注意**：IPEX 优化仅在训练阶段使用，不影响推理系统的兼容性。

### 2. 推理阶段优化

**默认使用 ONNX Runtime**

推理系统默认使用 ONNX Runtime 作为推理引擎，确保跨平台兼容性：

- ✓ 跨平台支持（Windows/Linux/macOS）
- ✓ CPU 优化（无需 GPU）
- ✓ 轻量级依赖
- ✓ 高性能推理

**配置方式**（`configs/system_config.yaml`）：

```yaml
model:
  use_onnx: true   # 使用 ONNX Runtime
  use_ipex: false  # 推理不使用 IPEX
  onnx_path: "checkpoints/gaze_net.onnx"
```

### 3. 批处理优化

**左右眼合并推理（batch=2）**

系统自动将左右眼图像合并为 batch=2 进行推理，减少推理调用次数：

```python
# 准备输入：左右眼合并为 batch=2
left_tensor = torch.from_numpy(left_eye).permute(2, 0, 1).float() / 255.0
right_tensor = torch.from_numpy(right_eye).permute(2, 0, 1).float() / 255.0
batch = torch.stack([left_tensor, right_tensor], dim=0)  # (2, 3, 128, 128)

# 批处理推理
gaze_vectors = model(batch)  # (2, 3)
gaze_vector = gaze_vectors.mean(dim=0)  # 平均左右眼结果
```

**性能提升**：批处理推理比逐个推理快约 1.5-2 倍。

## 模型导出

### 导出 ONNX 模型

使用提供的脚本将训练好的 PyTorch 模型导出为 ONNX 格式：

```bash
conda run -n gaze-env python scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --output checkpoints/gaze_net.onnx
```

脚本会自动验证导出的 ONNX 模型与原始 PyTorch 模型的输出一致性。

## 性能指标

在 Intel Ultra 5 125H CPU 上的性能表现：

| 模块 | 耗时 (ms) |
|------|-----------|
| 人脸检测 | ~15 |
| 头部姿态估计 | ~2 |
| 视线回归（batch=2） | ~10 |
| 坐标转换 | <1 |
| 射线-平面求交 | <1 |
| 时序平滑 | <1 |
| **总计** | **~30** |

**端到端帧率**：约 30 FPS（满足 15 FPS 的需求）

## 兼容性说明

### 推理系统兼容性

- ✓ 无需 CUDA
- ✓ 无需 IPEX（可选）
- ✓ 仅依赖 ONNX Runtime + CPU
- ✓ 可在各种 Intel/AMD CPU 上运行

### 训练系统兼容性

- **默认配置**：使用 Intel XPU + IPEX 加速训练
- **回退支持**：自动检测硬件，不可用时回退到 CPU
- **纯 CPU 训练**：可手动配置为 CPU 模式（无需 IPEX）
- **模型导出**：导出的 ONNX 模型与训练方式无关，确保跨平台兼容

## 正确性验证

系统通过 Property-Based Testing 验证优化的正确性：

### Property 15: 模型格式转换输出一致性

验证 ONNX 模型输出与 PyTorch 模型输出一致（误差 < 1e-4）。

### Property 16: 批处理与逐个推理一致性

验证 batch=2 批处理推理与逐个推理结果一致（误差 < 1e-6）。

## 使用建议

1. **训练加速**：默认使用 XPU + IPEX（已配置），无需手动设置
2. **推理部署**：使用 ONNX Runtime（默认配置）
3. **跨平台部署**：使用 ONNX 模型确保兼容性
4. **性能调优**：根据实际硬件调整 batch size 和模型参数
5. **环境切换**：如需 CPU 训练，修改 `train_config.yaml` 中的 `device` 参数

## 参考资料

- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)
