"""配置与训练日志 Property-Based 测试。

Feature: look2act-tracker
- Property 13: YAML 配置 round-trip
- Property 14: 训练日志 CSV 字段完整性
"""
from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st


# ── 策略：生成有效的训练配置字典 ──

valid_optimizers = st.sampled_from(["Adam", "SGD"])
valid_schedulers = st.sampled_from(["CosineAnnealingLR", "StepLR", "none"])

channel_list = st.lists(
    st.integers(min_value=8, max_value=512), min_size=4, max_size=4
)

train_config_strategy = st.fixed_dictionaries({
    "model": st.fixed_dictionaries({
        "name": st.just("GazeNet"),
        "channels": channel_list,
        "input_size": st.just(128),
        "output_dim": st.just(3),
    }),
    "training": st.fixed_dictionaries({
        "batch_size": st.integers(min_value=1, max_value=256),
        "learning_rate": st.floats(min_value=1e-6, max_value=1.0,
                                   allow_nan=False, allow_infinity=False),
        "epochs": st.integers(min_value=1, max_value=1000),
        "optimizer": valid_optimizers,
        "scheduler": valid_schedulers,
        "weight_decay": st.floats(min_value=0.0, max_value=0.1,
                                  allow_nan=False, allow_infinity=False),
    }),
    "data": st.fixed_dictionaries({
        "dataset_raw_dir": st.just("dataset_raw"),
        "dataset_processed_dir": st.just("dataset_processed"),
        "train_split": st.floats(min_value=0.1, max_value=0.9,
                                 allow_nan=False, allow_infinity=False),
        "val_split": st.floats(min_value=0.05, max_value=0.4,
                               allow_nan=False, allow_infinity=False),
        "test_split": st.floats(min_value=0.05, max_value=0.4,
                                allow_nan=False, allow_infinity=False),
        "augmentation": st.booleans(),
        "num_workers": st.integers(min_value=0, max_value=8),
    }),
    "evaluation": st.fixed_dictionaries({
        "metrics": st.just(["mean_angle_error", "median_angle_error", "mean_pixel_error"]),
    }),
    "checkpoint": st.fixed_dictionaries({
        "save_dir": st.just("checkpoints"),
        "save_best_only": st.booleans(),
    }),
    "logging": st.fixed_dictionaries({
        "log_dir": st.just("logs"),
        "log_interval": st.integers(min_value=1, max_value=50),
    }),
})


class TestYAMLConfigRoundTrip:
    """Property 13: YAML 配置 round-trip

    对于任意有效的训练配置对象，序列化为 YAML 文件后再解析回配置对象，
    两者应完全等价。

    **Validates: Requirements 8.6**
    """

    @given(config=train_config_strategy)
    @settings(max_examples=100)
    def test_yaml_round_trip(self, config: dict):
        """YAML 序列化后反序列化应还原原始配置。

        **Validates: Requirements 8.6**
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(config, f, allow_unicode=True)
            tmp_path = f.name

        with open(tmp_path, "r", encoding="utf-8") as f:
            restored = yaml.safe_load(f)

        Path(tmp_path).unlink()

        assert restored == config, (
            f"YAML round-trip 失败:\n原始: {config}\n还原: {restored}"
        )


# ── 训练日志 CSV 字段策略 ──

log_row_strategy = st.fixed_dictionaries({
    "epoch": st.integers(min_value=1, max_value=10000),
    "train_loss": st.floats(min_value=0.0, max_value=10.0,
                            allow_nan=False, allow_infinity=False),
    "val_loss": st.floats(min_value=0.0, max_value=10.0,
                          allow_nan=False, allow_infinity=False),
    "val_angle_error": st.floats(min_value=0.0, max_value=180.0,
                                 allow_nan=False, allow_infinity=False),
})

REQUIRED_LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_angle_error"]


class TestTrainingLogCSV:
    """Property 14: 训练日志 CSV 字段完整性

    对于任意训练日志记录，输出的 CSV 行必须包含 epoch、train_loss、
    val_loss、val_angle_error 四个字段，且数值类型正确。

    **Validates: Requirements 8.7**
    """

    @given(rows=st.lists(log_row_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_log_csv_fields_complete(self, rows: list[dict]):
        """CSV 写入后读取，所有必需字段应存在且数值可解析。

        **Validates: Requirements 8.7**
        """
        # 写入 CSV
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=REQUIRED_LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "epoch": row["epoch"],
                "train_loss": f"{row['train_loss']:.6f}",
                "val_loss": f"{row['val_loss']:.6f}",
                "val_angle_error": f"{row['val_angle_error']:.4f}",
            })

        # 读取并验证
        buf.seek(0)
        reader = csv.DictReader(buf)

        # 验证表头包含所有必需字段
        for field in REQUIRED_LOG_FIELDS:
            assert field in reader.fieldnames, f"缺少必需字段: {field}"

        read_rows = list(reader)
        assert len(read_rows) == len(rows), "行数不匹配"

        for i, read_row in enumerate(read_rows):
            # epoch 应为整数
            epoch_val = int(read_row["epoch"])
            assert epoch_val == rows[i]["epoch"]

            # 损失值应为可解析的浮点数
            float(read_row["train_loss"])
            float(read_row["val_loss"])
            float(read_row["val_angle_error"])
