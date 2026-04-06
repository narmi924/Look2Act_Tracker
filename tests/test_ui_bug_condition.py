"""UI Bug Condition 探索性测试。

验证 5 个 bug 条件在未修复代码上存在。
这些测试编码的是**期望行为**（修复后应该通过的状态）。
在未修复代码上运行时，测试应该 FAIL（证明 bug 存在）。

使用源码分析（读取文件内容检查字符串）而非实例化 Qt 对象。

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestBugConditionA:
    """条件 A：校准全屏失败。

    验证 CalibrationPage._handle_start_calibration 创建
    CalibrationFullscreenWidget 时不应传入 parent=self。
    未修复代码中传入了 parent=self，导致 showFullScreen() 无效。

    **Validates: Requirements 1.1**
    """

    def test_calibration_widget_no_parent_self(self):
        """CalibrationFullscreenWidget 构造调用不应包含 parent=self。"""
        source = (PROJECT_ROOT / "src" / "ui" / "calibration_page.py").read_text(
            encoding="utf-8"
        )

        # 查找 CalibrationFullscreenWidget( 调用中是否包含 parent=self
        # 期望行为：不包含 parent=self（修复后通过）
        pattern = r"CalibrationFullscreenWidget\([^)]*parent\s*=\s*self[^)]*\)"
        match = re.search(pattern, source)
        assert match is None, (
            f"CalibrationFullscreenWidget 构造时传入了 parent=self，"
            f"这会导致 showFullScreen() 无法正确全屏。"
            f"\n找到: {match.group()}"
        )


class TestBugConditionB:
    """条件 B：样式覆盖。

    验证 MainWindow 源码中不应包含覆盖性 setStyleSheet 调用
    （包含 background: transparent）。
    未修复代码中存在此覆盖，导致 fluent_theme 主题失效。

    **Validates: Requirements 1.2**
    """

    def test_main_window_no_override_stylesheet(self):
        """MainWindow 源码不应包含 'background: transparent' 的 setStyleSheet。"""
        source = (PROJECT_ROOT / "src" / "ui" / "main_window.py").read_text(
            encoding="utf-8"
        )

        # 检查是否有 self.setStyleSheet 包含 background: transparent
        # 期望行为：不包含此覆盖性样式（修复后通过）
        has_override = (
            "self.setStyleSheet" in source
            and "background: transparent" in source
        )
        assert not has_override, (
            "MainWindow 中存在覆盖性 setStyleSheet 调用（包含 'background: transparent'），"
            "这会覆盖 fluent_theme.py 设定的浅米色背景和深褐色主题。"
        )


class TestBugConditionC:
    """条件 C：硬编码颜色。

    验证 home_page.py、tracking_page.py、camera_page.py、settings_page.py
    源码中不应包含硬编码 #0078D4。
    未修复代码中多处使用了此蓝色值，与 PALETTE["accent"] 不一致。

    **Validates: Requirements 1.3, 1.4**
    """

    UI_FILES = [
        "home_page.py",
        "tracking_page.py",
        "camera_page.py",
        "settings_page.py",
    ]

    @pytest.mark.parametrize("filename", UI_FILES)
    def test_no_hardcoded_blue(self, filename: str):
        """UI 文件中不应包含硬编码 #0078D4 蓝色值。"""
        filepath = PROJECT_ROOT / "src" / "ui" / filename
        source = filepath.read_text(encoding="utf-8")

        # 期望行为：不包含 #0078D4（修复后通过）
        assert "#0078D4" not in source, (
            f"{filename} 中包含硬编码颜色 #0078D4，"
            f"应使用 PALETTE['accent']（#452829）替代。"
        )


class TestBugConditionD:
    """条件 D：缺少窗口模式配置。

    验证 system_config.yaml 应包含 ui.window_mode 配置键。
    未修复代码中缺少此配置键。

    **Validates: Requirements 1.5**
    """

    def test_config_has_window_mode(self):
        """system_config.yaml 应包含 ui -> window_mode 配置键。"""
        config_path = PROJECT_ROOT / "configs" / "system_config.yaml"
        assert config_path.exists(), f"配置文件不存在: {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 期望行为：存在 ui.window_mode 键（修复后通过）
        ui_section = data.get("ui")
        assert ui_section is not None, (
            "system_config.yaml 中缺少 'ui' 配置节。"
        )
        assert "window_mode" in ui_section, (
            "system_config.yaml 的 'ui' 节中缺少 'window_mode' 配置键，"
            "应包含 'fullscreen' 或 'adaptive' 值。"
        )


class TestBugConditionE:
    """条件 E：卡片固定尺寸。

    验证 FeatureCard 源码中不应使用 setFixedHeight(280) 固定高度。
    未修复代码中使用了固定像素值，不随屏幕分辨率缩放。

    **Validates: Requirements 1.6**
    """

    def test_feature_card_no_fixed_height(self):
        """FeatureCard 不应使用 setFixedHeight(280) 固定高度。"""
        source = (PROJECT_ROOT / "src" / "ui" / "home_page.py").read_text(
            encoding="utf-8"
        )

        # 期望行为：不包含 setFixedHeight(280)（修复后通过）
        assert "setFixedHeight(280)" not in source, (
            "FeatureCard 使用了 setFixedHeight(280) 固定高度，"
            "应根据屏幕尺寸动态计算卡片高度。"
        )
