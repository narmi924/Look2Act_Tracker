"""UI Preservation 保持性属性测试。

验证非 bug 条件下的现有行为在修复前后保持一致。
这些测试在未修复代码上运行时应该 PASS（确认基线行为可保持）。

使用源码分析方法（读取文件内容检查字符串/结构），避免实例化 Qt 对象。

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 观察 1：FluentWindow 页面导航结构
# ---------------------------------------------------------------------------

class TestPreservationNavigation:
    """观察 1：FluentWindow 页面导航结构保持不变。

    验证 main_window.py 中 addSubInterface 调用了 5 个页面，
    以及 switchTo 方法在各导航方法中被调用。

    **Validates: Requirements 3.1**
    """

    EXPECTED_PAGES = [
        "page_home",
        "page_camera",
        "page_calibration",
        "page_tracking",
        "page_settings",
    ]

    EXPECTED_NAV_METHODS = [
        "go_home",
        "go_camera",
        "go_calibration",
        "go_tracking",
        "go_settings",
    ]

    @pytest.fixture(scope="class")
    def main_window_source(self) -> str:
        """读取 main_window.py 源码。"""
        return (PROJECT_ROOT / "src" / "ui" / "main_window.py").read_text(
            encoding="utf-8"
        )

    def test_add_sub_interface_calls(self, main_window_source: str):
        """验证 addSubInterface 调用了 5 个页面。"""
        matches = re.findall(r"self\.addSubInterface\(self\.(page_\w+)", main_window_source)
        assert len(matches) == 5, (
            f"期望 5 个 addSubInterface 调用，实际找到 {len(matches)} 个: {matches}"
        )
        for page in self.EXPECTED_PAGES:
            assert page in matches, f"缺少 addSubInterface 调用: self.{page}"

    def test_switch_to_in_nav_methods(self, main_window_source: str):
        """验证 switchTo 方法在各导航方法中被调用。"""
        for method_name in self.EXPECTED_NAV_METHODS:
            # 提取方法体
            pattern = rf"def {method_name}\(self\).*?(?=\n    def |\nclass |\Z)"
            match = re.search(pattern, main_window_source, re.DOTALL)
            assert match is not None, f"未找到方法: {method_name}"
            method_body = match.group()
            assert "self.switchTo(" in method_body, (
                f"{method_name} 方法中未调用 self.switchTo()"
            )

    @given(page_index=st.integers(min_value=0, max_value=4))
    @settings(max_examples=20)
    def test_navigation_page_registered(self, page_index: int):
        """属性测试：随机选择页面索引，验证对应页面已注册到导航。"""
        source = (PROJECT_ROOT / "src" / "ui" / "main_window.py").read_text(
            encoding="utf-8"
        )
        page_name = self.EXPECTED_PAGES[page_index]
        assert f"self.{page_name}" in source, (
            f"页面 {page_name} 未在 main_window.py 中定义"
        )
        assert f"self.addSubInterface(self.{page_name}" in source, (
            f"页面 {page_name} 未通过 addSubInterface 注册到导航"
        )


# ---------------------------------------------------------------------------
# 观察 2：GazeCursorOverlay 窗口标志
# ---------------------------------------------------------------------------

class TestPreservationGazeCursorOverlay:
    """观察 2：GazeCursorOverlay 窗口标志保持不变。

    验证 tracking_page.py 中 GazeCursorOverlay 设置了正确的窗口标志和属性。

    **Validates: Requirements 3.3**
    """

    EXPECTED_WINDOW_FLAGS = [
        "FramelessWindowHint",
        "WindowStaysOnTopHint",
        "Tool",
        "WindowTransparentForInput",
    ]

    EXPECTED_ATTRIBUTES = [
        "WA_TranslucentBackground",
        "WA_TransparentForMouseEvents",
    ]

    @pytest.fixture(scope="class")
    def tracking_source(self) -> str:
        """读取 tracking_page.py 源码。"""
        return (PROJECT_ROOT / "src" / "ui" / "tracking_page.py").read_text(
            encoding="utf-8"
        )

    def test_window_flags_present(self, tracking_source: str):
        """验证 GazeCursorOverlay 设置了所有必需的窗口标志。"""
        for flag in self.EXPECTED_WINDOW_FLAGS:
            assert flag in tracking_source, (
                f"GazeCursorOverlay 缺少窗口标志: {flag}"
            )

    def test_widget_attributes_present(self, tracking_source: str):
        """验证 GazeCursorOverlay 设置了所有必需的 widget 属性。"""
        for attr in self.EXPECTED_ATTRIBUTES:
            assert attr in tracking_source, (
                f"GazeCursorOverlay 缺少 widget 属性: {attr}"
            )

    @given(flag_index=st.integers(min_value=0, max_value=3))
    @settings(max_examples=20)
    def test_random_flag_preserved(self, flag_index: int):
        """属性测试：随机选择窗口标志，验证其存在于源码中。"""
        source = (PROJECT_ROOT / "src" / "ui" / "tracking_page.py").read_text(
            encoding="utf-8"
        )
        flag = self.EXPECTED_WINDOW_FLAGS[flag_index]
        assert flag in source, (
            f"GazeCursorOverlay 窗口标志 {flag} 丢失"
        )


# ---------------------------------------------------------------------------
# 观察 3：CalibrationFullscreenWidget 信号连接
# ---------------------------------------------------------------------------

class TestPreservationCalibrationSignals:
    """观察 3：CalibrationFullscreenWidget 信号连接保持不变。

    验证 calibration_page.py 中 calibration_finished 和 calibration_cancelled
    信号被定义，且在 _handle_start_calibration 中连接了这两个信号。

    **Validates: Requirements 3.5**
    """

    EXPECTED_SIGNALS = [
        "calibration_finished",
        "calibration_cancelled",
    ]

    @pytest.fixture(scope="class")
    def calibration_source(self) -> str:
        """读取 calibration_page.py 源码。"""
        return (PROJECT_ROOT / "src" / "ui" / "calibration_page.py").read_text(
            encoding="utf-8"
        )

    def test_signals_defined(self, calibration_source: str):
        """验证 CalibrationFullscreenWidget 定义了 calibration_finished 和 calibration_cancelled 信号。"""
        for signal in self.EXPECTED_SIGNALS:
            pattern = rf"{signal}\s*=\s*pyqtSignal"
            assert re.search(pattern, calibration_source), (
                f"CalibrationFullscreenWidget 缺少信号定义: {signal}"
            )

    def test_signals_connected_in_handle_start(self, calibration_source: str):
        """验证 _handle_start_calibration 中连接了 calibration_finished 和 calibration_cancelled 信号。"""
        # 提取 _handle_start_calibration 方法体
        pattern = r"def _handle_start_calibration\(self\).*?(?=\n    def |\nclass |\Z)"
        match = re.search(pattern, calibration_source, re.DOTALL)
        assert match is not None, "未找到 _handle_start_calibration 方法"
        method_body = match.group()

        for signal in self.EXPECTED_SIGNALS:
            assert f"{signal}.connect(" in method_body, (
                f"_handle_start_calibration 中未连接信号: {signal}"
            )

    @given(signal_index=st.integers(min_value=0, max_value=1))
    @settings(max_examples=20)
    def test_random_signal_preserved(self, signal_index: int):
        """属性测试：随机选择信号，验证其定义和连接。"""
        source = (PROJECT_ROOT / "src" / "ui" / "calibration_page.py").read_text(
            encoding="utf-8"
        )
        signal = self.EXPECTED_SIGNALS[signal_index]

        # 信号定义存在
        assert re.search(rf"{signal}\s*=\s*pyqtSignal", source), (
            f"信号 {signal} 未定义"
        )

        # 信号在 _handle_start_calibration 中被连接
        handle_match = re.search(
            r"def _handle_start_calibration\(self\).*?(?=\n    def |\nclass |\Z)",
            source,
            re.DOTALL,
        )
        assert handle_match is not None
        assert f"{signal}.connect(" in handle_match.group(), (
            f"信号 {signal} 未在 _handle_start_calibration 中连接"
        )


# ---------------------------------------------------------------------------
# 观察 4：SettingsPage 配置保存
# ---------------------------------------------------------------------------

class TestPreservationSettingsSave:
    """观察 4：SettingsPage 配置保存功能保持不变。

    验证 settings_page.py 中 _handle_save 方法存在，
    且使用了 yaml.dump 写入配置。

    **Validates: Requirements 3.2**
    """

    @pytest.fixture(scope="class")
    def settings_source(self) -> str:
        """读取 settings_page.py 源码。"""
        return (PROJECT_ROOT / "src" / "ui" / "settings_page.py").read_text(
            encoding="utf-8"
        )

    def test_handle_save_exists(self, settings_source: str):
        """验证 _handle_save 方法存在。"""
        assert "def _handle_save(self)" in settings_source, (
            "SettingsPage 缺少 _handle_save 方法"
        )

    def test_yaml_dump_used(self, settings_source: str):
        """验证 _handle_save 使用了 yaml.dump 写入配置。"""
        # 提取 _handle_save 方法体
        pattern = r"def _handle_save\(self\).*?(?=\n    def |\nclass |\Z)"
        match = re.search(pattern, settings_source, re.DOTALL)
        assert match is not None, "未找到 _handle_save 方法"
        method_body = match.group()

        assert "yaml.dump(" in method_body, (
            "_handle_save 方法中未使用 yaml.dump 写入配置"
        )

    def test_config_path_defined(self, settings_source: str):
        """验证 SettingsPage 定义了配置文件路径。"""
        assert "system_config.yaml" in settings_source, (
            "SettingsPage 未引用 system_config.yaml 配置文件路径"
        )


# ---------------------------------------------------------------------------
# 观察 5：fluent_theme.py 主题设置
# ---------------------------------------------------------------------------

class TestPreservationFluentTheme:
    """观察 5：fluent_theme.py 主题设置保持不变。

    验证 apply_fluent_theme 函数存在，调用了 setTheme(Theme.LIGHT) 和 setThemeColor，
    PALETTE 字典包含必需的键，以及 app.setStyleSheet 被调用。

    **Validates: Requirements 3.1, 3.2**
    """

    EXPECTED_PALETTE_KEYS = ["bg", "card", "text", "accent", "red_exit"]

    @pytest.fixture(scope="class")
    def theme_source(self) -> str:
        """读取 fluent_theme.py 源码。"""
        return (PROJECT_ROOT / "src" / "ui" / "fluent_theme.py").read_text(
            encoding="utf-8"
        )

    def test_apply_fluent_theme_exists(self, theme_source: str):
        """验证 apply_fluent_theme 函数存在。"""
        assert "def apply_fluent_theme(" in theme_source, (
            "fluent_theme.py 缺少 apply_fluent_theme 函数"
        )

    def test_set_theme_light(self, theme_source: str):
        """验证调用了 setTheme(Theme.LIGHT)。"""
        assert "setTheme(Theme.LIGHT)" in theme_source, (
            "apply_fluent_theme 未调用 setTheme(Theme.LIGHT)"
        )

    def test_set_theme_color(self, theme_source: str):
        """验证调用了 setThemeColor。"""
        assert "setThemeColor(" in theme_source, (
            "apply_fluent_theme 未调用 setThemeColor"
        )

    def test_palette_keys(self, theme_source: str):
        """验证 PALETTE 字典包含所有必需的键。"""
        for key in self.EXPECTED_PALETTE_KEYS:
            pattern = rf'"{key}"'
            assert re.search(pattern, theme_source), (
                f"PALETTE 字典缺少键: {key}"
            )

    def test_app_set_stylesheet(self, theme_source: str):
        """验证 app.setStyleSheet 被调用。"""
        assert "app.setStyleSheet(" in theme_source, (
            "apply_fluent_theme 未调用 app.setStyleSheet"
        )

    @given(key_index=st.integers(min_value=0, max_value=4))
    @settings(max_examples=20)
    def test_random_palette_key_preserved(self, key_index: int):
        """属性测试：随机选择 PALETTE 键，验证其存在于源码中。"""
        source = (PROJECT_ROOT / "src" / "ui" / "fluent_theme.py").read_text(
            encoding="utf-8"
        )
        key = self.EXPECTED_PALETTE_KEYS[key_index]
        assert f'"{key}"' in source, (
            f"PALETTE 键 {key} 丢失"
        )
