"""GazeSmoother Property-Based 测试。

Feature: look2act-tracker, Property 10: EMA 平滑常数输入收敛
"""
from hypothesis import given, settings
from hypothesis import strategies as st

from src.tracker.smoother import GazeSmoother


class TestEMASmoothConvergence:
    """Property 10: EMA 平滑常数输入收敛

    对于任意常数注视点 (cx, cy) 和任意平滑系数 alpha，
    连续输入 N 次相同的点后（N 足够大），
    EMA 滤波器的输出应收敛到 (cx, cy)（允许误差 1e-3）。

    **Validates: Requirements 5.6**
    """

    @given(
        cx=st.floats(min_value=0.0, max_value=1920.0,
                     allow_nan=False, allow_infinity=False),
        cy=st.floats(min_value=0.0, max_value=1080.0,
                     allow_nan=False, allow_infinity=False),
        alpha=st.floats(min_value=0.01, max_value=1.0,
                        allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_constant_input_convergence(
        self, cx: float, cy: float, alpha: float
    ):
        """连续输入相同点后输出应收敛到该点。

        **Validates: Requirements 5.6**
        """
        smoother = GazeSmoother(alpha=alpha)

        # 输入足够多次（200 次足以让任何 alpha 收敛）
        result = (0.0, 0.0)
        for _ in range(200):
            result = smoother.update((cx, cy))

        assert abs(result[0] - cx) < 1e-3, (
            f"X 未收敛: 期望 {cx}, 实际 {result[0]}"
        )
        assert abs(result[1] - cy) < 1e-3, (
            f"Y 未收敛: 期望 {cy}, 实际 {result[1]}"
        )
