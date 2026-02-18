"""几何计算模块 Property-Based 测试。

Feature: look2act-tracker
- Property 8: 坐标转换恒等变换
- Property 9: 射线-平面求交点在平面上
"""
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.geometry.coordinate import transform_gaze_to_camera


def unit_vectors():
    """生成随机 3D 单位向量。"""
    return arrays(
        dtype=np.float64, shape=(3,),
        elements=st.floats(min_value=-1.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False),
    ).filter(lambda v: np.linalg.norm(v) > 0.1).map(
        lambda v: v / np.linalg.norm(v)
    )


def translation_vectors():
    """生成随机平移向量。"""
    return arrays(
        dtype=np.float64, shape=(3,),
        elements=st.floats(min_value=-1000.0, max_value=1000.0,
                           allow_nan=False, allow_infinity=False),
    )


class TestCoordinateTransformIdentity:
    """Property 8: 坐标转换恒等变换

    对于任意视线方向向量，当旋转矩阵为单位矩阵（无旋转）时，
    坐标转换后的视线方向应与原始方向一致（允许浮点误差 1e-6）。

    **Validates: Requirements 5.1**
    """

    @given(gaze=unit_vectors(), tvec=translation_vectors())
    @settings(max_examples=100)
    def test_identity_rotation_preserves_direction(
        self, gaze: np.ndarray, tvec: np.ndarray
    ):
        """单位旋转矩阵下，转换后视线方向不变。

        **Validates: Requirements 5.1**
        """
        identity = np.eye(3)
        origin, direction = transform_gaze_to_camera(gaze, identity, tvec)

        # 方向应不变
        assert np.allclose(direction, gaze, atol=1e-6), (
            f"方向应不变: 原始={gaze}, 转换后={direction}"
        )

        # 起点应为平移向量
        assert np.allclose(origin, tvec.flatten(), atol=1e-6), (
            f"起点应为 tvec: tvec={tvec.flatten()}, origin={origin}"
        )


from src.geometry.ray_plane import ray_plane_intersect


def plane_normals():
    """生成随机平面法向量（非零）。"""
    return arrays(
        dtype=np.float64, shape=(3,),
        elements=st.floats(min_value=-10.0, max_value=10.0,
                           allow_nan=False, allow_infinity=False),
    ).filter(lambda v: np.linalg.norm(v) > 0.1)


def plane_points():
    """生成随机平面上的点。"""
    return arrays(
        dtype=np.float64, shape=(3,),
        elements=st.floats(min_value=-500.0, max_value=500.0,
                           allow_nan=False, allow_infinity=False),
    )


class TestRayPlaneIntersection:
    """Property 9: 射线-平面求交点在平面上

    对于任意非平行于平面的射线和平面，计算得到的交点 P 必须同时满足：
    (1) P 在平面上，即 N·(P - P0) ≈ 0
    (2) P 在射线上，即存在 t > 0 使得 P = O + t*D

    **Validates: Requirements 5.2, 5.5**
    """

    @given(
        ray_origin=plane_points(),
        ray_dir=unit_vectors(),
        plane_pt=plane_points(),
        plane_n=plane_normals(),
    )
    @settings(max_examples=100)
    def test_intersection_on_plane_and_ray(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        plane_pt: np.ndarray,
        plane_n: np.ndarray,
    ):
        """交点应同时在平面上和射线上。

        **Validates: Requirements 5.2, 5.5**
        """
        result = ray_plane_intersect(ray_origin, ray_dir, plane_pt, plane_n)

        # 跳过无交点的情况（平行或反方向）
        assume(result is not None)

        P = result

        # (1) 交点在平面上：N·(P - P0) ≈ 0
        dist_to_plane = abs(np.dot(plane_n, P - plane_pt))
        assert dist_to_plane < 1e-6, (
            f"交点不在平面上: N·(P-P0) = {dist_to_plane}"
        )

        # (2) 交点在射线上：P = O + t*D，t > 0
        # 求 t：取分量最大的维度避免除零
        diff = P - ray_origin
        max_dim = np.argmax(np.abs(ray_dir))
        t = diff[max_dim] / ray_dir[max_dim]
        assert t >= -1e-6, f"t 应 >= 0，实际为 {t}"

        # 验证 P ≈ O + t*D
        reconstructed = ray_origin + t * ray_dir
        assert np.allclose(P, reconstructed, atol=1e-5), (
            f"交点不在射线上: P={P}, O+t*D={reconstructed}"
        )
