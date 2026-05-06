import numpy as np

from src.vision import face_detector as fd


class _Landmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def test_runtime_deep_eye_crop_matches_offline_padding_contract(monkeypatch):
    detector = fd.FaceDetector.__new__(fd.FaceDetector)
    detector.eye_crop_size = 128
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    landmarks = [_Landmark(0.0, 0.0) for _ in range(500)]

    coords = [
        (100, 100),
        (120, 100),
        (120, 110),
        (100, 110),
    ]
    for idx, (x, y) in zip(fd._LEFT_EYE_INDICES, coords * 4):
        landmarks[idx] = _Landmark(x / 320, y / 240)

    captured = {}

    def fake_resize(crop, size, interpolation=None):
        captured["shape"] = crop.shape
        captured["size"] = size
        captured["interpolation"] = interpolation
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(fd.cv2, "resize", fake_resize)

    out = detector._crop_eye(frame, landmarks, fd._LEFT_EYE_INDICES, 320, 240)

    assert out.shape == (128, 128, 3)
    assert captured["shape"][:2] == (40, 40)
    assert captured["size"] == (128, 128)
    assert captured["interpolation"] == fd.cv2.INTER_AREA
