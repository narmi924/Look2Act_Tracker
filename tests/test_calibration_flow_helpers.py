from src.ui.calibration_page import (
    CalibrationPage,
    calibration_path_for_backend,
    min_samples_per_calibration_point,
    min_valid_points_for_calibration,
)


class FakeControl:
    def __init__(self):
        self._text = ""
        self._enabled = True
        self._hidden = False
        self._style = ""

    def setText(self, value):
        self._text = value

    def text(self):
        return self._text

    def setEnabled(self, value):
        self._enabled = value

    def isEnabled(self):
        return self._enabled

    def hide(self):
        self._hidden = True

    def show(self):
        self._hidden = False

    def isHidden(self):
        return self._hidden

    def setStyleSheet(self, value):
        self._style = value


class FakeCalibrator:
    @property
    def is_calibrated(self):
        return True


def test_calibration_path_is_backend_specific():
    assert calibration_path_for_backend("classic").name == "calibration_classic.json"
    assert calibration_path_for_backend("deep").name == "calibration_deep.json"
    assert calibration_path_for_backend("unknown").name == "calibration_classic.json"


def test_min_valid_points_for_calibration():
    assert min_valid_points_for_calibration(25, "polynomial") == 12
    assert min_valid_points_for_calibration(9, "polynomial") == 6
    assert min_valid_points_for_calibration(9, "affine") == 3


def test_min_samples_per_calibration_point():
    assert min_samples_per_calibration_point(45) == 15
    assert min_samples_per_calibration_point(12) == 6


def test_calibration_finished_stays_on_result_actions():
    page = CalibrationPage.__new__(CalibrationPage)
    page.status_label = FakeControl()
    page.residual_label = FakeControl()
    page.start_btn = FakeControl()
    page.save_btn = FakeControl()
    page.load_btn = FakeControl()
    page.home_btn = FakeControl()
    page.calibrator = FakeCalibrator()

    page._on_calibration_finished(True, 32.5)

    assert page.save_btn.text().startswith("保存并进入验证")
    assert page.start_btn.text().startswith("重新校准")
    assert page.load_btn.isHidden()
    assert not page.home_btn.isHidden()
