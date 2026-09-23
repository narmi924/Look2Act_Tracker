"""Fresh-process native import regression; never opens a camera or runs detection."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_collect_native_dependencies_before_qt():
    if importlib.util.find_spec('mediapipe') is None:
        pytest.skip('MediaPipe runtime is not installed')
    code = r'''
import builtins
import runpy
import sys
import cv2

def forbidden(*args, **kwargs):
    raise AssertionError('Camera access is forbidden in startup test')
cv2.VideoCapture = forbidden
original_import = builtins.__import__

def checked_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if name == 'src.ui.experiment_window':
        def check_ready(self):
            # Exercise the detector import formerly deferred until Start, after Qt.
            from src.vision.face_detector import FaceDetector
            from PyQt6.QtCore import QTimer
            from PyQt6.QtWidgets import QApplication
            assert self.pipeline is None and self.recorder is None
            QTimer.singleShot(0, QApplication.instance().quit)
        module.ExperimentWindow.showFullScreen = check_ready
    return module
builtins.__import__ = checked_import
sys.argv = ['scripts/experiment.py', 'collect', '--config', 'configs/classic.yaml']
runpy.run_path('scripts/experiment.py', run_name='__main__')
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
        capture_output=True, timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode('utf-8', errors='replace')


def test_offline_commands_do_not_load_native_detector_or_qt(tmp_path):
    code = r'''
import builtins
import runpy
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.startswith(('mediapipe', 'src.vision.face_detector', 'PyQt6')):
        raise AssertionError('Offline command loaded native UI/detector: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.argv = ['scripts/experiment.py'] + sys.argv[1:]
runpy.run_path('scripts/experiment.py', run_name='__main__')
'''
    session = str(tmp_path / 'synthetic')
    for arguments in (['selftest', '--output', session], ['replay', session]):
        result = subprocess.run(
            [sys.executable, '-c', code, *arguments],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, timeout=30,
        )
        assert result.returncode == 0, (result.stdout + result.stderr).decode('utf-8', errors='replace')
