"""Fresh-process R6 entry checks; no real camera or OS operation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_shadow_collect_loads_native_detector_before_qt():
    if importlib.util.find_spec('mediapipe') is None:
        pytest.skip('MediaPipe runtime is not installed')
    code = r'''
import builtins, cv2, runpy, sys
cv2.VideoCapture = lambda *_: (_ for _ in ()).throw(AssertionError('camera access'))
original = builtins.__import__
def checked(name, *args, **kwargs):
    module = original(name, *args, **kwargs)
    if name == 'src.ui.r6_shadow_window':
        def show(self):
            from src.vision.face_detector import FaceDetector
            from PyQt6.QtCore import QTimer
            from PyQt6.QtWidgets import QApplication
            assert self.pipeline is None and self.recorder is None
            QTimer.singleShot(0, QApplication.instance().quit)
        module.ShadowWindow.showFullScreen = show
    return module
builtins.__import__ = checked
sys.argv = ['scripts/r6_shadow.py', 'collect', '--config', 'configs/classic.yaml']
runpy.run_path('scripts/r6_shadow.py', run_name='__main__')
'''
    proc = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                          env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen', 'PYTHONUTF8': '1'},
                          capture_output=True, timeout=30)
    assert proc.returncode == 0, (proc.stdout + proc.stderr).decode('utf-8', errors='replace')


def test_shadow_offline_commands_do_not_import_qt_or_detector(tmp_path):
    code = r'''
import builtins, runpy, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.startswith(('PyQt6', 'mediapipe', 'src.vision.face_detector')):
        raise AssertionError('offline command loaded native UI/detector: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
sys.argv = ['scripts/r6_shadow.py'] + sys.argv[1:]
runpy.run_path('scripts/r6_shadow.py', run_name='__main__')
'''
    session = str(tmp_path / 'synthetic')
    for args in (['selftest', '--output', session], ['replay', session]):
        proc = subprocess.run([sys.executable, '-c', code, *args], cwd=ROOT,
                              env={**os.environ, 'PYTHONUTF8': '1'}, capture_output=True, timeout=30)
        assert proc.returncode == 0, (proc.stdout + proc.stderr).decode('utf-8', errors='replace')
