from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_DLL_HANDLES = []


def _add_dll_directory(path: Path) -> None:
    if not path.exists():
        return
    try:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"


base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
mediapipe_python_dir = base_dir / "mediapipe" / "python"
_add_dll_directory(mediapipe_python_dir)

opencv_world = mediapipe_python_dir / "opencv_world3410.dll"
if opencv_world.exists():
    try:
        _DLL_HANDLES.append(ctypes.WinDLL(str(opencv_world)))
    except OSError:
        pass
