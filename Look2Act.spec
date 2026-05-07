# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs


block_cipher = None
PROJECT_ROOT = Path(SPECPATH).resolve()
APP_ICON = PROJECT_ROOT / "Look2Act.ico"


def existing(path: Path, dest: str):
    return (str(path), dest) if path.exists() else None


project_datas = [
    existing(PROJECT_ROOT / "configs", "configs"),
    existing(PROJECT_ROOT / "README.md", "."),
    existing(PROJECT_ROOT / "readme-images", "readme-images"),
    existing(APP_ICON, "."),
]

checkpoint_datas = []
for name in ("gaze_net.onnx", "gaze_pog_zero.onnx", "gaze_pog.onnx"):
    item = existing(PROJECT_ROOT / "checkpoints" / name, "checkpoints")
    if item:
        checkpoint_datas.append(item)

mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
mp_datas = [
    item for item in mp_datas
    if not (
        Path(item[0]).name == "__init__.py"
        and item[1].replace("\\", "/") in {"mediapipe", "mediapipe/python/solutions"}
    )
]
qfw_binaries = collect_dynamic_libs("qfluentwidgets")
ort_binaries = collect_dynamic_libs("onnxruntime")

cv2_binaries = collect_dynamic_libs("cv2")

hiddenimports = [
    "cv2",
    "numpy",
    "yaml",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",
    "qframelesswindow",
    "mediapipe.python.solutions.face_mesh",
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    *mp_hidden,
]

datas = [
    *(item for item in project_datas if item),
    *checkpoint_datas,
    *mp_datas,
]

binaries = [
    *mp_binaries,
    *qfw_binaries,
    *ort_binaries,
    *cv2_binaries,
]

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "torio",
    "torchgen",
    "functorch",
    "intel_extension_for_pytorch",
    "jax",
    "jaxlib",
    "ml_dtypes",
    "onnx",
    "onnxruntime.backend",
    "onnxruntime.datasets",
    "onnxruntime.quantization",
    "onnxruntime.tools",
    "onnxruntime.transformers",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends",
    "sklearn",
    "dlib",
    "imutils",
    "pytest",
    "hypothesis",
    "IPython",
    "jupyter",
    "notebook",
    "test",
    "tests",
    "unittest",
    "pydoc",
    "doctest",
    "pdb",
    "cProfile",
    "profile",
    "tkinter",
    "PyQt6.QtPdf",
    "PyQt6.QtWebEngine",
    "PyQt6.QtMultimedia",
]

a = Analysis(
    ["main.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "tools" / "pyinstaller_rth_mediapipe.py")],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.pure = [
    item for item in a.pure
    if item[0] not in {"mediapipe", "mediapipe.python.solutions"}
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Look2Act",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(APP_ICON) if APP_ICON.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Look2Act",
)
