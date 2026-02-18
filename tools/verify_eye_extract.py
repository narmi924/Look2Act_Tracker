"""
批量验证 extract_eyes_from_mosaic 在所有 Session 上的表现。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
from data.preprocessing import extract_eyes_from_mosaic

raw_dir = Path(__file__).resolve().parent.parent / "dataset_raw"
total, success, fail = 0, 0, 0

for sd in sorted(raw_dir.iterdir()):
    fp = sd / "frames_private"
    if not fp.exists():
        continue
    imgs = sorted(fp.iterdir())
    # 每个 session 测试前 3 张
    for img_path in imgs[:3]:
        if img_path.suffix.lower() not in (".jpg", ".png"):
            continue
        total += 1
        img = cv2.imread(str(img_path))
        le, re = extract_eyes_from_mosaic(img)
        if le is not None and re is not None:
            assert le.shape == (128, 128, 3)
            assert re.shape == (128, 128, 3)
            success += 1
        else:
            fail += 1
            print(f"  FAIL: {img_path.relative_to(raw_dir)}, shape={img.shape}")

print(f"\nTotal: {total}, Success: {success}, Fail(no eyes): {fail}")
