"""Same-frame eye crops and full landmark arrays for an R8 session. Local files only, never JSON events.

The pipeline calls `FrameWriter.submit` from its capture thread with the frame it just processed.
Cropping happens there (two small copies); PNG encoding and file writes happen on this writer's
own thread through a bounded queue. Overflow drops the frame and counts it; it never blocks capture.
"""
import json
from pathlib import Path
import queue
import threading

import cv2
import numpy as np

EYE_CORNERS = {'left': (33, 133), 'right': (362, 263)}
CROP_WIDTH_FACTOR, CROP_HEIGHT_FACTOR = 2.2, 1.2  # multiples of the eye-corner width
LANDMARK_DTYPE = np.float32


def eye_boxes(landmarks, frame_size):
    """Integer crop boxes [x, y, w, h] per eye around the corner midpoint, clipped to the frame."""
    width, height = frame_size
    boxes = {}
    for side, (a, b) in EYE_CORNERS.items():
        if landmarks is None or len(landmarks) <= max(a, b):
            return None
        pa, pb = landmarks[a][:2], landmarks[b][:2]
        mid = (pa + pb) / 2
        eye_width = float(np.linalg.norm(pb - pa))
        if not np.isfinite(eye_width) or eye_width < 2:
            return None
        half_w, half_h = CROP_WIDTH_FACTOR * eye_width / 2, CROP_HEIGHT_FACTOR * eye_width / 2
        x0, y0 = int(max(0, np.floor(mid[0] - half_w))), int(max(0, np.floor(mid[1] - half_h)))
        x1, y1 = int(min(width, np.ceil(mid[0] + half_w))), int(min(height, np.ceil(mid[1] + half_h)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        boxes[side] = [x0, y0, x1 - x0, y1 - y0]
    return boxes


class FrameWriter:
    def __init__(self, directory, capacity=64):
        self.directory = Path(directory)
        self.frames_dir = self.directory / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.closed = False
        self.submitted = self.dropped = self.skipped_no_landmarks = self.written = self.landmark_records = 0
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name='r8-frame-writer', daemon=True)
        self.thread.start()

    def submit(self, result, frame, landmarks):
        """Capture-thread entry: crop now, enqueue small arrays, never block."""
        observation = getattr(result, 'observation', None)
        if observation is None:
            return False
        with self.lock:
            if self.closed:
                return False
            self.submitted += 1
            if landmarks is None:
                self.skipped_no_landmarks += 1
                return False
        boxes = eye_boxes(np.asarray(landmarks), (frame.shape[1], frame.shape[0]))
        if boxes is None:
            with self.lock:
                self.skipped_no_landmarks += 1
            return False
        crops = {side: frame[y:y + h, x:x + w].copy() for side, (x, y, w, h) in boxes.items()}
        item = dict(observation=[observation.session, observation.sequence], source_time=observation.timestamp,
                    published_at=getattr(result, 'published_at', None), valid=bool(getattr(result, 'valid', False)),
                    frame_size=[int(frame.shape[1]), int(frame.shape[0])], boxes=boxes, crops=crops,
                    landmarks=np.ascontiguousarray(np.asarray(landmarks, dtype=LANDMARK_DTYPE)))
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            with self.lock:
                self.dropped += 1
            return False

    def _run(self):
        try:
            with (self.directory / 'frames.jsonl').open('w', encoding='utf-8') as index, \
                    (self.directory / 'landmarks.f32').open('ab') as landmarks:
                while not self.done.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    files = {}
                    for side, crop in item['crops'].items():
                        name = f"{item['observation'][1]:07d}_{side[0].upper()}.png"
                        ok, encoded = cv2.imencode('.png', crop)
                        if not ok:
                            raise RuntimeError('png_encode_failed')
                        (self.frames_dir / name).write_bytes(encoded.tobytes())
                        files[side] = 'frames/' + name
                    landmarks.write(item['landmarks'].tobytes())
                    record = dict(observation=item['observation'], source_time=item['source_time'],
                                  published_at=item['published_at'], valid=item['valid'], frame_size=item['frame_size'],
                                  boxes=item['boxes'], files=files, landmark_record=self.landmark_records,
                                  landmark_shape=list(item['landmarks'].shape))
                    self.landmark_records += 1
                    index.write(json.dumps(record) + '\n')
                    index.flush()
                    self.written += 1
        except Exception as exc:
            self.error = type(exc).__name__  # no personal paths in metadata

    def close(self):
        with self.lock:
            if self.closed:
                return self.stats()
            self.closed = True
        self.done.set()
        self.thread.join(timeout=10.)
        stats = self.stats()
        (self.directory / 'landmarks_meta.json').write_text(json.dumps(dict(
            dtype='float32', record_shape=[478, 3], order='x_px,y_px,z_times_frame_width',
            records=self.landmark_records, source='MediaPipe FaceMesh refine_landmarks landmarks, same frame as events'),
            indent=2), encoding='utf-8')
        return stats

    def stats(self):
        return dict(submitted=self.submitted, written=self.written, dropped=self.dropped,
                    skipped_no_landmarks=self.skipped_no_landmarks, landmark_records=self.landmark_records,
                    writer_error=self.error, writer_still_alive=self.thread.is_alive(),
                    crop_box='corner midpoint, %.1fw x %.1fw, clipped' % (CROP_WIDTH_FACTOR, CROP_HEIGHT_FACTOR),
                    image_format='png_bgr_native_resolution')


def read_frames_index(directory):
    path = Path(directory) / 'frames.jsonl'
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_landmarks(directory):
    """All landmark records as (n, 478, 3) float32; empty array when the file is missing."""
    path = Path(directory) / 'landmarks.f32'
    if not path.is_file():
        return np.zeros((0, 478, 3), dtype=LANDMARK_DTYPE)
    data = np.fromfile(path, dtype=LANDMARK_DTYPE)
    if data.size % (478 * 3):
        raise ValueError('landmarks.f32 is not a whole number of 478x3 records')
    return data.reshape(-1, 478, 3)
