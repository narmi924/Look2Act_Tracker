"""R8 recording window: R3 flow plus moving stimuli, same-frame eye crops and full landmarks.

Still no gaze-driven action and no prediction feedback. Images and landmark arrays stay in the
session directory under the ignored experiment_sessions/; nothing is uploaded.
"""
import time

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen

from src.experiment.frames import FrameWriter
from src.experiment.r8_protocol import stimulus_at
from src.experiment.recording import write_json
from src.tracker.pipeline import TrackerPipeline
from src.ui.experiment_window import ExperimentWindow

SESSION_TYPE = 'r8_pursuit_v1'


class R8RecordWindow(ExperimentWindow):
    def __init__(self, config, parameters=None, clock=time.perf_counter, pipeline_factory=TrackerPipeline):
        def factory(checkpoint, cfg):
            pipeline = pipeline_factory(checkpoint, cfg)
            pipeline.collect_full_landmarks = True  # detector keeps all 478 points for this session only
            return pipeline
        super().__init__(config, 'R8', parameters, None, clock, factory)
        self.frames = None
        self.setWindowTitle('Look2Act R8 录制')
        self.info.setText('本轮会额外保存每帧的双眼裁剪图和全部面部关键点，只写入本机 experiment_sessions/，不上传。\n'
                          '流程约 3 分钟：9 点注视 → 跟随移动点 → 跟随带圈的小点 → 短头动 → 再次 9 点。\n'
                          '点击开始才开启摄像头与保存。')

    def start_recording(self):
        super().start_recording()
        if self.recorder is None or self.pipeline is None or self.recorder.closed:
            return
        try:
            self.frames = FrameWriter(self.recorder.directory)
        except OSError as exc:
            self.status.setText('图像写入器创建失败：' + type(exc).__name__)
            self.finish(complete=False)
            return
        self.pipeline.frame_sink = self.frames.submit
        self.recorder.metadata.update(
            session_type=SESSION_TYPE,
            images=dict(eye_crops='frames/<sequence>_L.png, _R.png; index frames.jsonl', format='png_bgr_native_resolution',
                        crop_box='eye-corner midpoint, 2.2w x 1.2w, clipped to frame', attached='after_pipeline_start'),
            full_landmarks=dict(file='landmarks.f32', dtype='float32', record_shape=[478, 3],
                                order='x_px,y_px,z_times_frame_width', index='frames.jsonl landmark_record'))
        write_json(self.recorder.directory / 'session.json', self.recorder.metadata)
        self.info.setText('本地录制中（数值 + 眼部裁剪图 + 全部关键点）· 可随时暂停、跳过或结束')

    def finish(self, checked=False, *, complete=True):
        if self.pipeline is not None:
            self.pipeline.frame_sink = None
        if self.frames is not None and self.recorder is not None and not self.recorder.closed:
            stats = self.frames.close()
            self.recorder.metadata['frames'] = stats
            if stats['writer_error'] or stats['writer_still_alive']:
                complete = False
        super().finish(checked, complete=complete)

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            if not (self.protocol and self.target and self.protocol.paused_at is None and not self.protocol.finished):
                return
            target = self.protocol.tick()  # re-evaluate deadlines at paint time, as the R3 window does
            if not target:
                return
            self.target = target
            phase = self.protocol.phase()
            if phase is None:
                return
            phase = max(0., phase)
            size = (self.width(), self.height())
            stimulus = stimulus_at(target, phase, size)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QColor('#d8dde2'))
            painter.setFont(QFont('Microsoft YaHei', 15))
            remaining = max(0., target['duration_s'] - phase)
            painter.drawText(QRectF(0, 56, size[0], 36), Qt.AlignmentFlag.AlignCenter,
                             f"{target['instruction']}   ·   剩余 {remaining:.0f} s   ·   第 {target['segment'] + 1}/{len(self.protocol.plan['segments'])} 段")
            painter.setPen(Qt.PenStyle.NoPen)
            if 'dot' in stimulus:
                x, y = stimulus['dot']
                painter.setBrush(QColor('#f2d35e'))
                painter.drawEllipse(QPointF(x, y), 10., 10.)
                self.protocol.painted()
                if target['stimulus'] == 'path':
                    self.protocol.moved('target_moved', phase_s=phase, x=float(x), y=float(y))
                return
            anchors = target['choice']['anchors']
            for index, (x, y) in enumerate(stimulus['markers']):
                ax, ay = anchors[index]
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 255, 255, 36))
                painter.drawRect(QRectF(ax - 14., ay - 14., 28., 28.))
                if index == stimulus['instructed']:
                    painter.setBrush(QColor('#f2d35e'))
                    painter.drawEllipse(QPointF(x, y), 11., 11.)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor('#f2d35e'), 3.))
                    painter.drawEllipse(QPointF(x, y), 20., 20.)
                else:
                    painter.setBrush(QColor('#8d969f'))
                    painter.drawEllipse(QPointF(x, y), 9., 9.)
            self.protocol.painted()
            self.protocol.moved('markers_moved', phase_s=phase, instructed=int(stimulus['instructed']),
                                positions=[[float(x), float(y)] for x, y in stimulus['markers']])
        finally:
            painter.end()
