"""A small opt-in experiment window. No gaze-controlled actions or prediction feedback."""
from pathlib import Path
import time
import uuid

from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import load_calibration
from src.experiment.consumer import Consumer
from src.experiment.protocol import make_plan, Protocol
from src.experiment.recording import Recorder, write_json
from src.experiment.session import metadata, calibration_from_snapshot, config_snapshot, replay
from src.tracker.pipeline import TrackerPipeline

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / 'experiment_sessions'


class ExperimentWindow(QWidget):
    def __init__(self, config, selection='AB', parameters=None, calibration_path=None,
                 clock=time.perf_counter, pipeline_factory=TrackerPipeline):
        super().__init__()
        self.config, self.selection, self.parameters = config, selection, parameters
        self.calibration_path, self.clock, self.pipeline_factory = calibration_path, clock, pipeline_factory
        self.pipeline = self.recorder = self.consumer = self.protocol = None
        self.target = None
        self.setWindowTitle('Look2Act 数值实验')
        self.setStyleSheet('background:#20252b; color:#eeeeee; font-size:18px;')
        self.info = QLabel('仅保存本地眼部/头部数值与校准快照；不保存图像，不上传。\n'
                           '数值数据也可能敏感。点击开始才开启摄像头与保存。\n保存目录：experiment_sessions/')
        self.info.setWordWrap(True)
        self.status = QLabel('准备就绪')
        self.start_button = QPushButton('开始')
        self.pause_button = QPushButton('暂停')
        self.resume_button = QPushButton('继续')
        self.skip_button = QPushButton('跳过')
        self.end_button = QPushButton('结束')
        for button in (self.pause_button, self.resume_button, self.skip_button, self.end_button):
            button.setEnabled(False)
        row = QHBoxLayout()
        for button in (self.start_button, self.pause_button, self.resume_button, self.skip_button, self.end_button):
            row.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.status)
        layout.addLayout(row)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.pause)
        self.resume_button.clicked.connect(self.resume)
        self.skip_button.clicked.connect(self.skip)
        self.end_button.clicked.connect(self.finish)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)

    def start_recording(self):
        if self.recorder is not None:
            return
        try:
            size = (self.width(), self.height())
            cal = None
            if self.calibration_path:
                cal = CalibrationModule()
                load_calibration(cal, str(self.calibration_path))
            plan = make_plan(size, self.selection, self.parameters)
            meta = metadata(self.config, cal, size, plan, self.clock())
            screen = self.screen()
            meta['screen_scale'] = dict(device_pixel_ratio=screen.devicePixelRatio(),
                                        logical_dpi=screen.logicalDotsPerInch(),
                                        physical_dpi_reported=screen.physicalDotsPerInch(),
                                        source='Qt_reported_not_independently_measured')
            meta['window_size'] = list(size)
            meta['screen_geometry'] = [screen.geometry().x(), screen.geometry().y(),
                                       screen.geometry().width(), screen.geometry().height()]
            cal = calibration_from_snapshot(meta['calibration'])  # freeze inputs, never reload live file
            self.recorder = Recorder(OUTPUT_ROOT / uuid.uuid4().hex, meta)
            self.start_button.setEnabled(False)
            self.pipeline = self.pipeline_factory(self.config.checkpoint_path, self.config)
            self.pipeline.record_sink = self.recorder.emit
            if not self.pipeline.start():
                raise RuntimeError('tracker_start_failed')
            actual = getattr(self.pipeline, 'actual_camera_size', None)
            self.recorder.metadata['camera_actual'] = actual
            self.recorder.metadata['config'] = config_snapshot(self.pipeline.config)
            self.recorder.metadata['runtime_backend'] = 'deep_pog' if self.pipeline.model_version == 'pog_v1' else self.config.normalized_backend
            estimator = self.pipeline.head_pose_estimator
            if estimator is not None:
                self.recorder.metadata['pnp']['camera_matrix'] = estimator.camera_matrix.tolist()
                self.recorder.metadata['pnp']['dist_coeffs'] = estimator.dist_coeffs.tolist()
            self.recorder.metadata['model']['runtime_version'] = self.pipeline.model_version
            geometry = self.pipeline.screen_geometry
            if geometry is not None:
                self.recorder.metadata['runtime_geometry'] = dict(
                    screen_size=[geometry.screen_w_px, geometry.screen_h_px],
                    **{name: getattr(geometry, '_' + name).tolist()
                       for name in ('plane_origin', 'plane_normal', 'screen_x_axis', 'screen_y_axis')})
            write_json(self.recorder.directory / 'session.json', self.recorder.metadata)
            self.consumer = Consumer(self.config, cal, size, self.clock, self.recorder.emit)
            self.consumer.reset(self.pipeline.session_id, 'experiment_start')
            self.protocol = Protocol(plan, self.recorder.emit, self.clock)
            self.protocol.start()
            self.start_button.setEnabled(False)
            self.info.setText('本地数值录制中 · 可随时暂停、跳过或结束')
            for button in (self.pause_button, self.skip_button, self.end_button):
                button.setEnabled(True)
            self.timer.start()
            self.tick()
        except Exception as exc:
            self.finish(complete=False)
            self.status.setText('无法开始：' + type(exc).__name__)

    def tick(self):
        if self.protocol is None:
            return
        self.target = self.protocol.tick()
        if self.protocol.finished:
            self.finish()
            return
        if self.target:
            self.status.setText(f"{self.target['protocol']} · {self.target['instruction']}")
        event = self.consumer.consume(self.pipeline.get_latest_result(),
                                      lambda: self.pipeline.get_dispatch_snapshot(clock=self.clock))
        if self.recorder.lost or self.recorder.error:
            self.info.setText('记录不完整：写入丢失或失败，请结束并检查文件。')
        self.update()

    def pause(self):
        if self.protocol and not self.protocol.finished:
            self.protocol.pause()
            self.consumer.reset(self.pipeline.session_id, 'pause', active=False)
            self.target = None
            self.status.setText('已暂停')
            self.pause_button.setEnabled(False)
            self.skip_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.update()

    def resume(self):
        if self.protocol and self.protocol.paused_at is not None:
            self.protocol.resume()
            self.consumer.reset(self.pipeline.session_id, 'resume')
            self.pause_button.setEnabled(True)
            self.skip_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.tick()

    def skip(self):
        if self.protocol and not self.protocol.finished:
            self.protocol.skip()
            self.consumer.reset(self.pipeline.session_id, 'skip', active=self.protocol.paused_at is None)
            self.tick()

    def finish(self, checked=False, *, complete=True):
        self.timer.stop()
        for button in (self.pause_button, self.resume_button, self.skip_button, self.end_button):
            button.setEnabled(False)
        self.target = None
        if self.protocol:
            self.protocol.end()
        if self.pipeline:
            self.pipeline.stop()
            if self.pipeline._thread is not None and self.pipeline._thread.is_alive():
                complete = False
            self.pipeline.record_sink = None
        if self.recorder and not self.recorder.closed:
            self.recorder.close(complete)
            if not self.recorder.thread.is_alive():
                try:
                    summary, _ = replay(self.recorder.directory)
                    self.status.setText(f"已保存 {self.recorder.directory.name} · 重算差异 {len(summary['replay']['mismatches'])}")
                except Exception:
                    self.status.setText('已保留记录；摘要失败，请用离线命令检查。')
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.protocol and self.target and self.protocol.paused_at is None and not self.protocol.finished:
            # Re-evaluate deadlines at paint, so a delayed repaint cannot claim an expired target.
            target = self.protocol.tick()
            if target:
                self.target = target
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor('#f2d35e'))
                painter.drawEllipse(QPointF(*target['instructed_target']), 10., 10.)
                painter.end()
                self.protocol.painted()  # host paint submission, NOT physical photon time

    def closeEvent(self, event):
        self.finish()
        super().closeEvent(event)

    def hideEvent(self, event):
        # A known hidden window must not keep assigning active task labels.
        self.pause()
        super().hideEvent(event)
