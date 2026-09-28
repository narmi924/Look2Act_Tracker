"""Dedicated F2 shadow window. No system cursor, click, keyboard send or launcher path."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import queue
import threading
import time
import uuid

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.experiment.protocol import Protocol, make_plan
from src.experiment.recording import Recorder, write_json
from src.experiment.r6_shadow import (CalibrationCollector, LATE_WAIT_S, ShadowConsumer,
                                      fit_personal, load_mapping, mapping_context,
                                      replay_shadow, save_mapping, shadow_summary)
from src.experiment.session import metadata, config_snapshot
from src.tracker.pipeline import TrackerPipeline


OUTPUT_ROOT = Path(__file__).resolve().parents[2] / 'experiment_sessions' / 'r6_runs'


def validation_plan(size):
    full = make_plan(size, 'AB')
    segments = []
    offset = 0.
    for item in full['segments'][9:]:
        copied = dict(item, source_plan_segment=item['segment'], segment=len(segments),
                      planned_offset_s=offset)
        segments.append(copied)
        offset += item['duration_s']
    return dict(parameters=full['parameters'], segments=segments, duration_s=offset,
                selection='A_second_round_then_B')


class ShadowWindow(QWidget):
    def __init__(self, config, *, load_mapping_path=None, clock=time.perf_counter,
                 pipeline_factory=TrackerPipeline, output_root=OUTPUT_ROOT):
        super().__init__()
        if config.normalized_backend != 'classic':
            raise ValueError('R6 requires Classic perception')
        self.config, self.load_mapping_path = config, load_mapping_path
        self.clock, self.pipeline_factory, self.output_root = clock, pipeline_factory, Path(output_root)
        self.pipeline = self.recorder = self.protocol = self.collector = self.consumer = None
        self.retiring_pipeline = None
        self.phase = 'prepare'
        self.paused_from = None
        self.target = self.candidate = None
        self.protocol_events = []
        self.producer_queue = queue.Queue(maxsize=512)
        self.pending = []
        self.drop_lock = threading.Lock()
        self.queue_drops = 0
        self.seal_cutoff = self.seal_deadline = None
        self.samples_sealed = False
        self.fit_token = 0
        self.fit_results = queue.Queue()
        self.context = None
        self.attempt_id = None
        self.summary_path = None

        self.setWindowTitle('Look2Act F2 shadow（无系统操作）')
        self.setStyleSheet('background:#20252b; color:#eeeeee; font-size:18px;')
        self.info = QLabel('仅在本地保存数值与个人映射；不保存图像、不上传。\n'
                           '候选点只在此窗口绘制，绝不移动系统光标或点击。点击开始才打开摄像头。')
        self.info.setWordWrap(True)
        self.status = QLabel('准备就绪')
        self.start_button = QPushButton('开始')
        self.pause_button = QPushButton('暂停')
        self.resume_button = QPushButton('继续')
        self.recalibrate_button = QPushButton('重新校准')
        self.end_button = QPushButton('结束 / Esc')
        for button in (self.pause_button, self.resume_button, self.recalibrate_button, self.end_button):
            button.setEnabled(False)
        row = QHBoxLayout()
        for button in (self.start_button, self.pause_button, self.resume_button,
                       self.recalibrate_button, self.end_button):
            row.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.status)
        layout.addLayout(row)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_session)
        self.pause_button.clicked.connect(self.pause)
        self.resume_button.clicked.connect(self.resume)
        self.recalibrate_button.clicked.connect(self.recalibrate)
        self.end_button.clicked.connect(self.finish)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)

    def _emit(self, kind, at, **payload):
        if self.recorder:
            self.recorder.emit(kind, at, **payload)

    def _protocol_emit(self, kind, at, **payload):
        attempt = self.attempt_id if self.phase == 'calibrating' else None
        event = dict(kind=kind, at=at, phase=self.phase, attempt_id=attempt, **payload)
        self.protocol_events.append(event)
        self._emit(kind, at, phase=self.phase, attempt_id=attempt, **payload)

    def _enqueue(self, result):
        try:
            self.producer_queue.put_nowait(result)
        except queue.Full:
            with self.drop_lock:
                self.queue_drops += 1

    def _set_controls(self):
        self.start_button.setEnabled(self.phase in ('prepare', 'failed', 'ended'))
        self.pause_button.setEnabled(self.phase in ('calibrating', 'validation', 'free'))
        self.resume_button.setEnabled(self.phase == 'paused')
        self.recalibrate_button.setEnabled(self.pipeline is not None and self.phase in
                                           ('fitting', 'validation', 'free', 'paused'))
        self.end_button.setEnabled(self.phase not in ('prepare', 'ended'))

    def start_session(self):
        if self.phase not in ('prepare', 'failed', 'ended'):
            return
        if self.retiring_pipeline is not None:
            old = self.retiring_pipeline
            if old._thread is not None and old._thread.is_alive():
                self.status.setText('旧相机线程仍在退出；重启被拒绝')
                return
            try:
                old.stop()  # R1 cleanup, only after the old worker has exited
            except Exception as exc:
                self.status.setText('旧相机资源清理失败：' + type(exc).__name__)
                return
            self.retiring_pipeline = None
        if self.pipeline is not None:
            self.finish()
        self.fit_token += 1
        self.phase = 'prepare'
        try:
            size = (self.width(), self.height())
            full = make_plan(size, 'AB')
            meta = metadata(self.config, None, size, full, self.clock())
            meta.update(session_type='f2_shadow_r6', mode='loaded_mapping_validation' if self.load_mapping_path
                        else 'new_personal_calibration', shadow_action_authority='none',
                        screen_scale=dict(device_pixel_ratio=self.screen().devicePixelRatio(),
                                          logical_dpi=self.screen().logicalDotsPerInch(),
                                          source='Qt_reported_not_physical_measurement'))
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.recorder = Recorder(self.output_root / uuid.uuid4().hex, meta)
            self.protocol_events = []
            self.pending = []
            self.producer_queue = queue.Queue(maxsize=512)
            self.queue_drops = 0
            self.pipeline = self.pipeline_factory(self.config.checkpoint_path, self.config)
            self.pipeline.numeric_observation_enabled = True
            self.pipeline.record_sink = self.recorder.emit
            self.pipeline.observation_sink = self._enqueue
            if not self.pipeline.start():
                raise RuntimeError('tracker_start_failed')
            actual = self.pipeline.actual_camera_size
            screen = self.screen()
            self.context = mapping_context(self.config, actual, size, screen.devicePixelRatio(),
                                           screen_origin=(screen.geometry().x(), screen.geometry().y()),
                                           window_origin=(self.x(), self.y()),
                                           logical_dpi=screen.logicalDotsPerInch())
            self.recorder.metadata['camera_actual'] = list(actual)
            self.recorder.metadata['config'] = config_snapshot(self.pipeline.config)
            self.recorder.metadata['mapping_context'] = self.context
            write_json(self.recorder.directory / 'session.json', self.recorder.metadata)
            self.consumer = ShadowConsumer(self.config.max_observation_age_ms, self.clock, self._emit)
            if self.load_mapping_path:
                model = load_mapping(self.load_mapping_path, self.context)
                self.consumer.activate(model)
                self._begin_validation('explicit_loaded_mapping')
            else:
                self._begin_calibration()
            self.timer.start()
            self.tick()
        except Exception as exc:
            self._fail('无法开始：' + type(exc).__name__ + ' · ' + str(exc)[:80])

    def _begin_calibration(self):
        self.fit_token += 1
        self.phase = 'calibrating'
        self.target = self.candidate = None
        self.protocol_events = []
        self.attempt_id = uuid.uuid4().hex
        self._emit('calibration_attempt', self.clock(), attempt_id=self.attempt_id)
        self.producer_queue = queue.Queue(maxsize=512)
        self.pending = []
        with self.drop_lock:
            self.queue_drops = 0
        self.pipeline.observation_sink = self._enqueue
        attempt = self.attempt_id
        self.collector = CalibrationCollector(self.pipeline.session_id,
            self.config.max_observation_age_ms,
            lambda kind, at, **payload: self._emit(kind, at, attempt_id=attempt, **payload))
        self.samples_sealed = False
        self.collector.bind_display((self.width(), self.height()))
        self.consumer.model = None  # never silently fall back to a previous mapping
        self.consumer.reset(self.pipeline.session_id, 'new_calibration', 'calibrating')
        self.protocol = Protocol(make_plan((self.width(), self.height()), 'A', {'a_rounds': 1}),
                                 self._protocol_emit, self.clock)
        self.protocol.start()
        self.target = self.protocol.tick()
        self.status.setText('校准 1/9：注视黄色目标')
        self._set_controls()
        self.update()

    def _begin_validation(self, reason):
        self.phase = 'validation'
        self.target = self.candidate = None
        self.collector = None
        self.pipeline.observation_sink = None
        omitted = len(self.pending)
        self.pending.clear()
        while True:
            try:
                self.producer_queue.get_nowait()
                omitted += 1
            except queue.Empty:
                break
        if omitted:
            self._emit('late_samples_omitted', self.clock(), count=omitted)
        self.protocol_events = []
        self.consumer.reset(self.pipeline.session_id, reason, 'validation')
        self.protocol = Protocol(validation_plan((self.width(), self.height())),
                                 self._protocol_emit, self.clock)
        self.protocol.start()  # the validation clock starts only after fitting finishes
        self.target = self.protocol.tick()
        self.status.setText('独立验证：只看黄色目标，预测点隐藏')
        self._set_controls()
        self.update()

    def _drain_samples(self):
        if self.collector is None or self.phase not in ('calibrating', 'fitting', 'paused'):
            return
        if self.samples_sealed:
            omitted = len(self.pending)
            self.pending.clear()
            while True:
                try:
                    self.producer_queue.get_nowait()
                    omitted += 1
                except queue.Empty:
                    break
            if omitted:
                self._emit('late_samples_omitted', self.clock(), count=omitted)
            return
        while True:
            try:
                self.pending.append(self.producer_queue.get_nowait())
            except queue.Empty:
                break
        remaining = []
        now = self.clock()
        guard = self.protocol.plan['parameters']['transition_guard_s']
        cutoff = (self.seal_cutoff if self.seal_cutoff is not None else
                  self.protocol.started + self.protocol.shift + self.protocol.plan['duration_s'])
        for result in self.pending:
            obs = result.observation
            if obs is not None and now < obs.timestamp + guard:
                remaining.append(result)
            else:
                self.collector.consider(result, self.protocol_events, cutoff)
        self.pending = remaining

    def _start_fit(self):
        self._drain_samples()
        if self.pending:
            self._emit('late_samples_omitted', self.clock(), count=len(self.pending))
            self.pending.clear()
        with self.drop_lock:
            dropped = self.queue_drops
        self._emit('calibration_sealed', self.clock(), cutoff=self.seal_cutoff,
                   attempt_id=self.attempt_id,
                   sample_count=len(self.collector.samples), per_target=dict(Counter(
                       s['segment'] for s in self.collector.samples)), queue_drops=dropped,
                   rejected=dict(self.collector.rejections))
        self.samples_sealed = True
        painted = {e['segment'] for e in self.protocol_events if e['kind'] == 'target_painted'}
        if painted != set(range(9)) or dropped or self.recorder.lost or self.recorder.error:
            self._fail('校准未完成：目标未全部绘制或数值队列/记录有缺口')
            return
        token = self.fit_token
        samples = tuple(self.collector.samples)
        context = dict(self.context)
        code_commit = self.recorder.metadata['code_commit']
        def worker():
            try:
                answer = fit_personal(samples, context, code_commit)
            except Exception as exc:
                answer = exc
            self.fit_results.put((token, answer))
        threading.Thread(target=worker, name='f2-personal-fit', daemon=True).start()

    def tick(self):
        if self.pipeline is None or self.phase in ('prepare', 'ended', 'failed'):
            return
        screen = self.screen()
        current_context = mapping_context(self.config, self.pipeline.actual_camera_size,
                                          (self.width(), self.height()), screen.devicePixelRatio(),
                                          screen_origin=(screen.geometry().x(), screen.geometry().y()),
                                          window_origin=(self.x(), self.y()),
                                          logical_dpi=screen.logicalDotsPerInch())
        if self.context and current_context != self.context:
            self._fail('显示尺寸或缩放发生变化；本次映射失效')
            return
        self._drain_samples()
        if self.phase == 'calibrating':
            self.target = self.protocol.tick()
            if self.protocol.finished:
                self.phase = 'fitting'
                self.target = None
                self.seal_cutoff = self.protocol.started + self.protocol.shift + self.protocol.plan['duration_s']
                self.seal_deadline = self.clock() + LATE_WAIT_S
                self.consumer.reset(self.pipeline.session_id, 'calibration_sealing', 'fitting')
                self.status.setText('封存校准样本并拟合中…')
            elif self.target:
                self.status.setText(f"校准 {self.protocol.current + 1}/9：{self.target['instruction']}")
        elif self.phase == 'fitting':
            if self.seal_deadline is not None and self.clock() >= self.seal_deadline:
                self.seal_deadline = None
                self._start_fit()
            try:
                token, answer = self.fit_results.get_nowait()
            except queue.Empty:
                pass
            else:
                if token == self.fit_token and self.phase == 'fitting':
                    if isinstance(answer, Exception):
                        self._fail('校准拟合失败：' + str(answer)[:100])
                    else:
                        try:
                            save_mapping(self.recorder.directory / 'f2_mapping.json', answer)
                            self.consumer.activate(answer, source_attempt_id=self.attempt_id)
                            self._begin_validation('new_mapping_activated')
                        except (OSError, ValueError) as exc:
                            self._fail('映射保存失败：' + type(exc).__name__)
        elif self.phase == 'validation':
            self.target = self.protocol.tick()
            if self.protocol.finished:
                self.phase = 'free'
                self.target = None
                self.consumer.reset(self.pipeline.session_id, 'validation_complete', 'free')
                self.status.setText('自由观察：候选点仅在本窗口；遮挡后应隐藏')
            elif self.target:
                self.status.setText(f"验证 {self.protocol.current + 1}/21：{self.target['instruction']}")
        if self.phase in ('validation', 'free'):
            event = self.consumer.consume(self.pipeline.get_latest_result(),
                lambda: self.pipeline.get_dispatch_snapshot(clock=self.clock))
            self.candidate = event['display_point'] if event['visible'] else None
            if self.phase == 'free' and not event['visible']:
                self.status.setText('自由观察：候选点隐藏 · ' + str(event['rejection_reason'] or '等待新观测'))
        if self.recorder and (self.recorder.lost or self.recorder.error):
            self._fail('数值记录不完整，请结束并检查本地会话')
        self._set_controls()
        self.update()

    def pause(self):
        if self.phase not in ('calibrating', 'validation', 'free'):
            return
        self.paused_from = self.phase
        if self.protocol and self.phase != 'free':
            self.protocol.pause()
        self.phase = 'paused'
        self.target = self.candidate = None
        self.consumer.reset(self.pipeline.session_id, 'pause', 'paused')
        self.status.setText('已暂停')
        self._set_controls()
        self.update()

    def resume(self):
        if self.phase != 'paused':
            return
        self.phase = self.paused_from
        if self.protocol and self.phase != 'free':
            self.protocol.resume()
        self.consumer.reset(self.pipeline.session_id, 'resume', self.phase)
        self._set_controls()
        self.tick()

    def recalibrate(self):
        if self.pipeline is None or self.phase not in ('fitting', 'validation', 'free', 'paused', 'failed'):
            return
        self.fit_token += 1
        if self.protocol and not self.protocol.finished:
            self.protocol.end('recalibration')
        self.seal_cutoff = self.seal_deadline = None
        self.pending.clear()
        while not self.producer_queue.empty():
            try:
                self.producer_queue.get_nowait()
            except queue.Empty:
                break
        self._begin_calibration()

    def _fail(self, message):
        self.fit_token += 1
        self.timer.stop()
        self.phase = 'failed'
        self.target = self.candidate = None
        self.status.setText(message)
        self._set_controls()
        self.update()
        if self.pipeline is not None:
            self.finish(complete=False, preserve_status=True)
            self.phase = 'failed'
            self._set_controls()

    def finish(self, checked=False, *, complete=None, preserve_status=False):
        prior = self.phase
        self.fit_token += 1
        self.timer.stop()
        self.target = self.candidate = None
        if self.protocol and not self.protocol.finished:
            self.protocol.end('user_end')
        if self.pipeline:
            self.pipeline.stop()
            alive = self.pipeline._thread is not None and self.pipeline._thread.is_alive()
            self.pipeline.record_sink = None
            self.pipeline.observation_sink = None
            self.pipeline.numeric_observation_enabled = False
            if alive:
                self.retiring_pipeline = self.pipeline
        else:
            alive = False
        if self.recorder and not self.recorder.closed:
            self.recorder.metadata['calibration_queue_drops'] = self.queue_drops
            self.recorder.close(complete=(prior == 'free' and not alive) if complete is None else
                                bool(complete and not alive))
            try:
                summary = shadow_summary(self.recorder.directory)
                summary['shadow_replay'] = replay_shadow(self.recorder.directory)
                write_json(self.recorder.directory / 'shadow_summary.json', summary)
                self.summary_path = self.recorder.directory / 'shadow_summary.json'
            except Exception:
                pass
        self.pipeline = None
        self.protocol = None
        self.collector = None
        self.consumer = None
        self.phase = 'ended'
        if not preserve_status:
            name = self.recorder.directory.name if self.recorder else '无会话'
            self.status.setText('已结束，记录保存在 r6_runs/' + name)
        self._set_controls()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.phase in ('calibrating', 'validation') and self.protocol:
            target = self.protocol.tick()
            if target and not self.protocol.finished:
                self.target = target
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor('#f2d35e'))
                painter.drawEllipse(QPointF(*target['instructed_target']), 10., 10.)
                painter.end()
                self.protocol.painted()  # host paint submission, not screen photon time
                return
        if self.phase == 'free' and self.consumer and self.candidate and self.pipeline:
            allowed, reason, state = self.consumer.recheck_display(
                lambda: self.pipeline.get_dispatch_snapshot(clock=self.clock))
            if allowed:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor('#65d6aa'))
                painter.drawEllipse(QPointF(*self.candidate), 8., 8.)
            else:
                self.candidate = None
            painter.end()
            stamp = None if self.consumer.last is None else self.consumer.last['observation']
            at = self.clock()
            self._emit('shadow_paint', at, visible=allowed,
                       observation_id=None if stamp is None else [stamp.session, stamp.sequence],
                       source_to_paint_s=None if stamp is None or not allowed else at - stamp.timestamp,
                       checked_state=state, rejection_reason=reason)
            return
        painter.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.finish()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event):
        self.pause()
        super().hideEvent(event)

    def closeEvent(self, event):
        self.finish()
        super().closeEvent(event)
