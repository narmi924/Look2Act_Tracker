"""Raw -> calibration -> screen smoothing -> display bounds, without Qt.

Call only for NEW observations accepted by ObservationGate. The caller resets on
interruptions/context changes and performs the producer recheck after processing.
Single-screen pixels use inclusive [0, width-1] x [0, height-1] bounds. This does
not change the backend's W/H normalization or geometric coordinate definitions.
"""
from dataclasses import replace
import math

from src.tracker.classic import ClassicScreenSmoother
from src.tracker.smoother import GazeSmoother


def finite_point(point):
    try:
        return point is not None and len(point) == 2 and all(math.isfinite(v) for v in point)
    except (TypeError, ValueError):
        return False


def in_screen(point, size):
    return finite_point(point) and all(0 <= v <= extent - 1 for v, extent in zip(point, size))


def display_point(point, size):
    if not finite_point(point):
        return None
    return tuple(float(max(0, min(v, extent - 1))) for v, extent in zip(point, size))


class ScreenMapper:
    def __init__(self, classic_smoother=None):
        self.classic = classic_smoother if classic_smoother is not None else ClassicScreenSmoother(history_len=60)
        self.ema = None
        self._configuration = None

    def reset(self):
        self.classic.reset()
        if self.ema is not None:
            self.ema.reset()

    def process(self, result, config, calibrator, size):
        """Return a detached result; source identity/time and raw input stay intact.

        Deep's actual R1 algorithm was EMA even when configured as 'kalman'. Keep
        that behavior/alpha; Classic keeps Kalman + 60-history for non-none modes.
        """
        key = (result.backend, config.normalized_smoother_type, config.smoother_alpha, id(calibrator), tuple(size))
        if key != self._configuration:
            self.reset()
            self.ema = GazeSmoother(config.smoother_alpha)
            self._configuration = key
        output = replace(result, calibrated_point=None, smoothed_point=None, display_point=None,
                         calibrated_in_bounds=None, smoothed_in_bounds=None, screen_rejection=None)

        def reject(reason):
            self.reset()
            return replace(output, screen_rejection=reason)

        if len(size) != 2 or not all(math.isfinite(v) and v >= 1 for v in size):
            return reject('invalid_screen_size')
        if not result.valid or result.point_kind != 'observed' or not finite_point(result.raw_point):
            return reject('missing_or_invalid_raw_point')
        calibrated = calibrator is not None and calibrator.is_calibrated
        if result.backend == 'classic' and not calibrated:
            return reject('classic_requires_calibration')
        try:
            point = calibrator.apply(result.raw_point) if calibrated else result.raw_point
            output.calibrated_point = point
            if not finite_point(point):
                return reject('nonfinite_calibrated_point')
            output.calibrated_in_bounds = in_screen(point, size)
            if not output.calibrated_in_bounds:
                # Do not feed an already rejected value into any filter.
                output.display_point = display_point(point, size)
                return reject('calibrated_out_of_bounds')
            if config.normalized_smoother_type == 'none':
                smooth = point
            else:
                smoother = self.classic if result.backend == 'classic' else self.ema
                smooth = smoother.update(point)
            output.smoothed_point = smooth
            if not finite_point(smooth):
                return reject('nonfinite_smoothed_point')
            output.smoothed_in_bounds = in_screen(smooth, size)
            output.display_point = display_point(smooth, size)
            if not output.smoothed_in_bounds:
                return reject('smoothed_out_of_bounds')
        except Exception as exc:
            return reject(f'screen_processing_failed: {exc}')
        return output
