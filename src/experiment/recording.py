"""Versioned, bounded asynchronous JSONL recorder with explicit incomplete prefixes."""
import copy
import json
import math
import queue
import threading
from pathlib import Path

SCHEMA_VERSION = 1


def encode(value):
    # Deliberately refuse ndarray/images. Numeric matrices must be explicitly selected.
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'nonfinite': str(value)}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    raise TypeError('unsupported trace value: ' + type(value).__name__)


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'nonfinite'}:
            return float(value['nonfinite'])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(encode(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class Recorder:
    def __init__(self, directory, metadata, capacity=512, opener=None):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = copy.deepcopy(metadata)
        self.metadata.update(schema_version=SCHEMA_VERSION, complete=False, write_lost=0)
        write_json(self.directory / 'session.json', self.metadata)
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.closed = False
        self.done = threading.Event()
        self.sequence = 0
        self.lost = 0
        self.first_lost = self.last_lost = None
        self.error = None
        self.written = 0
        self.opener = opener or (lambda path: path.open('w', encoding='utf-8'))
        self.thread = threading.Thread(target=self._write, name='experiment-writer', daemon=True)
        self.thread.start()

    def emit(self, kind, at, **payload):
        # Short recorder lock only. JSON and all file operations are in _write.
        event = copy.deepcopy(dict(kind=kind, at=at, **payload))
        with self.lock:
            if self.closed:
                return False
            self.sequence += 1
            event['event_id'] = self.sequence
            try:
                if self.error:
                    raise queue.Full
                self.queue.put_nowait(event)
                return True
            except queue.Full:
                self.lost += 1
                self.first_lost = self.first_lost or self.sequence
                self.last_lost = self.sequence
                return False

    def _write(self):
        try:
            with self.opener(self.directory / 'events.jsonl') as stream:
                while not self.done.is_set() or not self.queue.empty():
                    try:
                        event = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    stream.write(json.dumps(encode(event), ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()  # process-crash prefix; not a power-loss durability promise
                    self.written += 1
        except Exception as exc:
            self.error = type(exc).__name__  # do not persist personal filesystem paths

    def close(self, complete=True):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.done.set()
        self.thread.join(timeout=5.)
        unconfirmed = max(0, self.sequence - self.written - self.lost)
        self.metadata.update(complete=bool(complete and not self.lost and not self.error and not self.thread.is_alive()),
                             write_lost=self.lost, lost_event_range=[self.first_lost, self.last_lost],
                             write_unconfirmed=unconfirmed,
                             writer_error=self.error, writer_still_alive=self.thread.is_alive(),
                             submitted=self.sequence, written=self.written)
        try:
            write_json(self.directory / 'session.json', self.metadata)
        except OSError:
            self.error = 'metadata_write_failed'  # initial incomplete marker remains


def read_session(directory):
    directory = Path(directory)
    metadata = decode(json.loads((directory / 'session.json').read_text(encoding='utf-8')))
    if metadata.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported schema_version')
    events, issues = [], []
    expected = 1
    path = directory / 'events.jsonl'
    if not path.exists():
        return metadata, events, ['missing_events']
    with path.open('rb') as stream:
        for line in stream:
            try:
                event = decode(json.loads(line))
                event_id = event['event_id']
                if (type(event_id) is not int or event_id < 1 or not isinstance(event['kind'], str)
                        or not isinstance(event['at'], (int, float)) or not math.isfinite(event['at'])):
                    raise ValueError('invalid event envelope')
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                issues.append('corrupt_tail_or_record')
                break  # only the complete prefix is usable
            if event_id != expected:
                issues.append('event_gap')
                events.append(dict(kind='gap', at=event['at']))
            events.append(event)
            expected = event_id + 1
    if not metadata.get('complete'):
        issues.append('incomplete_session')
    if metadata.get('write_lost'):
        issues.append('write_lost')
    if metadata.get('written') is not None and metadata['written'] != sum('event_id' in e for e in events):
        issues.append('record_count_mismatch')  # also detect a tail cut on a complete-line boundary
    return metadata, events, issues
