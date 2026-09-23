"""Bounded read-only audit of legacy labels and eye mosaics; no reprocessing."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from src.experiment.r4_analysis import hash_file
from src.experiment.recording import read_session


def old_source_key(processed_row):
    """Processed filename encodes original session/frame, not an independent sample."""
    session = processed_row.get('session_id', '')
    name = Path(processed_row.get('eye_img_path', '')).name
    match = re.fullmatch(re.escape(session) + r'_(\d{6})_L\.jpg', name)
    return (session, int(match.group(1))) if match else None


def _csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _spread(items, limit):
    if type(limit) is not int or not 0 <= limit <= 32:
        raise ValueError('image_limit must be an integer between 0 and 32 pairs per group')
    if not items or limit == 0:
        return []
    if limit == 1:
        return items[:1]
    if len(items) <= limit:
        return items
    return [items[int(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def audit_legacy(raw_dir, processed_dir, *, image_limit=32):
    _spread([], image_limit)  # validate even when a source has no images
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    raw_files = sorted(raw_dir.rglob('labels.csv'))
    processed_files = sorted(processed_dir.rglob('labels.csv'))
    raw, processed = [], []
    for file in raw_files:
        for row in _csv(file):
            raw.append((file.parent, row))
    for file in processed_files:
        for row in _csv(file):
            processed.append((file.parent.name, file.parent, row))
    raw_keys = [(r['session_id'], int(r['frame_idx'])) for _, r in raw]
    processed_keys = [old_source_key(r) for _, _, r in processed]
    source = dict(zip(raw_keys, (r for _, r in raw)))
    split_keys = defaultdict(set)
    split_sessions = defaultdict(set)
    split_users = defaultdict(set)
    missing_processed_images = 0
    norm_mismatch = 0
    user_mismatch = 0
    pose_copy_mismatch = 0
    for split, directory, row in processed:
        key = old_source_key(row)
        split_keys[split].add(key)
        split_sessions[split].add(row['session_id'])
        split_users[split].add(row['user_id'])
        missing_processed_images += sum(not (directory / row[column]).is_file()
                                        for column in ('eye_img_path', 'right_eye_img_path'))
        parent = source.get(key)
        if parent is not None and (abs(float(row['norm_target_x']) - float(parent['target_x']) / float(parent['screen_w'])) > 1e-12
                                   or abs(float(row['norm_target_y']) - float(parent['target_y']) / float(parent['screen_h'])) > 1e-12):
            norm_mismatch += 1
        if parent is not None:
            user_mismatch += row['user_id'] != parent['user_id']
            pose_copy_mismatch += any(abs(float(row[axis]) - float(parent[axis])) > 1e-12
                                      for axis in ('head_yaw', 'head_pitch', 'head_roll')
                                      if axis in row and axis in parent)
    paired_splits = [(a, b) for a in split_keys for b in split_keys if a < b]
    time_anomalies = 0
    by_session = defaultdict(list)
    for _, row in raw:
        by_session[row['session_id']].append(row)
    for rows in by_session.values():
        rows.sort(key=lambda r: int(r['frame_idx']))
        time_anomalies += sum(float(b['timestamp_ms']) <= float(a['timestamp_ms'])
                              for a, b in zip(rows, rows[1:]))
    target_elapsed_invalid = sum(not math.isfinite(float(r['target_elapsed_ms'])) or float(r['target_elapsed_ms']) < 0
                                 for _, r in raw if 'target_elapsed_ms' in r)
    raw_images = [(directory / r['img_path'], r) for directory, r in raw if (directory / r['img_path']).is_file()]
    # One fixed first valid sample per source session covers local raw layouts, capped at 32.
    one_per_session = {}
    for path, row in raw_images:
        one_per_session.setdefault(row['session_id'], path)
    sampled_raw = _spread(sorted(one_per_session.values()), image_limit) if image_limit else []
    raw_layout = Counter()
    right_blank = 0
    for path in sampled_raw:
        image = cv2.imread(str(path))
        if image is None:
            raw_layout['decode_failed'] += 1
            continue
        raw_layout[str(tuple(image.shape))] += 1
        if image.shape[1] >= 256:
            right_blank += bool(np.mean(image[:128, 128:256]) < 1)
    sampled_processed = _spread(processed, image_limit) if image_limit else []
    processed_layout = Counter()
    near_flip = 0
    for _, directory, row in sampled_processed:
        left = cv2.imread(str(directory / row['eye_img_path']))
        right = cv2.imread(str(directory / row['right_eye_img_path']))
        if left is None or right is None:
            processed_layout['decode_failed'] += 1
            continue
        processed_layout[str((tuple(left.shape), tuple(right.shape)))] += 1
        if left.shape == right.shape:
            near_flip += bool(np.mean((cv2.flip(left, 1).astype(float) - right.astype(float)) ** 2) < 10)
    raw_valid_missing = sum(r['valid'] == '1' and not (directory / r['img_path']).is_file()
                            for directory, r in raw)
    raw_invalid_missing = sum(r['valid'] != '1' and not (directory / r['img_path']).is_file()
                              for directory, r in raw)
    raw_file_types = dict(Counter(p.suffix.lower() for p in raw_dir.rglob('*') if p.is_file()))
    processed_file_types = dict(Counter(p.suffix.lower() for p in processed_dir.rglob('*') if p.is_file()))
    split_info_path = processed_dir / 'split_info.json'
    if split_info_path.exists():
        split_info = json.loads(split_info_path.read_text(encoding='utf-8'))
        split_info_matches = all(set(split_info.get(split + '_sessions', [])) == split_sessions[split]
                                 for split in ('train', 'val', 'test'))
    else:
        split_info_matches = None
    return dict(raw=dict(label_files=len(raw_files), rows=len(raw), unique_sources=len(set(raw_keys)),
                         users=len({r['user_id'] for _, r in raw}), sessions=len(by_session),
                         devices=len({r['device_id'] for _, r in raw}),
                         valid=sum(r['valid'] == '1' for _, r in raw),
                         missing_images=len(raw) - len(raw_images), valid_missing_images=raw_valid_missing,
                         invalid_missing_images=raw_invalid_missing,
                         file_types=raw_file_types, timestamp_nonincreasing=time_anomalies,
                         target_elapsed_invalid=target_elapsed_invalid,
                         target_ids=len({r['target_id'] for _, r in raw if 'target_id' in r}),
                         max_user_labels_per_session=max((len({r['user_id'] for r in rows}) for rows in by_session.values()), default=0),
                         max_device_labels_per_session=max((len({r['device_id'] for r in rows}) for rows in by_session.values()), default=0),
                         duplicate_sources=len(raw_keys) - len(set(raw_keys)),
                         screen_sizes=dict(Counter((r['screen_w'] + 'x' + r['screen_h']) for _, r in raw)),
                         frame_sizes=dict(Counter((r['frame_w'] + 'x' + r['frame_h']) for _, r in raw)),
                         columns=list(raw[0][1]) if raw else []),
                processed=dict(label_files=len(processed_files), rows=len(processed),
                               unique_sources=len(set(processed_keys)),
                               matched_raw=sum(k in source for k in processed_keys),
                               unmatched_raw=sum(k not in source for k in processed_keys),
                               missing_images=missing_processed_images, file_types=processed_file_types,
                               duplicate_sources=len(processed_keys) - len(set(processed_keys)),
                               norm_target_mismatch=norm_mismatch, user_label_mismatch=user_mismatch,
                               copied_head_pose_mismatch=pose_copy_mismatch,
                               split_info_matches=split_info_matches,
                               split_rows=dict(Counter(s for s, _, _ in processed)),
                               split_source_overlap={a + '_' + b: len(split_keys[a] & split_keys[b]) for a, b in paired_splits},
                               split_session_overlap={a + '_' + b: len(split_sessions[a] & split_sessions[b]) for a, b in paired_splits},
                               split_user_overlap={a + '_' + b: len(split_users[a] & split_users[b]) for a, b in paired_splits},
                               columns=list(processed[0][2]) if processed else []),
                sampled_images=dict(raw_pairs=len(sampled_raw), raw_shapes=dict(raw_layout),
                                    raw_blank_right_tiles=right_blank,
                                    processed_pairs=len(sampled_processed), processed_shapes=dict(processed_layout),
                                    processed_near_flipped_left=near_flip),
                source_hashes=dict(raw_labels=[hash_file(p) for p in raw_files],
                                   processed_labels=[hash_file(p) for p in processed_files],
                                   split_info=hash_file(split_info_path) if split_info_path.exists() else None))


def audit_r3_sessions(directory):
    """Metadata/event counts only; generated analysis directories are never sessions."""
    directory = Path(directory)
    totals, candidates = Counter(), []
    for child in sorted(directory.iterdir()):
        if not (child / 'session.json').is_file():
            continue
        meta, events, issues = read_session(child)
        if meta.get('analysis_run'):
            continue
        totals['sessions'] += 1
        totals['synthetic' if meta.get('synthetic') else 'real'] += 1
        totals['complete' if meta.get('complete') and not issues else 'incomplete'] += 1
        selection = meta.get('plan', {}).get('selection', 'unknown')
        totals['protocol_' + selection] += 1
        if not meta.get('synthetic'):
            candidates.append(dict(session_sha256=hash_file(child / 'session.json'),
                                   complete=bool(meta.get('complete') and not issues),
                                   backend=meta.get('backend'), protocol=selection,
                                   plan_segments=len(meta.get('plan', {}).get('segments', [])),
                                   painted=sum(e['kind'] == 'target_painted' for e in events),
                                   producers=sum(e['kind'] == 'producer' for e in events),
                                   pauses=sum(e['kind'] == 'pause' for e in events),
                                   skips=sum(e['kind'] == 'skip' for e in events)))
    return dict(counts=dict(totals), real_candidates=candidates,
                file_types=dict(Counter(p.suffix.lower() for p in directory.rglob('*') if p.is_file()
                                        and 'r4_runs' not in p.parts)))
